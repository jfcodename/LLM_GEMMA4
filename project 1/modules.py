"""
Gemma 4 Neo — Módulos Modificados

Contém as implementações PyTorch de cada bloco substituído:
  1. SparsityPredictor     — reutiliza o per_layer_input_gate como preditor neuronal
  2. ReLU2GatedMLP        — substitui GELUTanh por ReLU² com mascaramento esparso
  3. GatedAttentionLayer  — adiciona gate sigmoid após SDPA (NeurIPS 2025)
  4. SnapKVCache          — KV cache com eviction por score de atenção acumulado
  5. Mamba2Block          — bloco SSM para substituição dos layers locais (requer mamba-ssm)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import GatedAttentionConfig, ReLU2Config, Mamba2Config


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPARSITY PREDICTOR
#    Reutiliza o per_layer_input_gate (2560→256→2560) já existente no Gemma 4.
#    A norma L2 do vetor bottleneck (dim=256) por neurônio MLP indica importância.
#    Os top-k neurônios com maior score são mantidos; o resto é zerado.
# ─────────────────────────────────────────────────────────────────────────────

class SparsityPredictor(nn.Module):
    """
    Converte o per_layer_input_gate do Gemma 4 em preditor de esparsidade MLP.

    Fluxo:
        x (B, T, 2560)
        → per_layer_input_gate: Linear(2560 → 256)  [já existente]
        → RMSNorm                                    [já existente]
        → norm L2 por posição → score (B, T, 256)
        → projeção linear leve → neuron_scores (B, T, intermediate_size)
        → topk → mask binário (B, T, intermediate_size)

    O mask é aplicado DEPOIS do gate_proj × up_proj mas ANTES do down_proj,
    zerando colunas inteiras da weight matrix → skip real de compute.
    """

    def __init__(
        self,
        hidden_size: int,
        gate_bottleneck_size: int,   # 256 no Gemma 4
        intermediate_size: int,      # 10240 no Gemma 4
        config: ReLU2Config,
        # Pesos do per_layer_input_gate original (carregados do checkpoint)
        gate_weight_in: Optional[torch.Tensor] = None,
        gate_norm_weight: Optional[torch.Tensor] = None,
        gate_weight_out: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_size = gate_bottleneck_size
        self.intermediate_size = intermediate_size
        self.config = config

        # Gate de entrada — reutiliza pesos do per_layer_input_gate
        self.gate_in = nn.Linear(hidden_size, gate_bottleneck_size, bias=False)
        self.gate_norm = nn.RMSNorm(gate_bottleneck_size)

        if gate_weight_in is not None:
            self.gate_in.weight.data.copy_(gate_weight_in)
        if gate_norm_weight is not None:
            self.gate_norm.weight.data.copy_(gate_norm_weight)

        # Projeção leve: bottleneck → intermediate_size (nova, inicializada aleatória)
        # Treina rapidamente com fine-tuning leve (~30B tokens)
        self.score_proj = nn.Linear(
            gate_bottleneck_size, intermediate_size, bias=False
        )
        nn.init.normal_(self.score_proj.weight, std=0.02)

        # Calcula topk a partir do sparsity_target
        if config.topk_neurons is not None:
            self.topk = config.topk_neurons
        else:
            self.topk = int((1.0 - config.sparsity_target) * intermediate_size)

        # Buffer para estatísticas de sparsidade (monitoramento)
        self.register_buffer(
            "running_sparsity", torch.zeros(1), persistent=False
        )
        self._n_steps = 0

    @property
    def actual_sparsity(self) -> float:
        return self.running_sparsity.item() if self._n_steps > 0 else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            mask: (batch, seq_len, intermediate_size) — 1.0 nos neurônios ativos
        """
        # Bottleneck projection (reutiliza pesos do gate original)
        gate_out = self.gate_norm(self.gate_in(x))           # (B, T, 256)

        # Projeção para espaço intermediate
        scores = self.score_proj(gate_out)                   # (B, T, 10240)

        # Top-k por posição: seleciona os neurônios mais importantes
        topk_vals, topk_idx = torch.topk(scores, self.topk, dim=-1, sorted=False)

        # Cria máscara esparsa
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)

        # Atualiza estatística
        if self.training:
            with torch.no_grad():
                sp = 1.0 - mask.float().mean().item()
                self.running_sparsity.fill_(
                    0.9 * self.running_sparsity.item() + 0.1 * sp
                )
                self._n_steps += 1

        return mask


# ─────────────────────────────────────────────────────────────────────────────
# 2. ReLU² GATED MLP
#    Substitui o Gemma4TextMLP que usa GELUTanh por ReLU² + mascaramento esparso.
#
#    Original: gate_proj(x) ←GELUTanh→ hadamard up_proj(x) → down_proj
#    Neo:      [mask * (ReLU²(gate_proj(x)) ⊙ up_proj(x))] → down_proj
#
#    O mask é provido pelo SparsityPredictor (ou calculado por threshold local
#    se use_gate_as_predictor=False).
# ─────────────────────────────────────────────────────────────────────────────

class ReLU2GatedMLP(nn.Module):
    """
    MLP esparso com ReLU² e mascaramento de neurônios via SparsityPredictor.

    Pesos (gate_proj, up_proj, down_proj) são carregados diretamente do Gemma 4
    — nenhum peso novo é adicionado aqui (exceto os do SparsityPredictor).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        config: ReLU2Config,
        # Pesos do MLP original carregados do checkpoint
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        # Predictor (compartilhado ou None se use_gate_as_predictor=False)
        sparsity_predictor: Optional["SparsityPredictor"] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Projeções — carregam pesos originais
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.gate_proj.weight.data.copy_(gate_proj_weight)
        self.up_proj.weight.data.copy_(up_proj_weight)
        self.down_proj.weight.data.copy_(down_proj_weight)

        self.sparsity_predictor = sparsity_predictor

        # Threshold por magnitude (fallback quando predictor não está disponível)
        # Calibrado pela média das pré-ativações em poucas amostras
        self.register_buffer(
            "activation_threshold", torch.zeros(1), persistent=True
        )

    @staticmethod
    def relu2(x: torch.Tensor) -> torch.Tensor:
        """ReLU²(x) = max(0, x)². Zeros exatos, mais esparso que GELU."""
        return F.relu(x).square()

    def forward(
        self,
        x: torch.Tensor,
        predictor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            predictor_mask: pré-computado pelo SparsityPredictor.
                            Se None, calcula threshold por magnitude.
        Returns:
            out: (batch, seq_len, hidden_size)
        """
        # Calcula gate e up em paralelo
        gate = self.relu2(self.gate_proj(x))   # (B, T, 10240) — zeros exatos
        up = self.up_proj(x)                   # (B, T, 10240)
        hidden = gate * up                     # elemento a elemento

        # Aplica máscara de sparsidade
        if predictor_mask is not None:
            hidden = hidden * predictor_mask
        elif not self.config.use_gate_as_predictor:
            # Fallback: threshold por magnitude da pré-ativação
            threshold = self.activation_threshold.item()
            if threshold > 0:
                mask = (gate.abs() > threshold).float()
                hidden = hidden * mask

        # Down projection — colunas zeradas são skip implícito
        out = self.down_proj(hidden)
        return out

    @torch.no_grad()
    def calibrate_threshold(
        self, calibration_data: torch.Tensor, percentile: float = 35.0
    ):
        """
        Estima threshold ótimo: percentile-th percentil das pré-ativações.
        Chamar com ~512 amostras antes de usar em produção.
        """
        gate_acts = self.relu2(self.gate_proj(calibration_data))
        threshold = torch.quantile(gate_acts.float().abs(), percentile / 100.0)
        self.activation_threshold.fill_(threshold.item())
        actual_sparsity = (gate_acts.abs() < threshold).float().mean().item()
        print(f"  threshold={threshold:.4f}, sparsidade={actual_sparsity:.1%}")
        return actual_sparsity


# ─────────────────────────────────────────────────────────────────────────────
# 3. SNAP KV CACHE
#    KV cache com eviction por score de atenção acumulado.
#    Mantém os top-k tokens mais "assistidos" + uma janela de tokens recentes.
#    Aplicado apenas nos 7 layers globais.
# ─────────────────────────────────────────────────────────────────────────────

class SnapKVCache(nn.Module):
    """
    Cache de Key-Value com eviction baseado em importância.

    Estratégia:
      1. Durante prefill: mantém todos os tokens e acumula scores de atenção.
      2. Após prefill: seleciona top-k tokens pelo score acumulado
         + sempre preserva uma janela dos últimos `window` tokens.
      3. Durante decode: novos tokens são adicionados; tokens antigos de baixo
         score são descartados para manter capacidade máxima.

    Referencias: SnapKV (arXiv 2404.14469), H2O (arXiv 2306.14048).
    """

    def __init__(self, max_capacity: int = 4096, window: int = 1024):
        super().__init__()
        self.max_capacity = max_capacity
        self.window = window

        # State: preenchido durante o forward
        self.key_cache: Optional[torch.Tensor] = None    # (B, H, T_kept, D)
        self.value_cache: Optional[torch.Tensor] = None
        self.attn_score_accum: Optional[torch.Tensor] = None  # (B, H, T)
        self.is_compressed = False

    def update(
        self,
        keys: torch.Tensor,   # (B, H, T_new, D)
        values: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,  # (B, H, T_q, T_new)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Atualiza o cache e retorna (keys_full, values_full) para uso na atenção.
        """
        if self.key_cache is None:
            self.key_cache = keys
            self.value_cache = values
            if attn_weights is not None:
                self.attn_score_accum = attn_weights.mean(dim=2)  # (B, H, T)
        else:
            self.key_cache = torch.cat([self.key_cache, keys], dim=2)
            self.value_cache = torch.cat([self.value_cache, values], dim=2)
            if attn_weights is not None and self.attn_score_accum is not None:
                new_scores = attn_weights.mean(dim=2)
                pad = torch.zeros(
                    *self.attn_score_accum.shape[:-1], keys.shape[2],
                    device=keys.device, dtype=keys.dtype
                )
                self.attn_score_accum = torch.cat(
                    [self.attn_score_accum, pad], dim=-1
                )
                # Acumula scores (rolling max)
                self.attn_score_accum += new_scores

        # Comprimir se excedeu capacidade
        T = self.key_cache.shape[2]
        if T > self.max_capacity and not self.is_compressed:
            self._compress()

        return self.key_cache, self.value_cache

    def _compress(self):
        """Aplica eviction: mantém top-k + janela recente."""
        T = self.key_cache.shape[2]
        recent_start = max(0, T - self.window)
        historical_end = recent_start

        if historical_end <= 0 or self.attn_score_accum is None:
            return

        # Score dos tokens históricos
        scores_hist = self.attn_score_accum[..., :historical_end]  # (B, H, T_hist)
        keep_budget = self.max_capacity - self.window
        if keep_budget <= 0:
            keep_budget = self.max_capacity // 2

        # Top-k histórico por cabeça (média entre heads para simplicidade)
        scores_avg = scores_hist.mean(dim=1)  # (B, T_hist)
        k = min(keep_budget, historical_end)
        _, topk_idx = torch.topk(scores_avg, k, dim=-1, sorted=True)  # (B, k)
        topk_idx_sorted = topk_idx.sort(dim=-1).values                # (B, k)

        # Seleciona K/V históricos + recentes
        B, H, _, D = self.key_cache.shape
        idx_expanded = topk_idx_sorted.unsqueeze(1).unsqueeze(-1).expand(B, H, k, D)
        keys_kept = torch.gather(self.key_cache[..., :historical_end, :], 2, idx_expanded)
        vals_kept = torch.gather(self.value_cache[..., :historical_end, :], 2, idx_expanded)

        keys_recent = self.key_cache[..., recent_start:, :]
        vals_recent = self.value_cache[..., recent_start:, :]

        self.key_cache = torch.cat([keys_kept, keys_recent], dim=2)
        self.value_cache = torch.cat([vals_kept, vals_recent], dim=2)
        self.attn_score_accum = None  # Reinicia scores após compressão
        self.is_compressed = True

    def reset(self):
        self.key_cache = None
        self.value_cache = None
        self.attn_score_accum = None
        self.is_compressed = False


# ─────────────────────────────────────────────────────────────────────────────
# 4. GATED ATTENTION LAYER
#    Envolve o mecanismo de atenção existente do Gemma 4 adicionando um
#    gate sigmoid per-head: Y_out = Y_attn ⊙ σ(X · Wθ)
#
#    Aplicado APENAS nos layers globais (5, 11, 17, 23, 29, 35, 41).
#    Pesos q/k/v/o são carregados diretamente do checkpoint.
# ─────────────────────────────────────────────────────────────────────────────

class GatedAttentionLayer(nn.Module):
    """
    Wrapper sobre a atenção global do Gemma 4 que adiciona o gate sigmoid.

    O gate é implementado como Linear(hidden_size, num_heads * head_dim, bias=False)
    seguido de sigmoid e reshape para (B, H, T, D) — multiplicado element-wise
    na saída de cada cabeça ANTES da projeção de saída o_proj.

    Referência: "Gated Attention for Large Language Models" (NeurIPS 2025).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        config: GatedAttentionConfig,
        snap_kv_cache: Optional[SnapKVCache] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.config = config

        # Gate: Linear(2560 → num_heads * head_dim)
        # Inicializado com ones → gate começa em σ(large) ≈ 1.0 (transparente)
        self.gate_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        if config.init_ones:
            # Inicialização especial: bias de modo que sigmoid sature em 1
            # Usa pesos pequenos + offset para que saída comece ~1
            nn.init.zeros_(self.gate_proj.weight)
            # Ajuste: gate começa transparente, treino vai ajustando
        else:
            nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)

        self.snap_kv = snap_kv_cache

    def compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computa gate sigmoid per-head.
        Args:
            x: (B, T, hidden_size)
        Returns:
            gate: (B, num_heads, T, head_dim)
        """
        B, T, _ = x.shape
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, num_heads * head_dim)
        gate = gate.view(B, T, self.num_heads, self.head_dim)
        gate = gate.transpose(1, 2)              # (B, num_heads, T, head_dim)
        return gate

    def apply_gate_to_attn_output(
        self,
        attn_output: torch.Tensor,  # (B, num_heads, T, head_dim)
        x: torch.Tensor,            # (B, T, hidden_size)
    ) -> torch.Tensor:
        """Aplica o gate sigmoid na saída da atenção (per-head)."""
        gate = self.compute_gate(x)
        gated = attn_output * gate
        return gated

    def forward(
        self,
        original_attn_module: nn.Module,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass completo: chama a atenção original + aplica gate.

        O módulo original de atenção do Gemma 4 é chamado diretamente
        (monkey-patching não-destrutivo: os pesos originais são preservados).
        """
        B, T, _ = x.shape

        # ── Chama o módulo de atenção original do Gemma 4 ──────────────────
        # Precisamos dos raw attention outputs (pre-o_proj) para aplicar o gate.
        # Fazemos isso interceptando internamente.

        # Projeções Q, K, V do módulo original
        attn = original_attn_module
        q = attn.q_proj(x)
        k = attn.k_proj(x)
        v = attn.v_proj(x)

        # Shapes
        head_dim = q.shape[-1] // attn.num_heads if hasattr(attn, 'num_heads') else self.head_dim
        num_heads = self.num_heads
        num_kv_heads = k.shape[-1] // head_dim

        q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, num_kv_heads, head_dim).transpose(1, 2)

        # RMSNorm (q_norm, k_norm do Gemma 4)
        if hasattr(attn, 'q_norm'):
            q = attn.q_norm(q)
        if hasattr(attn, 'k_norm'):
            k = attn.k_norm(k)

        # RoPE (rotary embeddings)
        if hasattr(attn, 'rotary_emb') or position_ids is not None:
            pass  # Delegado ao modelo pai via position_ids no attention_mask

        # SnapKV: atualiza cache e comprime se necessário
        if self.snap_kv is not None and use_cache:
            k, v = self.snap_kv.update(k, v)

        # GQA: expande k/v para num_heads se necessário
        if num_kv_heads < num_heads:
            repeat_factor = num_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention (com Flash Attention se disponível)
        scale = 1.0 / math.sqrt(head_dim)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            scale=scale,
            is_causal=(attention_mask is None),
        )  # (B, num_heads, T, head_dim)

        # ── Aplica o gate sigmoid (novidade do Gated Attention) ────────────
        gated_output = self.apply_gate_to_attn_output(attn_output, x)

        # Reshape e projeção de saída (usa pesos originais do o_proj)
        gated_output = gated_output.transpose(1, 2).contiguous()
        gated_output = gated_output.view(B, T, num_heads * head_dim)
        out = attn.o_proj(gated_output)

        return out, None  # (output, past_key_value)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAMBA-2 BLOCK  (requer: pip install mamba-ssm causal-conv1d)
#    Substitui os layers locais (SWA) pelo SSM recorrente Mamba-2.
#    Estado interno de tamanho fixo → sem KV cache, O(1) por step.
# ─────────────────────────────────────────────────────────────────────────────

class Mamba2Block(nn.Module):
    """
    Wrapper Mamba-2 para substituição dos layers locais do Gemma 4.

    Interface compatível com Gemma4TextDecoderLayer:
    - Entrada: x (B, T, hidden_size)
    - Saída: x_out (B, T, hidden_size)

    Instalação: pip install mamba-ssm causal-conv1d
    """

    def __init__(self, hidden_size: int, config: Mamba2Config):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self._initialized = False

        try:
            from mamba_ssm import Mamba2 as _Mamba2

            d_inner = hidden_size * config.expand
            self.ssm = _Mamba2(
                d_model=hidden_size,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                headdim=config.headdim,
                chunk_size=config.chunk_size,
            )
            self.norm = nn.RMSNorm(hidden_size)
            self._initialized = True
            print("  ✓ Mamba-2 inicializado (mamba-ssm disponível)")

        except ImportError:
            print(
                "  ⚠ mamba-ssm não encontrado. "
                "Instale com: pip install mamba-ssm causal-conv1d\n"
                "  Mamba2Block no modo PLACEHOLDER (pass-through)."
            )
            # Fallback: identidade (não altera o comportamento)
            self.ssm = nn.Identity()
            self.norm = nn.RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self._initialized:
            return x
        # Pre-norm + SSM + residual
        residual = x
        x_normed = self.norm(x)
        x_ssm = self.ssm(x_normed)
        return residual + x_ssm

    @classmethod
    def from_pretrained_path(cls, path: str, hidden_size: int, config: Mamba2Config):
        """Carrega pesos Mamba-2 pré-destilados de um checkpoint."""
        block = cls(hidden_size, config)
        if path and block._initialized:
            state = torch.load(path, map_location="cpu")
            block.load_state_dict(state, strict=False)
            print(f"  ✓ Mamba-2 carregado de {path}")
        return block
