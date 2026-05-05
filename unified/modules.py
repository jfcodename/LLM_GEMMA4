"""
Gemma 4 E4B — Módulos de Otimização

Implementações PyTorch dos blocos modificados:
  1. SparsityPredictor     — reutiliza per_layer_input_gate como preditor neuronal
  2. ReLU2GatedMLP        — substitui GELUTanh por ReLU² com mascaramento esparso
  3. GatedAttentionLayer  — gate sigmoid após SDPA (NeurIPS 2025)
  4. SnapKVCache          — KV cache com eviction por score de atenção
  5. SparsityMasks        — geradores de máscara 2:4 e 6:8

Adaptados para E4B: suporte a KV sharing (layers 24-41) e PLE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import Gemma4E4BConfig, ReLU2Config, GatedAttentionConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPARSITY PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class SparsityPredictor(nn.Module):
    """
    Converte o per_layer_input_gate do E4B em preditor de esparsidade MLP.

    Fluxo:
        x (B, T, 2560)
        → gate_in: Linear(2560 → 256)    [pesos reutilizados do checkpoint]
        → RMSNorm
        → score_proj: Linear(256 → 10240) [nova, inicializada leve]
        → topk → mask binário (B, T, 10240)
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        gate_bottleneck_size: int = 256,
        intermediate_size: int = 10240,
        config: ReLU2Config = None,
        gate_weight_in: Optional[torch.Tensor] = None,
        gate_norm_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_size = gate_bottleneck_size
        self.intermediate_size = intermediate_size
        self.config = config or ReLU2Config()

        # Gate de entrada (reutiliza pesos do per_layer_input_gate)
        self.gate_in = nn.Linear(hidden_size, gate_bottleneck_size, bias=False)
        self.gate_norm = nn.RMSNorm(gate_bottleneck_size)

        if gate_weight_in is not None:
            self.gate_in.weight.data.copy_(gate_weight_in)
        if gate_norm_weight is not None:
            self.gate_norm.weight.data.copy_(gate_norm_weight)

        # Projeção: bottleneck → intermediate (nova, treinável)
        self.score_proj = nn.Linear(gate_bottleneck_size, intermediate_size, bias=False)
        nn.init.normal_(self.score_proj.weight, std=0.02)

        # Top-k
        if self.config.topk_neurons is not None:
            self.topk = self.config.topk_neurons
        else:
            self.topk = int((1.0 - self.config.sparsity_target) * intermediate_size)

        # Estatísticas
        self.register_buffer("running_sparsity", torch.zeros(1), persistent=False)
        self._n_steps = 0

    @property
    def actual_sparsity(self) -> float:
        return self.running_sparsity.item() if self._n_steps > 0 else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna mask (B, T, intermediate_size) com 1.0 nos neurônios ativos."""
        gate_out = self.gate_norm(self.gate_in(x))           # (B, T, 256)
        scores = self.score_proj(gate_out)                   # (B, T, 10240)

        _, topk_idx = torch.topk(scores, self.topk, dim=-1, sorted=False)

        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)

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
# ─────────────────────────────────────────────────────────────────────────────

class ReLU2GatedMLP(nn.Module):
    """
    MLP esparso com ReLU² e mascaramento de neurônios.

    Original Gemma 4: gate_proj(x) ←GELUTanh→ hadamard up_proj(x) → down_proj
    Neo:              [mask * (ReLU²(gate_proj(x)) ⊙ up_proj(x))] → down_proj
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        intermediate_size: int = 10240,
        config: ReLU2Config = None,
        gate_proj_weight: Optional[torch.Tensor] = None,
        up_proj_weight: Optional[torch.Tensor] = None,
        down_proj_weight: Optional[torch.Tensor] = None,
        sparsity_predictor: Optional[SparsityPredictor] = None,
    ):
        super().__init__()
        self.config = config or ReLU2Config()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        if gate_proj_weight is not None:
            self.gate_proj.weight.data.copy_(gate_proj_weight)
        if up_proj_weight is not None:
            self.up_proj.weight.data.copy_(up_proj_weight)
        if down_proj_weight is not None:
            self.down_proj.weight.data.copy_(down_proj_weight)

        self.sparsity_predictor = sparsity_predictor

        # Threshold (fallback)
        self.register_buffer(
            "activation_threshold", torch.zeros(1), persistent=True
        )

    @staticmethod
    def relu2(x: torch.Tensor) -> torch.Tensor:
        """ReLU²(x) = max(0, x)². Zeros exatos."""
        return F.relu(x).square()

    def forward(
        self,
        x: torch.Tensor,
        predictor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate = self.relu2(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        if predictor_mask is not None:
            hidden = hidden * predictor_mask
        elif not self.config.use_gate_as_predictor:
            threshold = self.activation_threshold.item()
            if threshold > 0:
                mask = (gate.abs() > threshold).float()
                hidden = hidden * mask

        out = self.down_proj(hidden)
        return out

    @torch.no_grad()
    def calibrate_threshold(
        self, calibration_data: torch.Tensor, percentile: float = 35.0
    ) -> float:
        """Estima threshold por percentil das pré-ativações não-zero.
        
        ReLU² já zera ~50% das ativações. O percentil é aplicado sobre os
        valores NÃO-zero para determinar um threshold adicional significativo.
        """
        gate_acts = self.relu2(self.gate_proj(calibration_data))
        nonzero_acts = gate_acts[gate_acts > 0]
        if nonzero_acts.numel() == 0:
            self.activation_threshold.fill_(0.0)
            return 1.0
        threshold = torch.quantile(nonzero_acts.float(), percentile / 100.0)
        self.activation_threshold.fill_(threshold.item())
        actual_sparsity = (gate_acts.abs() <= threshold).float().mean().item()
        return actual_sparsity


# ─────────────────────────────────────────────────────────────────────────────
# 2.5 PRUNED MLP (PODA ESTRUTURAL FÍSICA)
# ─────────────────────────────────────────────────────────────────────────────

class PrunedMLP(nn.Module):
    """
    MLP com dimensão intermediate fisicamente reduzida (Pruning Estrutural).

    Pesos são SUBCONJUNTOS dos pesos originais:
      gate_proj.weight: (keep_neurons, hidden_size)    ← linhas selecionadas
      up_proj.weight:   (keep_neurons, hidden_size)    ← linhas selecionadas
      down_proj.weight: (hidden_size, keep_neurons)    ← colunas selecionadas

    Resultado: forward pass faz matmuls menores → speedup real em hardware.
    """

    def __init__(
        self,
        hidden_size: int,
        kept_neurons: int,
        gate_proj_weight: Optional[torch.Tensor] = None,
        up_proj_weight: Optional[torch.Tensor] = None,
        down_proj_weight: Optional[torch.Tensor] = None,
        use_relu2: bool = True,
        neuron_indices: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = kept_neurons
        self.use_relu2 = use_relu2

        self.gate_proj = nn.Linear(hidden_size, kept_neurons, bias=False)
        self.up_proj   = nn.Linear(hidden_size, kept_neurons, bias=False)
        self.down_proj = nn.Linear(kept_neurons, hidden_size, bias=False)

        if gate_proj_weight is not None:
            self.gate_proj.weight.data.copy_(gate_proj_weight)
        if up_proj_weight is not None:
            self.up_proj.weight.data.copy_(up_proj_weight)
        if down_proj_weight is not None:
            self.down_proj.weight.data.copy_(down_proj_weight)

        if neuron_indices is not None:
            self.register_buffer("kept_neuron_indices", neuron_indices, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.use_relu2:
            gate = F.relu(gate).square()
        else:
            gate = F.gelu(gate, approximate='tanh')
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ─────────────────────────────────────────────────────────────────────────────
# 2.6 DEEPSLICE MOE (DEEPSEEK + SLICEGPT + MAMBA-2 ROUTING)
# ─────────────────────────────────────────────────────────────────────────────

class Mamba2Router(nn.Module):
    """
    Roteador SSM (State Space Model) minimalista inspirado no Mamba-2.
    Resolve a 'Miopia de Token' criando um contexto contínuo temporal da frase
    antes de prever os logits dos especialistas. Ultra-leve e com tempo linear.
    """
    def __init__(self, hidden_size: int, num_experts: int, d_state: int = 16, num_experts_per_tok: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, d_state, bias=False)
        
        # 1D Depthwise Conv (mistura contexto localmente)
        self.conv1d = nn.Conv1d(
            in_channels=d_state, out_channels=d_state, 
            kernel_size=3, padding=2, groups=d_state
        )
        
        # Parâmetros da Recorrência SSM (Decaimento e Entrada)
        # Log-space para garantir que A fique entre [0, 1] pós-exp
        self.A_log = nn.Parameter(torch.log(torch.rand(d_state) * 0.1 + 0.9))
        self.B = nn.Linear(d_state, d_state, bias=False)
        
        # DeepSeek-V3 Style: Bias adaptativo para load balancing sem perda auxiliar
        self.out_proj = nn.Linear(d_state, num_experts, bias=True)
        self.register_buffer("adaptive_bias", torch.zeros(num_experts))
        
        # Inicialização neutra
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
        self.act = nn.SiLU()

    def update_bias(self, load, target_load, gamma=0.01):
        """
        Atualiza o bias adaptativo baseado no load real dos experts.
        load: (num_experts) - fração de tokens processados por cada expert no batch
        """
        with torch.no_grad():
            # Se load < target, aumenta o bias para atrair mais tokens
            # Se load > target, diminui o bias para repelir tokens
            self.adaptive_bias += gamma * (target_load - load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # 1. Compressão para espaço de estado
        u = self.in_proj(x) # (B, T, d_state)
        
        # 2. Mistura Convolucional (Causal)
        u_conv = u.transpose(1, 2) # (B, d_state, T)
        
        # FIX: causal padding manual correto
        u_conv = F.pad(u_conv, (2, 0)) # pad left
        
        # FIX: Evitar bug do cuDNN em GPUs Turing (T4) para Depthwise Conv1d com bfloat16
        orig_dtype = u_conv.dtype
        w = self.conv1d.weight.to(torch.float32)
        b = self.conv1d.bias.to(torch.float32) if self.conv1d.bias is not None else None
        
        # Como já fizemos o pad manual acima, o padding interno tem que ser 0
        u_conv = F.conv1d(
            u_conv.to(torch.float32), 
            w, b, 
            padding=0, 
            groups=self.conv1d.groups
        ).to(orig_dtype)
        
        u_conv = u_conv[:, :, :T] # fatiar o padding causal
        u_conv = self.act(u_conv.transpose(1, 2)) # (B, T, d_state)
        
        # 3. Recorrência Linear SSM (Construção do Cérebro Contextual)
        A = torch.exp(self.A_log) # (d_state,)
        B_u = self.B(u_conv)      # (B, T, d_state)
        
        state = torch.zeros(B, A.size(0), device=x.device, dtype=x.dtype)
        out_states = []
        for t in range(T):
            state = state * A + B_u[:, t, :]
            out_states.append(state)
            
        ssm_out = torch.stack(out_states, dim=1) # (B, T, d_state)
        
        # 4. Logits de Roteamento baseados no contexto
        return self.out_proj(ssm_out)


class DeepSliceMoE(nn.Module):
    """
    Arquitetura Fina MoE inspirada no DeepSeek V2/V3 com Shared Experts.
    Substitui a MLP original com perda zero de parâmetros.
    """
    def __init__(
        self,
        hidden_size: int,
        shared_expert: nn.Module,
        routed_experts: nn.ModuleList,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.shared_expert = shared_expert
        self.routed_experts = routed_experts
        self.num_experts_per_tok = min(num_experts_per_tok, len(routed_experts))
        
        self.router = Mamba2Router(hidden_size, len(routed_experts), num_experts_per_tok=self.num_experts_per_tok)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Auto-casting dinâmico para contornar bloqueios de offload da 'accelerate'
        if self.shared_expert.gate_proj.weight.device != x.device:
            self.to(x.device)
            
        B, T, D = x.shape
        
        # 1. Especialista Compartilhado (Sempre roda, Inteligência Core)
        shared_out = self.shared_expert(x)
        
        if len(self.routed_experts) == 0 or self.num_experts_per_tok == 0:
            return shared_out
            
        # 2. Roteador Mamba-2 (DeepSeek-V3 Style)
        router_logits = self.router(x)
        # s_ij = Sigmoid(u_i * e_j)
        scores = torch.sigmoid(router_logits)
        
        # Seleção usa o bias adaptativo (score_ij + b_j), mas o peso da soma NÃO usa
        selection_scores = scores + self.router.adaptive_bias
        
        top_k_scores, top_k_indices = torch.topk(
            selection_scores, self.num_experts_per_tok, dim=-1
        )
        
        # Pesos reais para a soma (extraídos dos scores originais, sem o bias de balanceamento)
        top_k_weights = scores.gather(-1, top_k_indices)
        
        # Normaliza apenas os pesos selecionados para manter escala de ativação
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 3. Forward Pass Esparso nos Routed Experts
        x_flat = x.view(-1, D)
        indices_flat = top_k_indices.view(-1, self.num_experts_per_tok)
        weights_flat = top_k_weights.view(-1, self.num_experts_per_tok)
        out_flat = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.routed_experts):
            mask = (indices_flat == i)
            token_idx, weight_idx = torch.where(mask)
            if len(token_idx) == 0:
                continue
                
            expert_tokens = x_flat[token_idx]
            expert_out = expert(expert_tokens)
            
            # Multiplicamos pelo peso do roteador correspondente (Diferenciabilidade)
            # weights_flat tem shape (B*T, K)
            current_weights = weights_flat[token_idx, weight_idx].unsqueeze(-1)
            out_flat[token_idx] += expert_out * current_weights
            
        routed_out = out_flat.view(B, T, D)
        
        # 4. Soma Final (Core + Especialistas Raros)
        return shared_out + routed_out
# ─────────────────────────────────────────────────────────────────────────────
# 3. SNAP KV CACHE
# ─────────────────────────────────────────────────────────────────────────────

class SnapKVCache(nn.Module):
    """
    KV cache com eviction baseado em importância.

    Mantém top-k tokens + janela recente. Para os layers globais.
    """

    def __init__(self, max_capacity: int = 4096, window: int = 1024):
        super().__init__()
        self.max_capacity = max_capacity
        self.window = window
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None
        self.attn_score_accum: Optional[torch.Tensor] = None
        self.is_compressed = False

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache is None:
            self.key_cache = keys
            self.value_cache = values
            if attn_weights is not None:
                self.attn_score_accum = attn_weights.mean(dim=2)
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
                self.attn_score_accum += new_scores

        T = self.key_cache.shape[2]
        if T > self.max_capacity and not self.is_compressed:
            self._compress()

        return self.key_cache, self.value_cache

    def _compress(self):
        T = self.key_cache.shape[2]
        recent_start = max(0, T - self.window)
        historical_end = recent_start

        if historical_end <= 0 or self.attn_score_accum is None:
            return

        scores_hist = self.attn_score_accum[..., :historical_end]
        keep_budget = self.max_capacity - self.window
        if keep_budget <= 0:
            keep_budget = self.max_capacity // 2

        scores_avg = scores_hist.mean(dim=1)
        k = min(keep_budget, historical_end)
        _, topk_idx = torch.topk(scores_avg, k, dim=-1, sorted=True)
        topk_idx_sorted = topk_idx.sort(dim=-1).values

        B, H, _, D = self.key_cache.shape
        idx_expanded = topk_idx_sorted.unsqueeze(1).unsqueeze(-1).expand(B, H, k, D)
        keys_kept = torch.gather(self.key_cache[..., :historical_end, :], 2, idx_expanded)
        vals_kept = torch.gather(self.value_cache[..., :historical_end, :], 2, idx_expanded)

        keys_recent = self.key_cache[..., recent_start:, :]
        vals_recent = self.value_cache[..., recent_start:, :]

        self.key_cache = torch.cat([keys_kept, keys_recent], dim=2)
        self.value_cache = torch.cat([vals_kept, vals_recent], dim=2)
        self.attn_score_accum = None
        self.is_compressed = True

    def reset(self):
        self.key_cache = None
        self.value_cache = None
        self.attn_score_accum = None
        self.is_compressed = False

    @property
    def current_size(self) -> int:
        return self.key_cache.shape[2] if self.key_cache is not None else 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. GATED ATTENTION LAYER
# ─────────────────────────────────────────────────────────────────────────────

class GatedAttentionLayer(nn.Module):
    """
    Gate sigmoid per-head aplicado após SDPA.

    Y_out = Y_attn ⊙ σ(X · Wθ)

    Adapta-se a shared KV: quando a layer não tem k/v_proj próprios,
    recebe k/v como argumentos em vez de calcular internamente.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        config: GatedAttentionConfig = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.config = config or GatedAttentionConfig()

        # Gate sigmoid per-head
        self.gate_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        if self.config.init_ones:
            nn.init.zeros_(self.gate_proj.weight)
        else:
            nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)

    def compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """σ(X·Wθ) reshaped para (B, num_heads, T, head_dim)."""
        B, T, _ = x.shape
        gate = torch.sigmoid(self.gate_proj(x))
        gate = gate.view(B, T, self.num_heads, self.head_dim)
        gate = gate.transpose(1, 2)
        return gate

    def apply_gate(
        self,
        attn_output: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Aplica gate na saída de atenção."""
        gate = self.compute_gate(x)
        return attn_output * gate


# ─────────────────────────────────────────────────────────────────────────────
# 5. SPARSITY MASKS (2:4 e 6:8)
# ─────────────────────────────────────────────────────────────────────────────

def magnitude_mask_24(weight: torch.Tensor) -> torch.Tensor:
    """
    Máscara 2:4: para cada grupo de 4 pesos consecutivos,
    mantém os 2 de maior magnitude e zera os outros.
    """
    assert weight.dim() == 2, f"Esperado 2D, recebeu {weight.dim()}D"
    rows, cols = weight.shape
    assert cols % 4 == 0, f"in_features ({cols}) deve ser múltiplo de 4"

    w = weight.abs().view(rows, -1, 4)
    _, indices = torch.topk(w, k=2, dim=-1)
    mask = torch.zeros_like(w, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask.view(rows, cols)


def magnitude_mask_68(weight: torch.Tensor) -> torch.Tensor:
    """
    Máscara 6:8: para cada grupo de 8 pesos,
    mantém os 6 de maior magnitude (25% zeros).
    """
    assert weight.dim() == 2, f"Esperado 2D, recebeu {weight.dim()}D"
    rows, orig_cols = weight.shape
    pad = (8 - orig_cols % 8) % 8
    if pad > 0:
        weight = F.pad(weight, (0, pad))

    rows, cols_padded = weight.shape
    w = weight.abs().view(rows, -1, 8)
    _, zero_indices = torch.topk(w, k=2, dim=-1, largest=False)
    mask = torch.ones_like(w, dtype=torch.bool)
    mask.scatter_(-1, zero_indices, False)
    mask = mask.view(rows, cols_padded)

    return mask[:, :orig_cols] if pad > 0 else mask


def check_dim_eligibility(weight: torch.Tensor, mode: str) -> Tuple[bool, str]:
    """Verifica se as dimensões são compatíveis com o modo de esparsidade."""
    if weight.dim() != 2:
        return False, f"Não é 2D (shape={weight.shape})"

    rows, cols = weight.shape

    if mode == "2:4":
        if cols % 4 != 0:
            return False, f"in_features={cols} não é múltiplo de 4"
        if rows < 64 or cols < 64:
            return False, f"Dims muito pequenas ({rows}×{cols})"
    elif mode == "6:8":
        if cols % 8 != 0:
            return False, f"in_features={cols} não é múltiplo de 8"
        if rows < 64:
            return False, f"out_features={rows} muito pequeno"

    return True, ""
