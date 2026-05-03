"""
Gemma 4 E4B — Configuração Unificada

Parametriza a arquitetura E4B com detecção automática de:
- KV sharing (layers 24-41 sem k_proj/v_proj)
- Per-Layer Embeddings (PLE)
- Dimensões reais extraídas da printagem da arquitetura

Foco: CPU inference — prioriza redução de compute e memória.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, FrozenSet
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DA ARQUITETURA E4B
# ─────────────────────────────────────────────────────────────────────────────

# Layers globais (full-attention) — padrão 5 local : 1 global
GLOBAL_LAYER_INDICES: FrozenSet[int] = frozenset({5, 11, 17, 23, 29, 35, 41})

# Layers que compartilham KV cache (sem k_proj/v_proj próprio)
# A partir da layer 24, o E4B usa shared KV das layers anteriores
KV_SHARED_LAYER_START = 24
KV_SHARED_LAYERS: FrozenSet[int] = frozenset(range(24, 42))

# Layers com KV próprio (têm k_proj, v_proj, k_norm, v_norm)
KV_OWN_LAYERS: FrozenSet[int] = frozenset(range(0, 24))


class SparsityMode(Enum):
    """Modos de esparsidade."""
    DENSE        = "dense"
    SEMI_24      = "2:4"        # 50% zeros — requer Ampere+ para HW accel
    SEMI_68      = "6:8"        # 25% zeros — SlideSparse, preserva reasoning
    SKIP         = "skip"       # Excluído explicitamente


@dataclass
class ReLU2Config:
    """
    Configuração do MLP esparso com ReLU².

    ReLU²(x) = max(0, x)²  →  zeros exatos, mais esparso que GELU/SiLU.
    O per_layer_input_gate (já existente no E4B) é reutilizado como
    preditor de importância de neurônio.
    """
    enabled: bool = True
    sparsity_target: float = 0.65       # 65% dos neurônios zerados
    use_gate_as_predictor: bool = True  # Reutiliza per_layer_input_gate
    topk_neurons: Optional[int] = None  # None = auto (1-sparsity_target)*intermediate
    calibration_dataset: str = "wikitext"
    calibration_samples: int = 512


@dataclass
class GatedAttentionConfig:
    """
    Gated Attention — σ(X·Wθ) per-head após SDPA.
    Aplicado nos 7 layers globais. Adapta-se a shared KV (layers 29,35,41).
    """
    enabled: bool = True
    init_ones: bool = True              # Gate começa transparente
    snap_kv_enabled: bool = True
    snap_kv_window: int = 1024
    snap_kv_max_capacity: int = 4096


@dataclass
class WeightSparsityConfig:
    """
    Esparsidade estruturada de pesos (2:4 ou 6:8).
    """
    enabled: bool = True
    text_decoder_mode: SparsityMode = SparsityMode.SEMI_68  # 6:8 conservador
    vision_tower_mode: SparsityMode = SparsityMode.SEMI_24  # 2:4 seguro
    audio_tower_mode: SparsityMode = SparsityMode.SEMI_24
    skip_per_layer_gate: bool = True    # Sempre skip no bottleneck 256-dim
    skip_embeddings: bool = True
    skip_lm_head: bool = True
    skip_ple: bool = True               # Per-Layer Embeddings (huge, skip)
    use_gate_score_for_pruning: bool = True  # Usa SparsityPredictor como sinal


@dataclass
class QuantizationConfig:
    """Quantização para CPU inference."""
    enabled: bool = True
    weight_bits: int = 4                # INT4 via GGUF/AWQ
    activation_bits: int = 8            # INT8 dinâmico
    ple_bits: int = 4                   # PLE é 70% do modelo — INT4 crítico
    kv_cache_bits: int = 8


@dataclass
class Gemma4E4BConfig:
    """
    Configuração mestra para otimização do Gemma 4 E4B.
    Todas as dimensões correspondem à arquitetura real.
    """
    # Modelo base
    base_model_id: str = "google/gemma-4-e4b-it"

    # ── Dimensões da arquitetura E4B ──────────────────────────────────────
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 42
    vocab_size: int = 262144

    # Atenção local (layers 0-4, 6-10, ...)
    num_local_q_heads: int = 8          # q_proj: 2560→2048 = 8 heads × 256
    num_local_kv_heads: int = 2         # k/v_proj: 2560→512 = 2 heads × 256
    local_head_dim: int = 256
    local_q_dim: int = 2048             # 8 × 256
    local_kv_dim: int = 512             # 2 × 256

    # Atenção global (layers 5, 11, 17, 23, 29, 35, 41)
    num_global_q_heads: int = 16        # q_proj: 2560→4096 = 16 heads × 256
    num_global_kv_heads: int = 4        # k/v_proj: 2560→1024 = 4 heads × 256
    global_q_dim: int = 4096            # 16 × 256
    global_kv_dim: int = 1024           # 4 × 256

    # Per-Layer Embeddings (PLE) — exclusivo do E4B
    ple_dim: int = 10752                # embed_tokens_per_layer: 262144 × 10752
    gate_bottleneck_size: int = 256     # per_layer_input_gate: 2560 → 256

    # Layer indices
    global_layers: List[int] = field(
        default_factory=lambda: sorted(GLOBAL_LAYER_INDICES)
    )
    kv_shared_layers: List[int] = field(
        default_factory=lambda: sorted(KV_SHARED_LAYERS)
    )

    # ── Sub-configurações ─────────────────────────────────────────────────
    relu2: ReLU2Config = field(default_factory=ReLU2Config)
    gated_attention: GatedAttentionConfig = field(default_factory=GatedAttentionConfig)
    weight_sparsity: WeightSparsityConfig = field(default_factory=WeightSparsityConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Output
    output_dir: str = "./gemma4_e4b_optimized"
    torch_dtype: str = "bfloat16"

    # ── Helpers ───────────────────────────────────────────────────────────

    def is_global_layer(self, idx: int) -> bool:
        return idx in GLOBAL_LAYER_INDICES

    def has_own_kv(self, idx: int) -> bool:
        """Layer tem k_proj/v_proj próprio (não shared)?"""
        return idx not in KV_SHARED_LAYERS

    def is_local_layer(self, idx: int) -> bool:
        return not self.is_global_layer(idx)

    def get_q_dim(self, layer_idx: int) -> int:
        return self.global_q_dim if self.is_global_layer(layer_idx) else self.local_q_dim

    def get_kv_dim(self, layer_idx: int) -> int:
        return self.global_kv_dim if self.is_global_layer(layer_idx) else self.local_kv_dim

    def get_num_q_heads(self, layer_idx: int) -> int:
        return self.num_global_q_heads if self.is_global_layer(layer_idx) else self.num_local_q_heads

    def get_num_kv_heads(self, layer_idx: int) -> int:
        return self.num_global_kv_heads if self.is_global_layer(layer_idx) else self.num_local_kv_heads

    def active_modifications(self) -> List[str]:
        active = []
        if self.relu2.enabled:
            active.append(f"ReLU² MLP (target={self.relu2.sparsity_target:.0%})")
        if self.gated_attention.enabled:
            active.append("Gated Attention (global layers)")
        if self.weight_sparsity.enabled:
            active.append(f"Weight Sparsity ({self.weight_sparsity.text_decoder_mode.value} text)")
        if self.quantization.enabled:
            active.append(f"W{self.quantization.weight_bits}A{self.quantization.activation_bits}")
        return active
