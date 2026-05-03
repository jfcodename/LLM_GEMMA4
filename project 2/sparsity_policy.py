"""
sparse_gemma4/configs/sparsity_policy.py
=========================================
Política de esparsidade por módulo/layer baseada na análise arquitetural do Gemma 4.

Fundamentação científica:
- 2:4 sparsity: 2x speedup em Ampere+ mas pode colapsar reasoning (15.3% vs 54% em Qwen3)
- 6:8 sparsity via SlideSparse: preserva acurácia (51.6% vs 54%) com ~1.33x speedup
- Layers com dim < 512 não devem ser esparsos (bottleneck = risco de degradação)
- Embeddings e LM head: excluídos (lookup tables, sem ganho computacional)
- Convoluções (audio): excluídas (não mapeiam para Sparse Tensor Cores)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SparsityMode(Enum):
    """Modos de esparsidade disponíveis."""
    DENSE        = "dense"          # Sem esparsidade (baseline)
    SEMI_24      = "2:4"            # NVIDIA Sparse TC nativo — 50% zeros, 2x speedup
    SEMI_68      = "6:8"            # SlideSparse — 25% zeros, ~1.33x speedup, melhor acurácia
    UNSTRUCTURED = "unstructured"   # L1 regularization (Sakana AI) — max flexibilidade
    SKIP         = "skip"           # Excluído explicitamente


@dataclass
class LayerPolicy:
    mode: SparsityMode
    sparsity_ratio: float = 0.5          # Fraction de zeros alvo
    min_dim: int = 64                     # Dimensão mínima para aplicar esparsidade
    skip_patterns: list = field(default_factory=list)  # Padrões de nome a pular
    fine_tune_steps: int = 0             # Passos de fine-tune após pruning (0 = post-hoc)
    note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# POLÍTICA CONSERVADORA (recomendada para deployment imediato sem fine-tune)
# Usa 6:8 no text decoder (preserva reasoning) e 2:4 na vision tower (seguro)
# ─────────────────────────────────────────────────────────────────────────────
CONSERVATIVE_POLICY: dict[str, LayerPolicy] = {

    # ── LANGUAGE MODEL ────────────────────────────────────────────────────────
    # FFN principal: maior concentração de FLOPs — alvo primário
    "language_model.layers.*.mlp.gate_proj": LayerPolicy(
        mode=SparsityMode.SEMI_68,
        sparsity_ratio=0.25,
        note="FFN gate — 6:8 preserva MoE-like gating"
    ),
    "language_model.layers.*.mlp.up_proj": LayerPolicy(
        mode=SparsityMode.SEMI_68,
        sparsity_ratio=0.25,
        note="FFN up — 10240-dim, ganho máximo"
    ),
    "language_model.layers.*.mlp.down_proj": LayerPolicy(
        mode=SparsityMode.SEMI_68,
        sparsity_ratio=0.25,
        note="FFN down — projeção de volta para hidden"
    ),

    # Atenção: local layers (dim 2048/512) — seguro
    "language_model.layers.*.self_attn.q_proj": LayerPolicy(
        mode=SparsityMode.SEMI_68,
        sparsity_ratio=0.25,
        min_dim=512,
        note="Q proj — GQA local layers"
    ),
    "language_model.layers.*.self_attn.k_proj": LayerPolicy(
        mode=SparsityMode.SEMI_68,
        sparsity_ratio=0.25,
        min_dim=512,
        note="K proj — GQA reduzido mas múltiplo de 16"
    ),
    "language_model.layers.*.self_attn.v_proj": LayerPolicy(
        mode=SparsityMode.SEMI_68,
        sparsity_ratio=0.25,
        min_dim=512,
        note="V proj"
    ),
    "language_model.layers.*.self_attn.o_proj": LayerPolicy(
        mode=SparsityMode.SEMI_68,
        sparsity_ratio=0.25,
        min_dim=512,
        note="Output proj"
    ),

    # per_layer gates: SKIP — bottleneck de 256-dim, crítico para AltUp/LAuReL
    "language_model.layers.*.per_layer_input_gate": LayerPolicy(
        mode=SparsityMode.SKIP,
        note="Bottleneck 2560→256 — AltUp gating, excluído"
    ),
    "language_model.layers.*.per_layer_projection": LayerPolicy(
        mode=SparsityMode.SKIP,
        note="Bottleneck 256→2560 — AltUp projection, excluído"
    ),

    # ── VISION ENCODER ────────────────────────────────────────────────────────
    # Gemma4ClippableLinear já tem wrapper de mascaramento — 2:4 é seguro aqui
    "vision_tower.encoder.layers.*.self_attn.q_proj": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Vision attn — 768-dim, ClippableLinear nativo"
    ),
    "vision_tower.encoder.layers.*.self_attn.k_proj": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Vision K proj"
    ),
    "vision_tower.encoder.layers.*.self_attn.v_proj": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Vision V proj"
    ),
    "vision_tower.encoder.layers.*.self_attn.o_proj": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Vision O proj"
    ),
    "vision_tower.encoder.layers.*.mlp.gate_proj": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Vision FFN gate — 768→3072"
    ),
    "vision_tower.encoder.layers.*.mlp.up_proj": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Vision FFN up"
    ),
    "vision_tower.encoder.layers.*.mlp.down_proj": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Vision FFN down"
    ),

    # ── AUDIO ENCODER ─────────────────────────────────────────────────────────
    # FFN linear layers do conformer — 1024↔4096, dims ok
    "audio_tower.layers.*.feed_forward1.ffw_layer_1": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Audio FFN1 up — 1024→4096"
    ),
    "audio_tower.layers.*.feed_forward1.ffw_layer_2": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Audio FFN1 down — 4096→1024"
    ),
    "audio_tower.layers.*.feed_forward2.ffw_layer_1": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Audio FFN2 up"
    ),
    "audio_tower.layers.*.feed_forward2.ffw_layer_2": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Audio FFN2 down"
    ),
    "audio_tower.layers.*.self_attn.q_proj": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Audio Q proj — 1024→1024"
    ),
    "audio_tower.layers.*.self_attn.k_proj": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Audio K proj"
    ),
    "audio_tower.layers.*.self_attn.v_proj": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Audio V proj"
    ),
    "audio_tower.layers.*.self_attn.post": LayerPolicy(
        mode=SparsityMode.SEMI_24,
        sparsity_ratio=0.5,
        note="Audio attn output"
    ),

    # EXCLUÍDOS — não aplicar jamais
    "audio_tower.layers.*.lconv1d.*": LayerPolicy(
        mode=SparsityMode.SKIP,
        note="Depthwise Conv1d — não mapeia para Sparse TC"
    ),
    "audio_tower.subsample_conv_projection.*": LayerPolicy(
        mode=SparsityMode.SKIP,
        note="Conv2d subsampling — excluído"
    ),
    "embed_tokens": LayerPolicy(
        mode=SparsityMode.SKIP,
        note="Embedding lookup — sem ganho computacional"
    ),
    "lm_head": LayerPolicy(
        mode=SparsityMode.SKIP,
        note="LM head — lookup inverso, excluído"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# POLÍTICA AGRESSIVA (com fine-tuning posterior — máximo speedup)
# ─────────────────────────────────────────────────────────────────────────────
AGGRESSIVE_POLICY: dict[str, LayerPolicy] = {
    k: LayerPolicy(
        mode=SparsityMode.SEMI_24 if v.mode != SparsityMode.SKIP else SparsityMode.SKIP,
        sparsity_ratio=0.5,
        fine_tune_steps=1000,
        min_dim=v.min_dim,
        note=v.note + " [AGGRESSIVE]"
    )
    for k, v in CONSERVATIVE_POLICY.items()
}
