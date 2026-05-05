"""
Mock Gemma 4 E4B — Modelo leve para testes sem GPU.

Reproduz fielmente o layout da arquitetura E4B incluindo:
- 42 text layers com alternância local/global (5:1)
- KV sharing nas layers 24-41 (sem k_proj/v_proj)
- Per-Layer Embeddings (PLE) com vocab reduzido para CPU
- Vision tower (16 layers, dim=768)
- Audio tower (12 layers, dim=1024)

Vocab reduzido para 1024 (vs 262144) para evitar OOM em CPU.
"""

import torch
import torch.nn as nn

from .config import (
    Gemma4E4BConfig,
    GLOBAL_LAYER_INDICES,
    KV_SHARED_LAYERS,
)


# Vocab reduzido para testes em CPU (original = 262144)
MOCK_VOCAB_SIZE = 1024


class MockClippableLinear(nn.Module):
    """Simula Gemma4ClippableLinear (wrapper com .linear interno)."""
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockRMSNorm(nn.Module):
    """Placeholder RMSNorm leve."""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-6)
        return x * self.weight


class MockTextMLP(nn.Module):
    """Simula Gemma4TextMLP para compatibilidade com scripts de pruning."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate='tanh')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = self.act_fn(self.gate_proj(x))
        up_out = self.up_proj(x)
        return self.down_proj(gate_out * up_out)


class MockTextDecoderLayer(nn.Module):
    """
    Simula Gemma4TextDecoderLayer do E4B.

    Args:
        layer_idx: Índice da layer (0-41)
        config: Configuração E4B

    Layers 0-23: Têm k_proj, v_proj, k_norm, v_norm próprios
    Layers 24-41: Sem k_proj/v_proj (KV cache compartilhado)
    """
    def __init__(self, layer_idx: int, config: Gemma4E4BConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = config.is_global_layer(layer_idx)
        self.has_own_kv = config.has_own_kv(layer_idx)
        hidden = config.hidden_size

        q_dim = config.get_q_dim(layer_idx)
        kv_dim = config.get_kv_dim(layer_idx)

        # ── Self-attention ────────────────────────────────────────────────
        self.self_attn = nn.ModuleDict()
        self.self_attn["q_proj"] = nn.Linear(hidden, q_dim, bias=False)
        self.self_attn["q_norm"] = MockRMSNorm(q_dim // config.get_num_q_heads(layer_idx))

        if self.has_own_kv:
            self.self_attn["k_proj"] = nn.Linear(hidden, kv_dim, bias=False)
            self.self_attn["v_proj"] = nn.Linear(hidden, kv_dim, bias=False)
            self.self_attn["k_norm"] = MockRMSNorm(kv_dim // config.get_num_kv_heads(layer_idx))
            self.self_attn["v_norm"] = MockRMSNorm(kv_dim // config.get_num_kv_heads(layer_idx))

        self.self_attn["o_proj"] = nn.Linear(q_dim, hidden, bias=False)

        # ── MLP (FFN) ─────────────────────────────────────────────────────
        intermediate = config.intermediate_size
        self.mlp = MockTextMLP(hidden, intermediate)

        # ── Norms ─────────────────────────────────────────────────────────
        self.input_layernorm = MockRMSNorm(hidden)
        self.post_attention_layernorm = MockRMSNorm(hidden)
        self.pre_feedforward_layernorm = MockRMSNorm(hidden)
        self.post_feedforward_layernorm = MockRMSNorm(hidden)

        # ── Per-layer gate (AltUp/LAuReL) — presente em TODAS as layers ──
        self.per_layer_input_gate = nn.Linear(hidden, config.gate_bottleneck_size, bias=False)
        self.per_layer_projection = nn.Linear(config.gate_bottleneck_size, hidden, bias=False)
        self.post_per_layer_input_norm = MockRMSNorm(hidden)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward simplificado (sem posição/máscara — apenas para teste de shapes)."""
        residual = x
        x = self.input_layernorm(x)

        # Atenção simulada (sem SDPA real — apenas projections)
        q = self.self_attn["q_proj"](x)
        if self.has_own_kv:
            k = self.self_attn["k_proj"](x)
            v = self.self_attn["v_proj"](x)
        # O_proj
        attn_out = self.self_attn["o_proj"](q)  # shape preservado
        x = residual + attn_out
        x = self.post_attention_layernorm(x)

        # Per-layer gate
        gate = self.per_layer_input_gate(x)
        gate = self.post_per_layer_input_norm(
            self.per_layer_projection(gate)
        )

        # MLP
        residual = x
        x = self.pre_feedforward_layernorm(x)
        mlp_out = self.mlp(x)
        x = residual + mlp_out
        x = self.post_feedforward_layernorm(x)

        return x


class MockVisionEncoderLayer(nn.Module):
    """Simula Gemma4VisionEncoderLayer (16 layers, dim=768)."""
    def __init__(self, dim: int = 768, intermediate: int = 3072):
        super().__init__()
        self.self_attn = nn.ModuleDict({
            "q_proj": MockClippableLinear(dim, dim),
            "k_proj": MockClippableLinear(dim, dim),
            "v_proj": MockClippableLinear(dim, dim),
            "o_proj": MockClippableLinear(dim, dim),
        })
        self.mlp = nn.ModuleDict({
            "gate_proj": MockClippableLinear(dim, intermediate),
            "up_proj":   MockClippableLinear(dim, intermediate),
            "down_proj": MockClippableLinear(intermediate, dim),
        })
        self.input_layernorm = MockRMSNorm(dim)
        self.post_attention_layernorm = MockRMSNorm(dim)

    def forward(self, x):
        return x


class MockAudioLayer(nn.Module):
    """Simula Gemma4AudioLayer (12 layers, Conformer, dim=1024)."""
    def __init__(self, dim: int = 1024, intermediate: int = 4096):
        super().__init__()
        self.feed_forward1 = nn.ModuleDict({
            "ffw_layer_1": MockClippableLinear(dim, intermediate),
            "ffw_layer_2": MockClippableLinear(intermediate, dim),
        })
        self.feed_forward2 = nn.ModuleDict({
            "ffw_layer_1": MockClippableLinear(dim, intermediate),
            "ffw_layer_2": MockClippableLinear(intermediate, dim),
        })
        self.self_attn = nn.ModuleDict({
            "q_proj": MockClippableLinear(dim, dim),
            "k_proj": MockClippableLinear(dim, dim),
            "v_proj": MockClippableLinear(dim, dim),
            "post":   MockClippableLinear(dim, dim),
        })
        # Conv1d (não esparsificável)
        self.lconv1d = nn.ModuleDict({
            "linear_start": MockClippableLinear(dim, dim * 2),
            "linear_end":   MockClippableLinear(dim, dim),
            "depthwise_conv1d": nn.Conv1d(dim, dim, 5, groups=dim, padding=2, bias=False),
        })

    def forward(self, x):
        return x


class MockGemma4E4B(nn.Module):
    """
    Modelo mock completo do Gemma 4 E4B para testes.

    Modes:
    - lite=True (default): Dimensões reduzidas (hidden=256, inter=1024) para
      rodar em CPU sem OOM. Layout arquitetural preservado fielmente.
    - lite=False: Dimensões reais do E4B (hidden=2560, inter=10240).
      Requer ~8GB+ RAM; use para testes de fidelidade dimensional.

    Layout sempre reproduz:
    - 42 text layers (35 local + 7 global)
    - KV sharing nas layers 24-41
    - PLE (per-layer embeddings)
    - 16 vision layers, 12 audio layers
    """

    def __init__(self, config: Gemma4E4BConfig = None, lite: bool = True):
        super().__init__()
        self.config = config or Gemma4E4BConfig()
        self.lite = lite
        c = self.config

        vocab = MOCK_VOCAB_SIZE  # Always reduced for mock

        # Lite mode: scale dims down by ~10x, keep divisibility
        if lite:
            self._scale = 0.1
            c_hidden = 256          # 2560 / 10
            c_inter = 1024          # 10240 / 10
            c_gate_bn = 32          # 256 / 8
            c_ple_dim = 128         # 10752 / ~84
            c_local_q = 256         # = hidden (simplificado)
            c_local_kv = 64         # 512 / 8
            c_global_q = 512        # 4096 / 8
            c_global_kv = 128       # 1024 / 8
            c_vision_dim = 128      # 768 / 6
            c_vision_inter = 512    # 3072 / 6
            c_audio_dim = 128       # 1024 / 8
            c_audio_inter = 512     # 4096 / 8
        else:
            self._scale = 1.0
            c_hidden = c.hidden_size
            c_inter = c.intermediate_size
            c_gate_bn = c.gate_bottleneck_size
            c_ple_dim = c.ple_dim
            c_local_q = c.local_q_dim
            c_local_kv = c.local_kv_dim
            c_global_q = c.global_q_dim
            c_global_kv = c.global_kv_dim
            c_vision_dim = 768
            c_vision_inter = 3072
            c_audio_dim = 1024
            c_audio_inter = 4096

        # Store effective dims for tests
        self._dims = {
            "hidden": c_hidden, "intermediate": c_inter,
            "gate_bottleneck": c_gate_bn, "ple_dim": c_ple_dim,
            "local_q": c_local_q, "local_kv": c_local_kv,
            "global_q": c_global_q, "global_kv": c_global_kv,
            "vision_dim": c_vision_dim, "audio_dim": c_audio_dim,
        }

        # ── Create a lite config clone for layer construction ─────────────
        from copy import copy
        lc = copy(c)
        lc.hidden_size = c_hidden
        lc.intermediate_size = c_inter
        lc.gate_bottleneck_size = c_gate_bn
        lc.ple_dim = c_ple_dim
        lc.local_q_dim = c_local_q
        lc.local_kv_dim = c_local_kv
        lc.global_q_dim = c_global_q
        lc.global_kv_dim = c_global_kv
        # Adjust head counts to match reduced dims
        lc.local_head_dim = max(32, c_local_q // max(1, c.num_local_q_heads))
        lc.num_local_q_heads = max(1, c_local_q // lc.local_head_dim)
        lc.num_local_kv_heads = max(1, c_local_kv // lc.local_head_dim)
        lc.num_global_q_heads = max(1, c_global_q // lc.local_head_dim)
        lc.num_global_kv_heads = max(1, c_global_kv // lc.local_head_dim)
        self._lite_config = lc

        # ── Embeddings ────────────────────────────────────────────────────
        self.model = nn.ModuleDict()

        language_model = nn.ModuleDict()
        language_model["embed_tokens"] = nn.Embedding(vocab, c_hidden, padding_idx=0)
        language_model["norm"] = MockRMSNorm(c_hidden)

        # Per-Layer Embeddings (PLE) — exclusivo do E4B
        language_model["embed_tokens_per_layer"] = nn.Embedding(
            vocab, c_ple_dim, padding_idx=0
        )
        language_model["per_layer_model_projection"] = nn.Linear(
            c_hidden, c_ple_dim, bias=False
        )
        language_model["per_layer_projection_norm"] = MockRMSNorm(c_ple_dim)

        # ── Text Decoder Layers ───────────────────────────────────────────
        layers = nn.ModuleList([
            MockTextDecoderLayer(i, lc) for i in range(c.num_hidden_layers)
        ])
        language_model["layers"] = layers

        self.model["language_model"] = language_model

        # ── Vision Tower ──────────────────────────────────────────────────
        vision_encoder = nn.ModuleDict({
            "layers": nn.ModuleList([
                MockVisionEncoderLayer(c_vision_dim, c_vision_inter) for _ in range(16)
            ])
        })
        vision_tower = nn.ModuleDict({
            "encoder": vision_encoder,
            "pooler": nn.Identity(),
        })
        self.model["vision_tower"] = vision_tower

        # ── Audio Tower ───────────────────────────────────────────────────
        audio_layers = nn.ModuleList([
            MockAudioLayer(c_audio_dim, c_audio_inter) for _ in range(12)
        ])
        audio_tower = nn.ModuleDict({
            "layers": audio_layers,
            "output_proj": nn.Linear(c_audio_dim, c_audio_dim + c_audio_dim // 2, bias=True),
        })
        self.model["audio_tower"] = audio_tower

        # ── Multimodal Embedders ──────────────────────────────────────────
        self.model["embed_vision"] = nn.ModuleDict({
            "embedding_projection": nn.Linear(c_vision_dim, c_hidden, bias=False),
        })
        self.model["embed_audio"] = nn.ModuleDict({
            "embedding_projection": nn.Linear(
                c_audio_dim + c_audio_dim // 2, c_hidden, bias=False
            ),
        })

        # ── LM Head ──────────────────────────────────────────────────────
        self.lm_head = nn.Linear(c_hidden, vocab, bias=False)

    @property
    def language_model(self):
        return self.model["language_model"]

    @property
    def text_layers(self):
        return self.language_model["layers"]

    def get_layer(self, idx: int) -> MockTextDecoderLayer:
        return self.text_layers[idx]

    def count_params_by_module(self) -> dict:
        """Conta parâmetros por módulo principal."""
        modules = {
            "text_decoder": self.language_model,
            "vision_tower": self.model["vision_tower"],
            "audio_tower": self.model["audio_tower"],
            "lm_head": self.lm_head,
        }
        result = {}
        for name, mod in modules.items():
            result[name] = sum(p.numel() for p in mod.parameters())
        result["total"] = sum(p.numel() for p in self.parameters())
        return result

    def forward(self, input_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Forward simplificado para testes de shape."""
        hidden = self._dims["hidden"]
        if input_ids is None:
            input_ids = torch.randint(0, MOCK_VOCAB_SIZE, (1, 16))

        x = self.language_model["embed_tokens"](input_ids)
        for layer in self.text_layers:
            x = layer(x)
        x = self.language_model["norm"](x)
        logits = self.lm_head(x)
        return logits
