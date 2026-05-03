"""
Test: Mock Gemma 4 E4B — Fidelidade Arquitetural

Valida que o mock reproduz fielmente o layout real do E4B:
- 42 text layers com padrão 5:1 (local:global)
- KV sharing nas layers 24-41
- Per-Layer Embeddings (PLE)
- Dimensões corretas de q/k/v por tipo de layer (lite mode)
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified.config import (
    Gemma4E4BConfig,
    GLOBAL_LAYER_INDICES,
    KV_SHARED_LAYERS,
    KV_OWN_LAYERS,
)
from unified.mock_gemma4_e4b import MockGemma4E4B, MockTextDecoderLayer


@pytest.fixture
def config():
    return Gemma4E4BConfig()


@pytest.fixture
def model():
    """Lite model for CPU testing."""
    return MockGemma4E4B(lite=True)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerLayout:
    """Valida o padrão 5 local : 1 global."""

    def test_total_layers(self, model):
        assert len(model.text_layers) == 42

    def test_global_layers_at_correct_indices(self, model):
        """Global layers em 5, 11, 17, 23, 29, 35, 41."""
        for i in range(42):
            layer = model.get_layer(i)
            expected_global = i in GLOBAL_LAYER_INDICES
            assert layer.is_global == expected_global, (
                f"Layer {i}: is_global={layer.is_global}, expected={expected_global}"
            )

    def test_seven_global_layers(self, model):
        global_count = sum(1 for i in range(42) if model.get_layer(i).is_global)
        assert global_count == 7

    def test_thirty_five_local_layers(self, model):
        local_count = sum(1 for i in range(42) if not model.get_layer(i).is_global)
        assert local_count == 35


# ─────────────────────────────────────────────────────────────────────────────
# KV SHARING
# ─────────────────────────────────────────────────────────────────────────────

class TestKVSharing:
    """Valida KV sharing nas layers 24-41."""

    def test_layers_0_23_have_own_kv(self, model):
        """Layers 0-23 devem ter k_proj e v_proj próprios."""
        for i in range(24):
            layer = model.get_layer(i)
            assert layer.has_own_kv, f"Layer {i} deveria ter KV próprio"
            assert "k_proj" in layer.self_attn, f"Layer {i} sem k_proj"
            assert "v_proj" in layer.self_attn, f"Layer {i} sem v_proj"
            assert "k_norm" in layer.self_attn, f"Layer {i} sem k_norm"
            assert "v_norm" in layer.self_attn, f"Layer {i} sem v_norm"

    def test_layers_24_41_share_kv(self, model):
        """Layers 24-41 NÃO devem ter k_proj/v_proj."""
        for i in range(24, 42):
            layer = model.get_layer(i)
            assert not layer.has_own_kv, f"Layer {i} não deveria ter KV próprio"
            assert "k_proj" not in layer.self_attn, f"Layer {i} tem k_proj indevidamente"
            assert "v_proj" not in layer.self_attn, f"Layer {i} tem v_proj indevidamente"

    def test_kv_sharing_boundary(self, model):
        """Layer 23 tem KV, layer 24 não."""
        assert model.get_layer(23).has_own_kv
        assert not model.get_layer(24).has_own_kv

    def test_kv_shared_constants(self):
        """Constantes KV_SHARED_LAYERS corretas."""
        assert len(KV_SHARED_LAYERS) == 18  # layers 24-41
        assert min(KV_SHARED_LAYERS) == 24
        assert max(KV_SHARED_LAYERS) == 41
        assert len(KV_OWN_LAYERS) == 24     # layers 0-23


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSÕES (LITE MODE — layout preservado, dims escaladas)
# ─────────────────────────────────────────────────────────────────────────────

class TestDimensions:
    """Valida dimensões corretas no lite mode."""

    def test_local_vs_global_q_dim_differ(self, model):
        """Local q_proj e global q_proj têm dimensões diferentes."""
        local_q = model.get_layer(0).self_attn["q_proj"].out_features
        global_q = model.get_layer(5).self_attn["q_proj"].out_features
        assert global_q > local_q, (
            f"Global Q ({global_q}) deve ser > Local Q ({local_q})"
        )

    def test_local_kv_smaller_than_q(self, model):
        """Local layers: kv_dim < q_dim (GQA)."""
        layer = model.get_layer(0)
        q_dim = layer.self_attn["q_proj"].out_features
        kv_dim = layer.self_attn["k_proj"].out_features
        assert kv_dim < q_dim, f"kv={kv_dim} should be < q={q_dim}"

    def test_mlp_dimensions_consistent(self, model):
        """MLP: hidden → intermediate → hidden em todas as layers."""
        hidden = model._dims["hidden"]
        inter = model._dims["intermediate"]
        for i in [0, 5, 23, 24, 41]:
            layer = model.get_layer(i)
            assert layer.mlp["gate_proj"].in_features == hidden
            assert layer.mlp["gate_proj"].out_features == inter
            assert layer.mlp["down_proj"].in_features == inter
            assert layer.mlp["down_proj"].out_features == hidden

    def test_per_layer_gate_in_all_layers(self, model):
        """Per-layer gate presente em todas as 42 layers."""
        hidden = model._dims["hidden"]
        gate_bn = model._dims["gate_bottleneck"]
        for i in range(42):
            layer = model.get_layer(i)
            assert layer.per_layer_input_gate.in_features == hidden
            assert layer.per_layer_input_gate.out_features == gate_bn
            assert layer.per_layer_projection.in_features == gate_bn
            assert layer.per_layer_projection.out_features == hidden

    def test_o_proj_matches_q_dim(self, model):
        """o_proj: q_dim → hidden (inverte q_proj)."""
        hidden = model._dims["hidden"]
        for i in [0, 5, 24, 41]:
            layer = model.get_layer(i)
            q_dim = layer.self_attn["q_proj"].out_features
            assert layer.self_attn["o_proj"].in_features == q_dim
            assert layer.self_attn["o_proj"].out_features == hidden


# ─────────────────────────────────────────────────────────────────────────────
# PER-LAYER EMBEDDINGS (PLE)
# ─────────────────────────────────────────────────────────────────────────────

class TestPLE:
    """Valida Per-Layer Embeddings (exclusivo do E4B)."""

    def test_ple_exists(self, model):
        lm = model.language_model
        assert "embed_tokens_per_layer" in lm
        assert "per_layer_model_projection" in lm
        assert "per_layer_projection_norm" in lm

    def test_ple_projection_shape(self, model):
        """per_layer_model_projection: hidden → ple_dim."""
        proj = model.language_model["per_layer_model_projection"]
        assert proj.in_features == model._dims["hidden"]
        assert proj.out_features == model._dims["ple_dim"]


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

class TestForward:
    """Valida que o forward pass funciona e produz shapes corretas."""

    def test_forward_basic(self, model):
        """Forward pass não crasheia."""
        x = torch.randint(0, 1024, (1, 16))
        out = model(x)
        assert out.shape == (1, 16, 1024)  # (batch, seq, vocab_mock)

    def test_single_layer_forward(self, model):
        """Uma layer individual preserva shape."""
        hidden = model._dims["hidden"]
        x = torch.randn(1, 8, hidden)
        for i in [0, 5, 23, 24, 41]:
            out = model.get_layer(i)(x)
            assert out.shape == x.shape, f"Layer {i}: shape mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# VISION / AUDIO TOWERS
# ─────────────────────────────────────────────────────────────────────────────

class TestMultimodal:
    """Valida vision e audio towers."""

    def test_vision_tower_16_layers(self, model):
        layers = model.model["vision_tower"]["encoder"]["layers"]
        assert len(layers) == 16

    def test_audio_tower_12_layers(self, model):
        layers = model.model["audio_tower"]["layers"]
        assert len(layers) == 12

    def test_vision_clippable_linear(self, model):
        """Vision usa ClippableLinear com .linear interno."""
        layer = model.model["vision_tower"]["encoder"]["layers"][0]
        q = layer.self_attn["q_proj"]
        assert hasattr(q, "linear"), "Vision Q deve ter .linear (ClippableLinear)"

    def test_audio_has_conv(self, model):
        """Audio tower tem Conv1d no lconv1d."""
        layer = model.model["audio_tower"]["layers"][0]
        assert "depthwise_conv1d" in layer.lconv1d


# ─────────────────────────────────────────────────────────────────────────────
# PARÂMETROS
# ─────────────────────────────────────────────────────────────────────────────

class TestParameters:
    """Valida contagem e distribuição de parâmetros."""

    def test_total_params_reasonable(self, model):
        """Params totais em lite mode devem ser ~10-100M."""
        params = model.count_params_by_module()
        total = params["total"]
        assert total > 1_000_000, f"Muito poucos params: {total}"
        assert total < 500_000_000, f"Muitos params para lite: {total}"

    def test_text_decoder_dominates(self, model):
        """Text decoder deve ter maioria dos params."""
        params = model.count_params_by_module()
        text_ratio = params["text_decoder"] / params["total"]
        assert text_ratio > 0.5, f"Text decoder é só {text_ratio:.1%} dos params"
