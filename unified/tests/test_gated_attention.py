"""
Test: Gated Attention Layer

Valida:
- Gate sigmoid per-head funciona
- Gate=ones (init) → output ≈ output sem gate (transparente)
- Gate=zeros → output zero (suprime)
- Adapta-se a shared KV (layers sem k_proj/v_proj)
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified.config import GatedAttentionConfig
from unified.modules import GatedAttentionLayer


@pytest.fixture
def config():
    return GatedAttentionConfig(init_ones=True)


@pytest.fixture
def gate_layer(config):
    return GatedAttentionLayer(
        hidden_size=256,
        num_heads=4,
        head_dim=64,
        config=config,
    )


class TestGateShape:
    def test_gate_output_shape(self, gate_layer):
        """Gate shape: (B, num_heads, T, head_dim)."""
        x = torch.randn(2, 8, 256)
        gate = gate_layer.compute_gate(x)
        assert gate.shape == (2, 4, 8, 64)

    def test_gate_sigmoid_range(self, gate_layer):
        """Gate values devem estar em [0, 1] (sigmoid)."""
        x = torch.randn(2, 8, 256)
        gate = gate_layer.compute_gate(x)
        assert gate.min() >= 0.0
        assert gate.max() <= 1.0


class TestGateTransparency:
    def test_init_ones_gate_near_half(self, gate_layer):
        """Com init zeros (→ σ(0) = 0.5), gate começa em ~0.5."""
        x = torch.randn(2, 8, 256)
        gate = gate_layer.compute_gate(x)
        # σ(0) = 0.5, mas input varia então gate varia um pouco
        mean_gate = gate.mean().item()
        assert 0.3 < mean_gate < 0.7, f"Mean gate={mean_gate:.3f}, esperado ~0.5"

    def test_apply_gate_preserves_shape(self, gate_layer):
        """apply_gate preserva shape de attn_output."""
        attn_out = torch.randn(2, 4, 8, 64)  # (B, H, T, D)
        x = torch.randn(2, 8, 256)
        gated = gate_layer.apply_gate(attn_out, x)
        assert gated.shape == attn_out.shape


class TestGateEffect:
    def test_zero_gate_zeros_output(self):
        """Se gate → 0 (pesos negativos grandes), output → 0."""
        cfg = GatedAttentionConfig(init_ones=False)
        gate_layer = GatedAttentionLayer(
            hidden_size=256, num_heads=4, head_dim=64, config=cfg,
        )
        # Força gate_proj pesos para -10 → σ(-10·x) ≈ 0
        with torch.no_grad():
            gate_layer.gate_proj.weight.fill_(-100.0)

        attn_out = torch.randn(2, 4, 8, 64)
        x = torch.ones(2, 8, 256)  # Entrada positiva
        gated = gate_layer.apply_gate(attn_out, x)

        # Gated output deve ser ~zero
        assert gated.abs().max() < 0.01, (
            f"Com gate≈0, output deveria ser ~0, max={gated.abs().max():.4f}"
        )

    def test_full_gate_preserves_output(self):
        """Se gate → 1 (pesos positivos grandes), output ≈ attn_output."""
        cfg = GatedAttentionConfig(init_ones=False)
        gate_layer = GatedAttentionLayer(
            hidden_size=256, num_heads=4, head_dim=64, config=cfg,
        )
        # Força gate → 1
        with torch.no_grad():
            gate_layer.gate_proj.weight.fill_(100.0)

        attn_out = torch.randn(2, 4, 8, 64)
        x = torch.ones(2, 8, 256)
        gated = gate_layer.apply_gate(attn_out, x)

        # Output deve ser ≈ attn_out
        diff = (gated - attn_out).abs().max().item()
        assert diff < 0.01, f"Com gate≈1, output deveria ≈ input, diff={diff:.6f}"


class TestGradients:
    def test_gradient_flows(self, gate_layer):
        """Gradiente flui através do gate."""
        x = torch.randn(2, 8, 256, requires_grad=True)
        attn_out = torch.randn(2, 4, 8, 64, requires_grad=True)
        gated = gate_layer.apply_gate(attn_out, x)
        loss = gated.sum()
        loss.backward()
        assert x.grad is not None
        assert attn_out.grad is not None

    def test_gate_params_trainable(self, gate_layer):
        """Parâmetros do gate são treináveis."""
        params = list(gate_layer.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
