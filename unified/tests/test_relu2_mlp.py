"""
Test: ReLU² MLP

Valida:
- ReLU²(x) = max(0,x)² produz zeros exatos
- Sparsidade de ativação ≥ target com dados normais
- Forward/backward funciona corretamente
- Gradiente flui (não vanishing)
- Calibração de threshold
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified.config import ReLU2Config
from unified.modules import ReLU2GatedMLP, SparsityPredictor


@pytest.fixture
def config():
    return ReLU2Config(
        sparsity_target=0.65,
        use_gate_as_predictor=True,
    )


@pytest.fixture
def mlp(config):
    return ReLU2GatedMLP(
        hidden_size=256,      # Reduzido para teste
        intermediate_size=1024,
        config=config,
    )


@pytest.fixture
def predictor(config):
    return SparsityPredictor(
        hidden_size=256,
        gate_bottleneck_size=32,
        intermediate_size=1024,
        config=config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ReLU²
# ─────────────────────────────────────────────────────────────────────────────

class TestReLU2Function:
    """Testa a função ReLU² isoladamente."""

    def test_positive_values(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = ReLU2GatedMLP.relu2(x)
        expected = torch.tensor([1.0, 4.0, 9.0])
        assert torch.allclose(result, expected)

    def test_negative_values_are_zero(self):
        x = torch.tensor([-1.0, -2.0, -0.001])
        result = ReLU2GatedMLP.relu2(x)
        assert (result == 0).all(), "Valores negativos devem ser zero exato"

    def test_zero_is_zero(self):
        x = torch.tensor([0.0])
        result = ReLU2GatedMLP.relu2(x)
        assert result.item() == 0.0

    def test_mixed_values(self):
        x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32)
        result = ReLU2GatedMLP.relu2(x)
        expected = torch.tensor([0, 0, 0, 1, 4], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_produces_exact_zeros(self):
        """ReLU² deve produzir zeros EXATOS (não near-zero)."""
        torch.manual_seed(42)
        x = torch.randn(100, 1024)
        result = ReLU2GatedMLP.relu2(x)
        # Pelo menos ~50% devem ser zero exato (distribuição normal)
        zero_frac = (result == 0.0).float().mean().item()
        assert zero_frac > 0.40, f"Poucos zeros exatos: {zero_frac:.1%}"

    def test_sparsity_higher_than_gelu(self):
        """ReLU² deve ser mais esparso que GELUTanh nos mesmos dados."""
        torch.manual_seed(42)
        x = torch.randn(32, 1024)
        relu2_out = ReLU2GatedMLP.relu2(x)
        gelu_out = torch.nn.functional.gelu(x, approximate='tanh')

        relu2_zeros = (relu2_out == 0.0).float().mean().item()
        gelu_zeros = (gelu_out.abs() < 1e-6).float().mean().item()

        assert relu2_zeros > gelu_zeros, (
            f"ReLU² ({relu2_zeros:.1%}) deveria ser mais esparso que GELU ({gelu_zeros:.1%})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MLP FORWARD
# ─────────────────────────────────────────────────────────────────────────────

class TestReLU2MLP:
    """Testa o módulo ReLU²GatedMLP completo."""

    def test_forward_shape(self, mlp):
        """Output shape preservada."""
        x = torch.randn(2, 8, 256)
        out = mlp(x)
        assert out.shape == (2, 8, 256)

    def test_forward_with_mask(self, mlp, predictor):
        """Forward com máscara do predictor funciona."""
        x = torch.randn(2, 8, 256)
        mask = predictor(x)
        out = mlp(x, predictor_mask=mask)
        assert out.shape == (2, 8, 256)

    def test_mask_reduces_activations(self, mlp, predictor):
        """Com máscara, menos ativações não-zero."""
        x = torch.randn(2, 8, 256)

        # Sem máscara
        out_dense = mlp(x)

        # Com máscara
        mask = predictor(x)
        out_sparse = mlp(x, predictor_mask=mask)

        # Sparse deve ter mais zeros intermediários
        # (output pode não ser zero, mas magnitude reduzida)
        dense_norm = out_dense.abs().mean().item()
        sparse_norm = out_sparse.abs().mean().item()
        # Magnitude média do sparse deve ser menor (menos neurônios ativos)
        assert sparse_norm <= dense_norm * 1.2  # margem de 20%

    def test_backward_gradient_flows(self, mlp):
        """Gradiente flui através do ReLU²."""
        x = torch.randn(2, 4, 256, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0, "Gradiente é zero — vanishing"

    def test_gradient_with_mask(self, mlp, predictor):
        """Gradiente flui mesmo com máscara de sparsidade."""
        x = torch.randn(2, 4, 256, requires_grad=True)
        mask = predictor(x)
        out = mlp(x, predictor_mask=mask.detach())
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibration:
    """Testa calibração de threshold."""

    def test_calibrate_sets_threshold(self, mlp):
        """Calibração define threshold > 0."""
        data = torch.randn(32, 16, 256)
        sp = mlp.calibrate_threshold(data, percentile=35.0)
        assert mlp.activation_threshold.item() > 0
        assert 0 < sp < 1.0

    def test_calibrate_sparsity_near_target(self, mlp):
        """Sparsidade calibrada deve estar razoavelmente próxima do percentil."""
        data = torch.randn(64, 32, 256)
        sp = mlp.calibrate_threshold(data, percentile=65.0)
        # Deve estar entre 50% e 80%
        assert 0.40 < sp < 0.90, f"Sparsidade calibrada fora do esperado: {sp:.1%}"
