"""
Test: SparsityPredictor

Valida que o preditor de esparsidade:
- Produz máscara com sparsidade alvo (~65%)
- Top-k seleciona número correto de neurônios
- Reutiliza pesos do per_layer_input_gate
- Estatísticas de running_sparsity funcionam
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified.config import ReLU2Config
from unified.modules import SparsityPredictor


@pytest.fixture
def config():
    return ReLU2Config(sparsity_target=0.65)


@pytest.fixture
def predictor(config):
    return SparsityPredictor(
        hidden_size=256,
        gate_bottleneck_size=32,
        intermediate_size=1024,
        config=config,
    )


@pytest.fixture
def predictor_full():
    """Predictor com dims reais do E4B."""
    config = ReLU2Config(sparsity_target=0.65)
    return SparsityPredictor(
        hidden_size=2560,
        gate_bottleneck_size=256,
        intermediate_size=10240,
        config=config,
    )


class TestMaskShape:
    def test_output_shape(self, predictor):
        x = torch.randn(2, 8, 256)
        mask = predictor(x)
        assert mask.shape == (2, 8, 1024)

    def test_mask_is_binary(self, predictor):
        """Máscara deve conter apenas 0.0 e 1.0."""
        x = torch.randn(2, 8, 256)
        mask = predictor(x)
        unique = mask.unique()
        assert len(unique) == 2, f"Valores únicos: {unique.tolist()}"
        assert 0.0 in unique
        assert 1.0 in unique


class TestSparsityTarget:
    def test_sparsity_near_target(self, predictor, config):
        """Sparsidade real deve estar próxima do target."""
        x = torch.randn(4, 16, 256)
        mask = predictor(x)
        actual_sparsity = 1.0 - mask.float().mean().item()
        target = config.sparsity_target
        assert abs(actual_sparsity - target) < 0.05, (
            f"Sparsidade real={actual_sparsity:.3f}, target={target:.3f}"
        )

    def test_topk_count(self, predictor, config):
        """Número de neurônios ativos = (1-sparsity) * intermediate."""
        x = torch.randn(1, 4, 256)
        mask = predictor(x)
        # Cada posição (B,T) deve ter exatamente topk neurônios ativos
        expected_topk = int((1.0 - config.sparsity_target) * 1024)
        for b in range(mask.shape[0]):
            for t in range(mask.shape[1]):
                active = mask[b, t].sum().item()
                assert active == expected_topk, (
                    f"Pos ({b},{t}): {int(active)} ativos, esperado {expected_topk}"
                )

    def test_full_dims_sparsity(self, predictor_full):
        """Com dims reais do E4B (2560/256/10240)."""
        x = torch.randn(1, 4, 2560)
        mask = predictor_full(x)
        assert mask.shape == (1, 4, 10240)
        actual_sp = 1.0 - mask.float().mean().item()
        assert abs(actual_sp - 0.65) < 0.05

    def test_different_sparsity_targets(self):
        """Diferentes targets produzem sparsidades corretas."""
        for target in [0.3, 0.5, 0.8, 0.95]:
            cfg = ReLU2Config(sparsity_target=target)
            pred = SparsityPredictor(
                hidden_size=256,
                gate_bottleneck_size=32,
                intermediate_size=1024,
                config=cfg,
            )
            x = torch.randn(2, 8, 256)
            mask = pred(x)
            actual = 1.0 - mask.float().mean().item()
            assert abs(actual - target) < 0.05, (
                f"target={target}, actual={actual:.3f}"
            )


class TestWeightReuse:
    def test_gate_weight_loaded(self):
        """Pesos do per_layer_input_gate são reutilizados."""
        cfg = ReLU2Config()
        # Simula pesos do gate original
        fake_gate_w = torch.randn(32, 256) * 0.1
        fake_norm_w = torch.ones(32)

        pred = SparsityPredictor(
            hidden_size=256,
            gate_bottleneck_size=32,
            intermediate_size=1024,
            config=cfg,
            gate_weight_in=fake_gate_w,
            gate_norm_weight=fake_norm_w,
        )

        assert torch.allclose(pred.gate_in.weight.data, fake_gate_w)
        assert torch.allclose(pred.gate_norm.weight.data, fake_norm_w)


class TestRunningStats:
    def test_no_stats_before_training(self, predictor):
        assert predictor.actual_sparsity == 0.0

    def test_stats_update_in_training(self, predictor):
        """Running sparsity atualiza durante training mode."""
        predictor.train()
        x = torch.randn(2, 8, 256)
        _ = predictor(x)
        assert predictor._n_steps > 0
        assert predictor.actual_sparsity > 0.0

    def test_stats_not_update_in_eval(self, predictor):
        """Não atualiza em eval mode."""
        predictor.eval()
        x = torch.randn(2, 8, 256)
        _ = predictor(x)
        assert predictor._n_steps == 0
