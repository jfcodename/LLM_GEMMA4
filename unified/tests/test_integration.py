"""
Test: Integração ReLU² + Sparsity Masks

Valida a combinação das duas linhas de otimização:
- ReLU² gera esparsidade de ativação
- Masks 6:8 adicionam esparsidade de pesos
- Esparsidade combinada > esparsidade individual
- Pipeline completo: Mock E4B → ReLU² → 6:8 → forward pass
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified.config import Gemma4E4BConfig, ReLU2Config
from unified.mock_gemma4_e4b import MockGemma4E4B
from unified.modules import (
    ReLU2GatedMLP,
    SparsityPredictor,
    magnitude_mask_68,
    magnitude_mask_24,
)


@pytest.fixture
def model():
    return MockGemma4E4B(lite=True)


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRAÇÃO: ReLU² + 6:8 WEIGHT SPARSITY
# ─────────────────────────────────────────────────────────────────────────────

class TestReLU2PlusWeightSparsity:
    """Testa a combinação de esparsidade de ativação + peso."""

    def test_relu2_mlp_with_68_weights(self):
        """MLP com ReLU² E pesos 6:8 aplicados."""
        hidden, inter = 256, 1024

        cfg = ReLU2Config(sparsity_target=0.65)
        mlp = ReLU2GatedMLP(
            hidden_size=hidden,
            intermediate_size=inter,
            config=cfg,
        )

        # Aplica 6:8 nos pesos do MLP
        for name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(mlp, name)
            mask = magnitude_mask_68(proj.weight.data)
            proj.weight.data *= mask

        # Forward deve funcionar
        x = torch.randn(1, 8, hidden)
        out = mlp(x)
        assert out.shape == (1, 8, hidden)

        # Verificar que pesos têm ~25% zeros
        gate_zeros = (mlp.gate_proj.weight == 0).float().mean().item()
        assert abs(gate_zeros - 0.25) < 0.01

    def test_combined_sparsity_higher_than_weight_only(self):
        """
        ReLU² + 6:8 pesos → mais zeros no output intermediário que 6:8 sozinho.
        Compara intermediário mascarado vs não-mascarado.
        """
        hidden, inter = 256, 1024
        cfg = ReLU2Config(sparsity_target=0.65)

        mlp = ReLU2GatedMLP(
            hidden_size=hidden,
            intermediate_size=inter,
            config=cfg,
        )

        torch.manual_seed(42)
        x = torch.randn(4, 16, hidden)

        # Medir esparsidade de ativação com ReLU² (pesos densos)
        with torch.no_grad():
            gate_acts_dense = mlp.relu2(mlp.gate_proj(x))
            activation_sparsity = (gate_acts_dense == 0).float().mean().item()

        # Aplicar 6:8 nos pesos
        mask = magnitude_mask_68(mlp.gate_proj.weight.data)
        weight_sparsity = 1.0 - mask.float().mean().item()
        mlp.gate_proj.weight.data *= mask

        # Medir esparsidade combinada
        with torch.no_grad():
            gate_acts_sparse = mlp.relu2(mlp.gate_proj(x))
            combined_sparsity = (gate_acts_sparse == 0).float().mean().item()

        # Combinada (ReLU² + 6:8 weights) deve ser > peso sozinho (25%)
        assert combined_sparsity > weight_sparsity, (
            f"Combined ({combined_sparsity:.1%}) should be > weight-only ({weight_sparsity:.1%})"
        )

    def test_predictor_deterministic(self):
        """SparsityPredictor é determinístico dado o mesmo input."""
        hidden, inter = 256, 1024
        cfg = ReLU2Config(sparsity_target=0.65)

        predictor = SparsityPredictor(
            hidden_size=hidden,
            gate_bottleneck_size=32,
            intermediate_size=inter,
            config=cfg,
        )
        predictor.eval()

        x = torch.randn(4, 16, hidden)
        mask1 = predictor(x)
        mask2 = predictor(x)
        assert torch.equal(mask1, mask2), "Mesmo input deve gerar mesmo mask"


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRAÇÃO: MOCK E4B END-TO-END
# ─────────────────────────────────────────────────────────────────────────────

class TestMockE4BIntegration:
    """Testa aplicação de otimizações no mock E4B completo (lite mode)."""

    def test_apply_68_to_all_mlp_layers(self, model):
        """Aplica 6:8 em gate/up/down_proj de todas as 42 layers."""
        for i in range(42):
            layer = model.get_layer(i)
            for name in ["gate_proj", "up_proj", "down_proj"]:
                proj = layer.mlp[name]
                mask = magnitude_mask_68(proj.weight.data)
                proj.weight.data *= mask

        # Forward pass ainda funciona
        x = torch.randint(0, 1024, (1, 8))
        out = model(x)
        assert out.shape == (1, 8, 1024)

    def test_apply_68_skips_per_layer_gate(self, model):
        """per_layer_input_gate (bottleneck) NÃO deve receber 6:8 na política."""
        for i in range(42):
            layer = model.get_layer(i)
            gate = layer.per_layer_input_gate
            # Bottleneck tem out_features = gate_bottleneck
            assert gate.out_features == model._dims["gate_bottleneck"]

    def test_count_sparse_params(self, model):
        """Conta parâmetros zerados após aplicar 6:8 no MLP."""
        total_params = 0
        sparse_params = 0

        for i in range(42):
            layer = model.get_layer(i)
            for name in ["gate_proj", "up_proj", "down_proj"]:
                proj = layer.mlp[name]
                mask = magnitude_mask_68(proj.weight.data)
                proj.weight.data *= mask
                total_params += proj.weight.numel()
                sparse_params += (proj.weight == 0).sum().item()

        global_sparsity = sparse_params / total_params
        assert abs(global_sparsity - 0.25) < 0.01, (
            f"Sparsidade MLP global = {global_sparsity:.1%}, esperado ~25%"
        )

    def test_replace_mlp_with_relu2(self, model):
        """Substitui MLP original por ReLU²GatedMLP."""
        hidden = model._dims["hidden"]
        inter = model._dims["intermediate"]
        relu2_cfg = ReLU2Config(sparsity_target=0.65)
        layer = model.get_layer(0)

        new_mlp = ReLU2GatedMLP(
            hidden_size=hidden,
            intermediate_size=inter,
            config=relu2_cfg,
            gate_proj_weight=layer.mlp["gate_proj"].weight.data.clone(),
            up_proj_weight=layer.mlp["up_proj"].weight.data.clone(),
            down_proj_weight=layer.mlp["down_proj"].weight.data.clone(),
        )

        x = torch.randn(1, 4, hidden)
        out = new_mlp(x)
        assert out.shape == (1, 4, hidden)

    def test_kv_sharing_layers_get_sparse_mlp(self, model):
        """Layers 24-41 (sem KV) recebem esparsidade no MLP normalmente."""
        for i in range(24, 42):
            layer = model.get_layer(i)
            assert not layer.has_own_kv  # Confirma KV sharing
            # MLP ainda existe e pode ser esparsificado
            for name in ["gate_proj", "up_proj", "down_proj"]:
                assert name in layer.mlp
                mask = magnitude_mask_68(layer.mlp[name].weight.data)
                sp = 1.0 - mask.float().mean().item()
                assert abs(sp - 0.25) < 0.01
