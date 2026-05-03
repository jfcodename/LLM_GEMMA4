"""
Test: Sparsity Masks (2:4 e 6:8)

Valida:
- Máscara 2:4 tem exatamente 50% de zeros (2 de cada 4)
- Máscara 6:8 tem exatamente 25% de zeros (2 de cada 8)
- Elegibilidade dimensional correta
- Mantém pesos de maior magnitude
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified.modules import magnitude_mask_24, magnitude_mask_68, check_dim_eligibility


# ─────────────────────────────────────────────────────────────────────────────
# 2:4 MASKS
# ─────────────────────────────────────────────────────────────────────────────

class TestMask24:
    def test_exact_sparsity(self):
        """Máscara 2:4 deve ter exatamente 50% de zeros."""
        torch.manual_seed(42)
        w = torch.randn(512, 1024)
        mask = magnitude_mask_24(w)
        sparsity = 1.0 - mask.float().mean().item()
        assert abs(sparsity - 0.5) < 0.001, f"Sparsidade 2:4 = {sparsity:.4f}, esperado 0.5"

    def test_two_of_four_pattern(self):
        """Cada grupo de 4 tem exatamente 2 ones e 2 zeros."""
        torch.manual_seed(42)
        w = torch.randn(128, 256)
        mask = magnitude_mask_24(w)
        mask_reshaped = mask.view(-1, 4)
        ones_per_group = mask_reshaped.sum(dim=-1)
        assert (ones_per_group == 2).all(), "Nem todos os grupos têm 2:4"

    def test_keeps_largest_magnitudes(self):
        """Mantém os 2 pesos de maior magnitude em cada grupo de 4."""
        w = torch.tensor([[1.0, 5.0, 2.0, 8.0]])  # Grupo único de 4
        mask = magnitude_mask_24(w)
        # Deve manter 5.0 e 8.0 (posições 1 e 3)
        expected = torch.tensor([[False, True, False, True]])
        assert torch.equal(mask, expected)

    def test_various_dimensions(self):
        """Funciona com diversas dimensões múltiplas de 4."""
        for rows, cols in [(64, 64), (128, 512), (2560, 10240), (768, 3072)]:
            w = torch.randn(rows, cols)
            mask = magnitude_mask_24(w)
            sp = 1.0 - mask.float().mean().item()
            assert abs(sp - 0.5) < 0.001, f"({rows},{cols}): sp={sp:.4f}"

    def test_rejects_non_multiple_of_4(self):
        """Deve falhar se cols não é múltiplo de 4."""
        w = torch.randn(64, 65)
        with pytest.raises(AssertionError):
            magnitude_mask_24(w)

    def test_rejects_1d(self):
        """Deve falhar com tensor 1D."""
        w = torch.randn(64)
        with pytest.raises(AssertionError):
            magnitude_mask_24(w)


# ─────────────────────────────────────────────────────────────────────────────
# 6:8 MASKS
# ─────────────────────────────────────────────────────────────────────────────

class TestMask68:
    def test_exact_sparsity(self):
        """Máscara 6:8 deve ter exatamente 25% de zeros."""
        torch.manual_seed(42)
        w = torch.randn(512, 1024)
        mask = magnitude_mask_68(w)
        sparsity = 1.0 - mask.float().mean().item()
        assert abs(sparsity - 0.25) < 0.001, f"Sparsidade 6:8 = {sparsity:.4f}, esperado 0.25"

    def test_six_of_eight_pattern(self):
        """Cada grupo de 8 tem exatamente 6 ones e 2 zeros."""
        torch.manual_seed(42)
        w = torch.randn(128, 256)
        mask = magnitude_mask_68(w)
        mask_reshaped = mask.view(-1, 8)
        ones_per_group = mask_reshaped.sum(dim=-1)
        assert (ones_per_group == 6).all(), "Nem todos os grupos têm 6:8"

    def test_keeps_largest_magnitudes_68(self):
        """Mantém os 6 maiores de cada grupo de 8."""
        w = torch.tensor([[1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 0.5, 6.0]])
        mask = magnitude_mask_68(w)
        # Menores: 0.5 (pos 6) e 1.0 (pos 0) → zerados
        expected = torch.tensor([[False, True, True, True, True, True, False, True]])
        assert torch.equal(mask, expected)

    def test_gemma4_dimensions(self):
        """Funciona com dimensões do Gemma 4 E4B."""
        for rows, cols in [
            (2560, 10240),   # MLP gate/up
            (10240, 2560),   # MLP down
            (2560, 2048),    # local q_proj
            (2560, 512),     # local k/v_proj
            (2560, 4096),    # global q_proj
            (2560, 1024),    # global k/v_proj
            (768, 3072),     # vision MLP
            (1024, 4096),    # audio FFN
        ]:
            w = torch.randn(rows, cols)
            mask = magnitude_mask_68(w)
            sp = 1.0 - mask.float().mean().item()
            assert abs(sp - 0.25) < 0.01, f"({rows},{cols}): sp={sp:.4f}"

    def test_handles_non_multiple_of_8(self):
        """Lida com cols que não são múltiplo de 8 (com padding)."""
        w = torch.randn(64, 100)  # 100 não é múltiplo de 8
        mask = magnitude_mask_68(w)
        assert mask.shape == (64, 100)


# ─────────────────────────────────────────────────────────────────────────────
# APLICAÇÃO DE MÁSCARA
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskApplication:
    def test_masked_weights_preserve_shape(self):
        """Peso mascarado mantém shape original."""
        w = torch.randn(512, 1024)
        mask = magnitude_mask_24(w)
        pruned = w * mask
        assert pruned.shape == w.shape

    def test_masked_weights_have_zeros(self):
        """Peso mascarado tem zeros nas posições corretas."""
        w = torch.randn(512, 1024) + 0.1  # Evita zeros naturais
        mask = magnitude_mask_24(w)
        pruned = w * mask
        zeros = (pruned == 0).float().mean().item()
        assert abs(zeros - 0.5) < 0.01

    def test_masked_weights_preserve_nonzeros(self):
        """Valores não-zerados são preservados exatamente."""
        w = torch.randn(64, 64)
        mask = magnitude_mask_24(w)
        pruned = w * mask
        # Onde mask=True, pruned deve ser igual a w
        assert torch.allclose(pruned[mask], w[mask])


# ─────────────────────────────────────────────────────────────────────────────
# ELEGIBILIDADE DIMENSIONAL
# ─────────────────────────────────────────────────────────────────────────────

class TestDimEligibility:
    def test_24_eligible(self):
        w = torch.randn(256, 1024)
        ok, _ = check_dim_eligibility(w, "2:4")
        assert ok

    def test_24_not_eligible_non_mult4(self):
        w = torch.randn(256, 1023)
        ok, reason = check_dim_eligibility(w, "2:4")
        assert not ok
        assert "múltiplo de 4" in reason

    def test_24_not_eligible_too_small(self):
        w = torch.randn(32, 32)
        ok, reason = check_dim_eligibility(w, "2:4")
        assert not ok
        assert "pequenas" in reason

    def test_68_eligible(self):
        w = torch.randn(256, 1024)
        ok, _ = check_dim_eligibility(w, "6:8")
        assert ok

    def test_68_not_eligible_non_mult8(self):
        w = torch.randn(256, 1023)
        ok, reason = check_dim_eligibility(w, "6:8")
        assert not ok

    def test_1d_not_eligible(self):
        w = torch.randn(256)
        ok, reason = check_dim_eligibility(w, "2:4")
        assert not ok
        assert "2D" in reason

    def test_bottleneck_256_eligible_68(self):
        """Bottleneck 2560→256: dimensionalmente elegível para 6:8."""
        w = torch.randn(256, 2560)
        ok, _ = check_dim_eligibility(w, "6:8")
        assert ok  # Mas a política deve SKIP por ser bottleneck crítico
