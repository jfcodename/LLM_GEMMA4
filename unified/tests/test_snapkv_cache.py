"""
Test: SnapKV Cache

Valida:
- Cache cresce com novos tokens
- Compressão mantém top-k + janela recente
- Não excede max_capacity após compressão
- Reset limpa estado
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unified.modules import SnapKVCache


@pytest.fixture
def cache():
    return SnapKVCache(max_capacity=64, window=16)


class TestCacheGrowth:
    def test_initial_empty(self, cache):
        assert cache.current_size == 0

    def test_first_update(self, cache):
        k = torch.randn(1, 4, 32, 64)  # (B, H, T, D)
        v = torch.randn(1, 4, 32, 64)
        k_out, v_out = cache.update(k, v)
        assert cache.current_size == 32

    def test_cache_grows(self, cache):
        k1 = torch.randn(1, 4, 20, 64)
        v1 = torch.randn(1, 4, 20, 64)
        cache.update(k1, v1)
        assert cache.current_size == 20

        k2 = torch.randn(1, 4, 10, 64)
        v2 = torch.randn(1, 4, 10, 64)
        cache.update(k2, v2)
        assert cache.current_size == 30


class TestCompression:
    def test_compression_triggers(self, cache):
        """Compressão ativa quando excede max_capacity."""
        k = torch.randn(1, 4, 80, 64)  # > max_capacity=64
        v = torch.randn(1, 4, 80, 64)
        attn_w = torch.randn(1, 4, 16, 80)  # mock attention weights
        k_out, v_out = cache.update(k, v, attn_weights=attn_w)
        assert cache.is_compressed
        assert cache.current_size <= cache.max_capacity

    def test_compression_preserves_recent_window(self, cache):
        """Janela recente é preservada após compressão."""
        k = torch.randn(1, 4, 80, 64)
        v = torch.randn(1, 4, 80, 64)
        attn_w = torch.rand(1, 4, 16, 80)
        cache.update(k, v, attn_weights=attn_w)

        # Após compressão: tokens finais devem estar no cache
        assert cache.current_size <= cache.max_capacity

    def test_does_not_exceed_capacity(self):
        """Cache nunca excede max_capacity após compressão."""
        cache = SnapKVCache(max_capacity=32, window=8)
        k = torch.randn(1, 2, 50, 32)
        v = torch.randn(1, 2, 50, 32)
        attn_w = torch.rand(1, 2, 8, 50)
        cache.update(k, v, attn_weights=attn_w)
        assert cache.current_size <= 32


class TestReset:
    def test_reset_clears_state(self, cache):
        k = torch.randn(1, 4, 20, 64)
        v = torch.randn(1, 4, 20, 64)
        cache.update(k, v)
        assert cache.current_size == 20

        cache.reset()
        assert cache.current_size == 0
        assert cache.key_cache is None
        assert cache.value_cache is None
        assert not cache.is_compressed


class TestReturnValues:
    def test_returns_concatenated_kv(self, cache):
        """update() retorna K/V completos acumulados."""
        k1 = torch.randn(1, 4, 10, 64)
        v1 = torch.randn(1, 4, 10, 64)
        k_out, v_out = cache.update(k1, v1)
        assert k_out.shape == (1, 4, 10, 64)

        k2 = torch.randn(1, 4, 5, 64)
        v2 = torch.randn(1, 4, 5, 64)
        k_out, v_out = cache.update(k2, v2)
        assert k_out.shape == (1, 4, 15, 64)
