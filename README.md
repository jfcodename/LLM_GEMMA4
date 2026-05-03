# Gemma 4 E4B — Optimization Framework

Unified framework for optimizing the **Google Gemma 4 E4B** architecture combining two complementary approaches:

| Approach | Technique | Speedup | Accuracy Impact |
|---|---|---|---|
| **Neo Hybrid** | ReLU² MLP, SparsityPredictor, Gated Attention, SnapKV | ~1.5-2× (activation skip) | Minimal with calibration |
| **Structured Sparsity** | 6:8 SlideSparse, 2:4 NVIDIA Sparse TC | 1.33×–2× | Low (6:8) to High (2:4) |

## Key Findings

- **KV Sharing**: Layers 24-41 share KV cache (no `k_proj`/`v_proj`) — unique to E4B
- **PLE**: Per-Layer Embeddings hold ~70% of model params in one embedding table
- **ReLU² + 6:8**: Combined activation + weight sparsity exceeds either alone

## Structure

```
unified/
├── config.py              # E4B architecture config (dims, KV sharing, PLE)
├── modules.py             # SparsityPredictor, ReLU²MLP, GatedAttention, SnapKV, Masks
├── mock_gemma4_e4b.py     # Lightweight mock (lite mode for CPU testing)
├── colab_runner.py        # Runner for Google Colab (T4 GPU)
└── tests/                 # 91 tests covering all components
    ├── test_mock_e4b.py
    ├── test_relu2_mlp.py
    ├── test_sparsity_predictor.py
    ├── test_gated_attention.py
    ├── test_snapkv_cache.py
    ├── test_sparsifier_masks.py
    └── test_integration.py
```

## Quick Start

```bash
# Run tests locally (CPU)
pip install torch pytest
cd LLM_GEMMA4
python -m pytest unified/tests/ -v

# Run on Colab (T4 GPU)
python unified/colab_runner.py --test        # Validate environment
python unified/colab_runner.py --mock        # Mock analysis
python unified/colab_runner.py --real-model  # With gemma-4-e4b-it
```

## Test Results

```
91 passed in 17s ✅
```

## References

- **Gemma 4** — Google DeepMind (2026)
- **SlideSparse (6:8)** — arxiv 2603.05232 (March 2026)
- **Gated Attention** — NeurIPS 2025
- **ReLU²** — Sparse activation for LLMs
