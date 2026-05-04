"""
Gemma 4 E4B — GGUF Quantized Benchmark
========================================
Usa llama-cpp-python para carregar GGUF pré-quantizado do Unsloth.
Compara throughput GGUF Q4/Q8 vs nosso bf16+sparsity.

Uso no Kaggle:
    !CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python -q --no-cache-dir
    !pip install huggingface_hub -q
    !python unified/phase4b_int4_sparse.py
"""

import logging, sys, time, os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Quant files to test (smallest first for T4)
GGUF_REPO = "unsloth/gemma-4-E4B-it-GGUF"
GGUF_VARIANTS = [
    ("Q4_K_M", "gemma-4-E4B-it-Q4_K_M.gguf", "~5 GB"),
    ("Q8_0",   "gemma-4-E4B-it-Q8_0.gguf",   "~8.5 GB"),
]

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
]

# Our previous bf16 results for comparison
BF16_RESULTS = {
    "bf16 denso":            {"tps": 8.5, "sparsity": "0%",    "quality": "perfect"},
    "bf16 + Top-K 50%":      {"tps": 7.1, "sparsity": "50%",   "quality": "perfect"},
    "bf16 + 6:8 + Top-K 50%":{"tps": 7.2, "sparsity": "62.5%", "quality": "perfect"},
}


def download_gguf(filename):
    """Download GGUF file from HuggingFace."""
    from huggingface_hub import hf_hub_download
    cache_dir = "/kaggle/working/gguf_cache"
    os.makedirs(cache_dir, exist_ok=True)

    logger.info(f"Baixando {filename}...")
    path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=filename,
        cache_dir=cache_dir,
        local_dir=cache_dir,
    )
    logger.info(f"Download completo: {path}")
    return path


def benchmark_llama_cpp(model_path, prompts, n_gpu_layers=-1, max_tokens=50):
    """Benchmark usando llama-cpp-python."""
    from llama_cpp import Llama

    logger.info(f"Carregando {Path(model_path).name}...")
    t_load = time.perf_counter()

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,  # -1 = all layers on GPU
        n_ctx=2048,
        verbose=False,
    )

    load_time = time.perf_counter() - t_load
    logger.info(f"Modelo carregado em {load_time:.1f}s")

    results = []
    for prompt in prompts:
        # Format as chat
        messages = [{"role": "user", "content": prompt}]

        t0 = time.perf_counter()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - t0

        text = response["choices"][0]["message"]["content"]
        n_tok = response["usage"]["completion_tokens"]
        tps = n_tok / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt[:50],
            "tok_per_s": tps,
            "n_tokens": n_tok,
            "elapsed": elapsed,
            "output": text[:120] if text else "(empty)",
        })

        print(f"      {tps:.1f} tok/s | {n_tok} tok | {elapsed:.2f}s")
        print(f"      → {(text or '(empty)')[:100]}\n")

    # Cleanup
    del llm

    return results


def main():
    print(f"\n{'═'*60}")
    print(f"  GGUF QUANTIZED BENCHMARK")
    print(f"  Repo: {GGUF_REPO}")
    print(f"{'═'*60}\n")

    # Check llama-cpp-python
    try:
        from llama_cpp import Llama
        logger.info("llama-cpp-python encontrado ✅")
    except ImportError:
        print("  ❌ llama-cpp-python não instalado!")
        print("  Instale com CUDA:")
        print("    CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --no-cache-dir")
        print("  Ou sem CUDA (CPU only):")
        print("    pip install llama-cpp-python")
        return 1

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  GPU: {gpu} ({vram:.1f} GB)")
            n_gpu = -1  # All layers on GPU
        else:
            print(f"  GPU: Não disponível (CPU mode)")
            n_gpu = 0
    except Exception:
        print(f"  GPU: Checking via llama.cpp...")
        n_gpu = -1  # Let llama.cpp decide

    all_results = {}

    for variant_name, filename, est_size in GGUF_VARIANTS:
        print(f"\n{'─'*60}")
        print(f"  {variant_name} ({est_size})")
        print(f"{'─'*60}")

        try:
            model_path = download_gguf(filename)
            results = benchmark_llama_cpp(model_path, PROMPTS, n_gpu_layers=n_gpu)
            avg_tps = sum(r["tok_per_s"] for r in results) / len(results)
            all_results[variant_name] = {"tps": avg_tps, "results": results}
        except Exception as e:
            logger.error(f"{variant_name} falhou: {e}")
            # Try next variant
            continue

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print(f"  COMPARATIVO: GGUF vs bf16+SPARSIDADE")
    print(f"{'═'*60}")
    print(f"  {'Config':<30} {'tok/s':>8} {'Notas':>20}")
    print(f"  {'─'*58}")

    # GGUF results
    for name, data in all_results.items():
        print(f"  {'GGUF ' + name:<30} {data['tps']:>7.1f} {'llama.cpp CUDA':>20}")

    # Our bf16+sparsity results
    print(f"  {'─'*58}")
    for name, data in BF16_RESULTS.items():
        print(f"  {name:<30} {data['tps']:>7.1f} {data['sparsity'] + ' sparse':>20}")

    print(f"\n  GGUF Q4 usa ~4-5 GB VRAM vs ~14.5 GB bf16")
    print(f"  GGUF já tem kernels otimizados para quantização")
    print(f"{'═'*60}\n")

    return 0


if __name__ == "__main__":
    main()
