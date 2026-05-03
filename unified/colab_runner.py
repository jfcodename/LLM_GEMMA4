"""
Gemma 4 E4B — Colab Runner
============================
Script para rodar toda a suite de testes e validação no Google Colab com T4.

Uso no Colab:
    # Célula 1: Upload do código
    !pip install torch pytest -q
    # Faça upload da pasta unified/ ou clone o repo

    # Célula 2: Rodar testes
    !python unified/colab_runner.py --test

    # Célula 3: Análise com modelo real (opcional, requer ~8GB VRAM)
    !python unified/colab_runner.py --real-model

    # Célula 4: Benchmark completo (requer modelo real)
    !python unified/colab_runner.py --benchmark
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Garante que o diretório pai está no path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import torch

# ─────────────────────────────────────────────────────────────────────────────
# DETECÇÃO DE AMBIENTE
# ─────────────────────────────────────────────────────────────────────────────

def print_environment():
    """Printa informações do ambiente de execução."""
    print(f"\n{'═'*60}")
    print(f"  GEMMA 4 E4B — OPTIMIZATION FRAMEWORK")
    print(f"{'═'*60}")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU:        {gpu}")
        print(f"  VRAM:       {mem:.1f} GB")
        # Check sparse tensor core support
        cc = torch.cuda.get_device_capability(0)
        has_sparse_tc = cc[0] >= 8  # Ampere+
        print(f"  Compute:    {cc[0]}.{cc[1]} {'(Sparse TC ✓)' if has_sparse_tc else '(No Sparse TC)'}")
    else:
        print("  GPU:        None (CPU only)")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODO 1: RODAR TESTES
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    """Roda toda a suite de testes via pytest."""
    import pytest

    print_environment()
    logger.info("Rodando suite de testes...")

    test_dir = str(Path(__file__).parent / "tests")
    exit_code = pytest.main([
        test_dir,
        "-v",
        "--tb=short",
        "-x",  # Para no primeiro erro
    ])

    if exit_code == 0:
        print(f"\n{'✓'*3} TODOS OS TESTES PASSARAM {'✓'*3}\n")
    else:
        print(f"\n{'✗'*3} ALGUNS TESTES FALHARAM {'✗'*3}\n")

    return exit_code


# ─────────────────────────────────────────────────────────────────────────────
# MODO 2: ANÁLISE COM MOCK (sem GPU)
# ─────────────────────────────────────────────────────────────────────────────

def run_mock_analysis():
    """Roda análise completa com MockGemma4E4B."""
    from unified.config import Gemma4E4BConfig
    from unified.mock_gemma4_e4b import MockGemma4E4B
    from unified.modules import (
        ReLU2GatedMLP, SparsityPredictor, magnitude_mask_68, magnitude_mask_24,
    )

    print_environment()
    logger.info("Criando MockGemma4E4B (lite mode)...")

    model = MockGemma4E4B(lite=True)
    params = model.count_params_by_module()

    print(f"\n  DISTRIBUIÇÃO DE PARÂMETROS (lite mock):")
    total = params["total"]
    for name, count in params.items():
        if name != "total":
            pct = count / total * 100
            print(f"    {name:<25} {count/1e6:>8.1f}M  ({pct:5.1f}%)")
    print(f"    {'TOTAL':<25} {total/1e6:>8.1f}M")

    # ── Teste de esparsidade ──────────────────────────────────────────────
    print(f"\n  ESPARSIDADE DE PESOS (6:8):")
    total_p, sparse_p = 0, 0
    for i in range(42):
        layer = model.get_layer(i)
        for name in ["gate_proj", "up_proj", "down_proj"]:
            w = layer.mlp[name].weight.data
            mask = magnitude_mask_68(w)
            layer.mlp[name].weight.data *= mask
            total_p += w.numel()
            sparse_p += (layer.mlp[name].weight == 0).sum().item()

    print(f"    MLP global: {sparse_p/total_p:.1%} zeros ({sparse_p:,} / {total_p:,})")

    # ── Teste ReLU² ───────────────────────────────────────────────────────
    print(f"\n  ESPARSIDADE DE ATIVAÇÃO (ReLU²):")
    hidden = model._dims["hidden"]
    inter = model._dims["intermediate"]

    mlp = ReLU2GatedMLP(hidden_size=hidden, intermediate_size=inter)
    x = torch.randn(4, 32, hidden)
    with torch.no_grad():
        gate_acts = mlp.relu2(mlp.gate_proj(x))
        act_sparsity = (gate_acts == 0).float().mean().item()
    print(f"    ReLU² ativação: {act_sparsity:.1%} zeros")

    # Forward completo
    print(f"\n  FORWARD PASS:")
    x = torch.randint(0, 1024, (1, 32))
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    t1 = time.perf_counter()
    print(f"    Input:  {x.shape}")
    print(f"    Output: {out.shape}")
    print(f"    Time:   {(t1-t0)*1000:.1f}ms")

    print(f"\n{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODO 3: ANÁLISE COM MODELO REAL (requer GPU)
# ─────────────────────────────────────────────────────────────────────────────

def run_real_model_analysis():
    """Carrega gemma-4-e4b-it e aplica otimizações."""
    if not torch.cuda.is_available():
        logger.error("GPU não disponível. Use --test ou --mock para testes sem GPU.")
        return 1

    print_environment()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers não instalado: pip install transformers>=4.50")
        return 1

    from unified.config import Gemma4E4BConfig
    from unified.modules import magnitude_mask_68, ReLU2GatedMLP, SparsityPredictor

    model_id = "google/gemma-4-e4b-it"
    logger.info(f"Carregando {model_id}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # ── Analisar arquitetura real ─────────────────────────────────────────
    print(f"\n  ANÁLISE DA ARQUITETURA REAL:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parâmetros totais: {total_params/1e9:.2f}B")

    # Detectar KV sharing
    kv_own = []
    kv_shared = []
    for i in range(42):
        layer_path = f"model.layers.{i}.self_attn"
        has_k = any(n.startswith(layer_path + ".k_proj") for n, _ in model.named_parameters())
        if has_k:
            kv_own.append(i)
        else:
            kv_shared.append(i)

    print(f"    Layers com KV próprio: {len(kv_own)} ({kv_own[:5]}...)")
    print(f"    Layers com KV shared:  {len(kv_shared)} ({kv_shared[:5]}...)")

    # ── Medir esparsidade natural de ativações ────────────────────────────
    logger.info("Medindo esparsidade natural de ativações...")
    prompt = "Explain the concept of neural network sparsity in detail."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    activation_stats = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                zero_frac = (output == 0).float().mean().item()
                activation_stats[name] = zero_frac
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "mlp" in name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    for h in hooks:
        h.remove()

    if activation_stats:
        avg_sp = sum(activation_stats.values()) / len(activation_stats)
        top5 = sorted(activation_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  ESPARSIDADE NATURAL DE ATIVAÇÕES (MLP):")
        print(f"    Média: {avg_sp:.1%}")
        for name, sp in top5:
            print(f"    {name[-60:]:<60} {sp:.1%}")

    # ── Benchmark rápido ──────────────────────────────────────────────────
    logger.info("Benchmark rápido (baseline)...")
    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    gen_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
    avg_time = sum(times) / len(times)
    tps = gen_tokens / avg_time

    mem_used = torch.cuda.max_memory_allocated() / (1024**3)

    print(f"\n  BENCHMARK BASELINE (dense):")
    print(f"    Tokens gerados: {gen_tokens}")
    print(f"    Tempo médio:    {avg_time:.3f}s")
    print(f"    Throughput:     {tps:.1f} tok/s")
    print(f"    VRAM peak:      {mem_used:.2f} GB")

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"    Output: {decoded[:200]}...")

    print(f"\n{'═'*60}\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 E4B — Colab Runner")
    parser.add_argument("--test", action="store_true", help="Rodar suite de testes")
    parser.add_argument("--mock", action="store_true", help="Análise com mock (sem GPU)")
    parser.add_argument("--real-model", action="store_true", help="Análise com modelo real (requer GPU)")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark completo dense vs sparse")
    parser.add_argument("--all", action="store_true", help="Rodar tudo")
    args = parser.parse_args()

    if not any([args.test, args.mock, args.real_model, args.benchmark, args.all]):
        args.test = True
        args.mock = True

    if args.all or args.test:
        exit_code = run_tests()
        if exit_code != 0 and not args.all:
            sys.exit(exit_code)

    if args.all or args.mock:
        run_mock_analysis()

    if args.all or args.real_model or args.benchmark:
        run_real_model_analysis()


if __name__ == "__main__":
    main()
