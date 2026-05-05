"""
Gemma 4 Neo — Pipeline de Pruning Estrutural
==============================================

Script principal que:
  1. Carrega Gemma 4 original
  2. Roda calibração para encontrar os 50% de neurônios mais importantes
  3. Remove fisicamente os outros 50% das weight matrices
  4. Mede speedup e qualidade antes vs depois
  5. Salva o modelo podado

O resultado é um modelo menor que roda nativamente mais rápido em qualquer
hardware — sem kernel especial, sem modo esparso, sem overhead.

Uso:
    # Calibração com dados reais (melhor qualidade)
    python prune_pipeline.py --model google/gemma-4-e4b-it --calibrate --samples 200

    # Calibração rápida com proxy de pesos (sem dados, sem GPU extra)
    python prune_pipeline.py --model google/gemma-4-e4b-it --proxy-calibrate

    # Pruning + benchmark completo
    python prune_pipeline.py --model google/gemma-4-e4b-it --calibrate --benchmark --save ./pruned_e4b

    # Teste rápido (sem dados de calibração, apenas verificação)
    python prune_pipeline.py --model google/gemma-4-e4b-it --proxy-calibrate --keep 0.5
"""

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from structural_pruning import (
    NeuronImportanceCalibrator,
    StructuralPruner,
    PrunedMLP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DADOS DE CALIBRAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

CALIBRATION_TEXTS = [
    # Variados — linguagem natural, código, raciocínio
    "The transformer architecture has revolutionized natural language processing by introducing "
    "the attention mechanism, which allows models to weigh the importance of different words "
    "in a sequence when making predictions.",
    "def fibonacci(n: int) -> list[int]:\n    \"\"\"Returns list of fibonacci numbers up to n.\"\"\"\n"
    "    a, b = 0, 1\n    result = []\n    while a <= n:\n        result.append(a)\n        a, b = b, a + b\n    return result",
    "Quantum entanglement is a phenomenon where two particles become correlated in such a way "
    "that the quantum state of each particle cannot be described independently. Einstein famously "
    "called this 'spooky action at a distance'.",
    "The French Revolution began in 1789 and resulted in the abolition of the monarchy, "
    "the execution of King Louis XVI, and the rise of Napoleon Bonaparte. It fundamentally "
    "changed the political landscape of Europe.",
    "To solve this calculus problem, we first need to find the derivative of f(x) = x³ + 2x² - 5x + 3. "
    "Using the power rule, f'(x) = 3x² + 4x - 5. Setting f'(x) = 0 gives us the critical points.",
    "Machine learning models can be trained using gradient descent, where we iteratively update "
    "the parameters in the direction that minimizes the loss function. The learning rate controls "
    "how large each update step is.",
    "The mitochondria is often called the powerhouse of the cell because it produces ATP through "
    "a process called oxidative phosphorylation. This process converts chemical energy from "
    "glucose into a form that cells can use directly.",
    "In Python, list comprehensions provide a concise way to create lists. For example, "
    "[x**2 for x in range(10) if x % 2 == 0] creates a list of squares of even numbers from 0 to 8.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second. This is "
    "a fundamental constant of nature and plays a crucial role in Einstein's theory of relativity.",
    "Neural networks learn by adjusting their weights through backpropagation. During training, "
    "the gradient of the loss function is computed with respect to each weight, and the weights "
    "are updated to reduce the loss.",
    "The Amazon rainforest is one of the most biodiverse regions on Earth, home to more than "
    "10% of all species. It plays a crucial role in regulating the global climate by absorbing "
    "large amounts of carbon dioxide.",
    "Binary search is an efficient algorithm for finding an element in a sorted array. It works "
    "by repeatedly dividing the search interval in half, giving it O(log n) time complexity.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide "
    "into glucose and oxygen. The equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2",
    "The Renaissance was a period of cultural, artistic, and scientific rebirth that began in "
    "Italy in the 14th century. It marked the transition from the Middle Ages to modernity.",
    "A recursive function is one that calls itself. The classic example is factorial: "
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    "Plate tectonics is the theory that Earth's lithosphere is divided into several large plates "
    "that move relative to each other. This movement causes earthquakes, volcanic eruptions, "
    "and the formation of mountain ranges.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse "
    "equals the sum of squares of the other two sides: a² + b² = c²",
    "Docker containers allow developers to package applications with their dependencies into "
    "isolated environments. This ensures that software runs consistently across different machines.",
    "The Big Bang theory suggests that the universe began as an extremely hot and dense point "
    "approximately 13.8 billion years ago and has been expanding ever since.",
    "Sorting algorithms can be categorized by their time complexity. Bubble sort is O(n²), "
    "merge sort is O(n log n), and radix sort can achieve O(n) in special cases.",
]


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def measure_throughput(
    model: nn.Module,
    tokenizer,
    prompts: list,
    max_new_tokens: int = 100,
    n_warmup: int = 2,
    n_runs: int = 3,
    device: str = "cuda",
) -> dict:
    """
    Mede throughput real (tok/s) e latência.
    Usa greedy decoding para máxima consistência.
    """
    model.eval()

    def _run(prompt: str):
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_len = inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        n_gen = out.shape[-1] - input_len
        return n_gen, elapsed

    # Warmup
    for _ in range(n_warmup):
        _run(prompts[0])

    # Benchmark
    all_tps = []
    all_latencies = []
    all_outputs = []

    for prompt in prompts:
        run_tps = []
        for _ in range(n_runs):
            n_tok, elapsed = _run(prompt)
            run_tps.append(n_tok / elapsed if elapsed > 0 else 0)
            all_latencies.append(elapsed)

        avg_tps = sum(run_tps) / len(run_tps)
        all_tps.append(avg_tps)

        # Captura output para verificar qualidade
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        all_outputs.append({"prompt": prompt, "output": decoded[:200], "tps": avg_tps})

    return {
        "avg_tps": sum(all_tps) / len(all_tps),
        "min_tps": min(all_tps),
        "max_tps": max(all_tps),
        "avg_latency_s": sum(all_latencies) / len(all_latencies),
        "outputs": all_outputs,
    }


def count_mlp_params(model: nn.Module) -> dict:
    """Conta parâmetros nos blocos MLP especificamente."""
    mlp_params = 0
    attn_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        if any(k in name for k in ["gate_proj", "up_proj", "down_proj"]):
            mlp_params += param.numel()
        elif any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            attn_params += param.numel()
        else:
            other_params += param.numel()

    total = mlp_params + attn_params + other_params
    return {
        "mlp": mlp_params,
        "attention": attn_params,
        "other": other_params,
        "total": total,
        "mlp_pct": mlp_params / total * 100,
    }


def print_model_stats(model: nn.Module, label: str = ""):
    """Imprime estatísticas do modelo."""
    params = count_mlp_params(model)
    print(f"\n  {label}")
    print(f"    Total params:     {params['total']/1e9:.3f}B")
    print(f"    MLP params:       {params['mlp']/1e9:.3f}B ({params['mlp_pct']:.1f}%)")
    print(f"    Attention params: {params['attention']/1e9:.3f}B")

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / (1024**3)
        print(f"    VRAM alocada:     {vram:.2f} GB")


def check_intermediate_dims(model: nn.Module) -> None:
    """Verifica as dimensões reais dos MLPs no modelo."""
    try:
        layers = model.model.language_model.layers
    except AttributeError:
        try:
            layers = model.language_model.layers
        except AttributeError:
            layers = model.model.layers

    print(f"\n  Dimensões MLP (primeiras e últimas layers):")
    for i in [0, 1, 5, 10, 20, 30, 40, 41]:
        if i < len(layers):
            layer = layers[i]
            inter = layer.mlp.gate_proj.out_features
            hidden = layer.mlp.gate_proj.in_features
            is_pruned = isinstance(layer.mlp, PrunedMLP)
            tag = "✓ PRUNED" if is_pruned else "original"
            print(f"    Layer {i:02d}: {hidden} → {inter:5d} [{tag}]")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_PROMPTS = [
    "What is the capital of France and why is it historically significant?",
    "Explain how attention mechanisms work in transformer models.",
    "Write a Python function that implements binary search.",
    "What are the main differences between supervised and unsupervised learning?",
    "Describe the process of photosynthesis step by step.",
]


def run_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'═'*65}")
    print(f"  GEMMA 4 — STRUCTURAL PRUNING PIPELINE")
    print(f"{'═'*65}")
    print(f"  Modelo:      {args.model}")
    print(f"  Keep ratio:  {args.keep:.0%} dos neurônios MLP")
    print(f"  Device:      {device}")
    print(f"  ReLU²:       {'sim' if args.relu2 else 'não'}")
    print(f"{'═'*65}\n")

    # ── Carrega modelo ────────────────────────────────────────────────────
    logger.info("Carregando modelo original...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    print_model_stats(model, "ANTES do pruning:")

    # ── Baseline benchmark ────────────────────────────────────────────────
    baseline_results = None
    if args.benchmark:
        print(f"\n{'─'*65}")
        print(f"  BASELINE (modelo original)")
        print(f"{'─'*65}")
        baseline_results = measure_throughput(
            model, tokenizer, BENCHMARK_PROMPTS,
            max_new_tokens=args.max_tokens, device=device,
        )
        print(f"  Throughput médio: {baseline_results['avg_tps']:.1f} tok/s")
        print(f"  Latência média:   {baseline_results['avg_latency_s']:.2f}s")

        # Mostra algumas saídas
        for r in baseline_results["outputs"][:2]:
            print(f"\n  Q: {r['prompt'][:60]}...")
            print(f"  A: {r['output'][:150]}...")

    # ── Calibração de importância ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  CALIBRAÇÃO DE IMPORTÂNCIA NEURONAL")
    print(f"{'─'*65}")

    calibrator = NeuronImportanceCalibrator(model, device=device)

    if args.calibrate:
        # Calibração com dados reais
        texts = CALIBRATION_TEXTS * max(1, args.samples // len(CALIBRATION_TEXTS))
        texts = texts[:args.samples]
        print(f"  Modo: dados reais ({len(texts)} textos)")

        importance = calibrator.calibrate_with_data(
            tokenizer, texts, max_length=512, batch_size=args.batch_size,
        )
    else:
        # Proxy via norma dos pesos (sem dados, sem GPU extra)
        print(f"  Modo: proxy de pesos (sem dados de calibração)")
        importance = calibrator.calibrate_with_gate(keep_ratio=args.keep)

    # Analisa distribuição de importância
    calibrator.analyze_importance_distribution(importance)

    # Seleciona neurônios a manter
    neuron_selection = calibrator.select_neurons(importance, keep_ratio=args.keep)

    # Verifica seleção
    sample_layer = 0
    n_orig = model.model.language_model.layers[0].mlp.gate_proj.out_features
    n_kept = len(neuron_selection.get(sample_layer, []))
    print(f"\n  Exemplo (layer 0): {n_orig} → {n_kept} neurônios "
          f"(-{1-n_kept/n_orig:.0%})")

    # ── Aplica pruning estrutural ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  APLICANDO PRUNING ESTRUTURAL")
    print(f"{'─'*65}")

    pruner = StructuralPruner(model, keep_ratio=args.keep, use_relu2=args.relu2)
    report = pruner.prune(neuron_selection, verbose=True)

    print_model_stats(model, "DEPOIS do pruning:")
    check_intermediate_dims(model)

    # ── Benchmark pós-pruning ─────────────────────────────────────────────
    if args.benchmark:
        print(f"\n{'─'*65}")
        print(f"  BENCHMARK PÓS-PRUNING")
        print(f"{'─'*65}")

        pruned_results = measure_throughput(
            model, tokenizer, BENCHMARK_PROMPTS,
            max_new_tokens=args.max_tokens, device=device,
        )
        print(f"  Throughput médio: {pruned_results['avg_tps']:.1f} tok/s")
        print(f"  Latência média:   {pruned_results['avg_latency_s']:.2f}s")

        # Mostra algumas saídas para verificar qualidade
        print(f"\n  Verificação de qualidade:")
        for r in pruned_results["outputs"][:3]:
            print(f"\n  Q: {r['prompt'][:60]}...")
            print(f"  A: {r['output'][:200]}...")

        # Comparativo
        if baseline_results is not None:
            speedup = pruned_results["avg_tps"] / baseline_results["avg_tps"]
            latency_improvement = baseline_results["avg_latency_s"] / pruned_results["avg_latency_s"]
            print(f"\n{'═'*65}")
            print(f"  COMPARATIVO: ORIGINAL vs PODADO")
            print(f"{'═'*65}")
            print(f"  {'Métrica':<30} {'Original':>10} {'Podado':>10} {'Δ':>8}")
            print(f"  {'─'*58}")
            print(f"  {'Throughput (tok/s)':<30} {baseline_results['avg_tps']:>9.1f} "
                  f"{pruned_results['avg_tps']:>9.1f} {speedup:>+7.2f}×")
            print(f"  {'Latência (s)':<30} {baseline_results['avg_latency_s']:>9.2f} "
                  f"{pruned_results['avg_latency_s']:>9.2f} "
                  f"{latency_improvement:>+7.2f}×")
            print(f"  {'Parâmetros MLP':<30} "
                  f"{report.params_before/1e9:>9.3f}B "
                  f"{report.params_after/1e9:>9.3f}B "
                  f"{report.param_reduction:>+6.1%}")
            print(f"  {'FLOPs MLP (est.)':<30} {'100.0%':>10} "
                  f"{(1-report.flops_reduction)*100:>8.1f}% "
                  f"{-report.flops_reduction:>+6.1%}")
            print(f"{'═'*65}")

    # ── Salva modelo podado ───────────────────────────────────────────────
    if args.save:
        print(f"\n{'─'*65}")
        print(f"  SALVANDO MODELO PODADO")
        print(f"{'─'*65}")
        pruner.save(args.save, tokenizer=tokenizer)

    return model, report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Gemma 4 Structural Pruning Pipeline")
    p.add_argument("--model", default="google/gemma-4-e4b-it",
                   help="Model ID HuggingFace ou caminho local")
    p.add_argument("--keep", type=float, default=0.5,
                   help="Fração de neurônios MLP a manter (0.0–1.0)")
    p.add_argument("--calibrate", action="store_true",
                   help="Usa dados reais para calibração (recomendado)")
    p.add_argument("--proxy-calibrate", action="store_true",
                   help="Calibração via proxy de pesos (sem dados, mais rápido)")
    p.add_argument("--samples", type=int, default=100,
                   help="Número de textos de calibração (--calibrate)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size para calibração")
    p.add_argument("--relu2", action="store_true", default=True,
                   help="Usa ReLU² em vez de GELUTanh no MLP podado")
    p.add_argument("--no-relu2", action="store_false", dest="relu2",
                   help="Mantém GELUTanh original")
    p.add_argument("--benchmark", action="store_true",
                   help="Roda benchmark antes e depois")
    p.add_argument("--max-tokens", type=int, default=100,
                   help="Tokens gerados no benchmark")
    p.add_argument("--save", type=str, default=None,
                   help="Diretório para salvar o modelo podado")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Se nenhum método de calibração especificado, usa proxy por padrão
    if not args.calibrate and not args.proxy_calibrate:
        args.proxy_calibrate = True

    run_pipeline(args)
