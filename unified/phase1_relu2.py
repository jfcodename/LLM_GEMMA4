"""
Gemma 4 E4B — Fase 1: ReLU² Activation Swap
=============================================
Substitui GELU por ReLU² in-place nas 42 camadas MLP do E4B real.
Mede esparsidade de ativação e impacto no throughput.

Uso no Kaggle:
    %cd /kaggle/working/LLM_GEMMA4
    !python unified/phase1_relu2.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ReLU² ACTIVATION
# ─────────────────────────────────────────────────────────────────────────────

class ReLU2Activation(nn.Module):
    """ReLU²(x) = max(0, x)². Drop-in replacement para GELU."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()

    def __repr__(self):
        return "ReLU2()"


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION SPARSITY PROFILER
# ─────────────────────────────────────────────────────────────────────────────

class ActivationProfiler:
    """
    Instrumenta MLPs para medir esparsidade de ativação.
    Registra hooks em act_fn output (post-gate_proj).
    """

    def __init__(self):
        self.stats: Dict[str, Dict] = {}
        self._hooks: List = []

    def attach(self, model: nn.Module):
        """Attach hooks a todas as camadas MLP."""
        for name, module in model.named_modules():
            # Hook nas saídas de gate_proj (pre-hadamard)
            if isinstance(module, nn.Linear) and "gate_proj" in name:
                layer_name = name.rsplit(".", 1)[0]  # Remove .gate_proj
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(layer_name))
                )

    def _make_hook(self, layer_name: str):
        def hook(module, input, output):
            with torch.no_grad():
                # output = act_fn(gate_proj(x)) — queremos medir DEPOIS da ativação
                # Mas o hook é no Linear, antes da ativação.
                # Vamos medir os zeros no output do Linear (pré-ativação)
                # e inferir a esparsidade pós-ativação
                total = output.numel()
                exact_zeros = (output == 0).sum().item()
                # Para ReLU²: valores negativos serão zero
                negative_count = (output < 0).sum().item()
                near_zero = (output.abs() < 1e-6).sum().item()

                self.stats[layer_name] = {
                    "total_activations": total,
                    "exact_zeros": exact_zeros,
                    "exact_zero_pct": exact_zeros / total,
                    "negative_pct": negative_count / total,
                    "near_zero_pct": near_zero / total,
                    # Com ReLU²: negativos + zeros = zeros na saída
                    "relu2_would_zero_pct": (exact_zeros + negative_count) / total,
                }
        return hook

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self) -> Dict:
        if not self.stats:
            return {}

        # Filtrar só text decoder (não vision/audio)
        text_stats = {k: v for k, v in self.stats.items()
                      if "language_model" in k or "model.layers" in k}
        if not text_stats:
            text_stats = self.stats

        avg_exact = sum(v["exact_zero_pct"] for v in text_stats.values()) / len(text_stats)
        avg_neg = sum(v["negative_pct"] for v in text_stats.values()) / len(text_stats)
        avg_relu2 = sum(v["relu2_would_zero_pct"] for v in text_stats.values()) / len(text_stats)

        return {
            "n_layers": len(text_stats),
            "avg_exact_zero": avg_exact,
            "avg_negative": avg_neg,
            "avg_relu2_would_zero": avg_relu2,
        }


# ─────────────────────────────────────────────────────────────────────────────
# POST-ACTIVATION SPARSITY PROFILER
# ─────────────────────────────────────────────────────────────────────────────

class PostActivationProfiler:
    """
    Mede esparsidade DEPOIS da ativação (gate_output * up_output).
    Hook no módulo MLP inteiro para capturar o intermediate state.
    """

    def __init__(self):
        self.stats: Dict[str, Dict] = {}
        self._hooks: List = []
        self._gate_outputs: Dict[str, torch.Tensor] = {}

    def attach(self, model: nn.Module):
        """Attach hooks para capturar gate_proj output pós-ativação."""
        for name, module in model.named_modules():
            if hasattr(module, 'act_fn') and hasattr(module, 'gate_proj'):
                # Este é um módulo MLP com act_fn
                self._hooks.append(
                    module.register_forward_hook(self._make_mlp_hook(name))
                )

    def _make_mlp_hook(self, name: str):
        def hook(module, input, output):
            # Captura intermediate: act_fn(gate_proj(x)) * up_proj(x)
            # Precisamos re-computar porque o hook só vê o output final
            with torch.no_grad():
                x = input[0] if isinstance(input, tuple) else input
                gate_out = module.act_fn(module.gate_proj(x))

                total = gate_out.numel()
                exact_zeros = (gate_out == 0).sum().item()
                near_zero = (gate_out.abs() < 1e-6).sum().item()

                self.stats[name] = {
                    "total": total,
                    "exact_zero_pct": exact_zeros / total,
                    "near_zero_pct": near_zero / total,
                }
        return hook

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self) -> Dict:
        if not self.stats:
            return {}
        text_stats = {k: v for k, v in self.stats.items()
                      if "language_model" in k or "model.layers" in k}
        if not text_stats:
            text_stats = self.stats

        avg_exact = sum(v["exact_zero_pct"] for v in text_stats.values()) / len(text_stats)
        avg_near = sum(v["near_zero_pct"] for v in text_stats.values()) / len(text_stats)
        return {
            "n_layers": len(text_stats),
            "avg_exact_zero": avg_exact,
            "avg_near_zero": avg_near,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SWAP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def swap_activation(model: nn.Module, new_act: nn.Module) -> int:
    """
    Substitui act_fn em todos os módulos MLP do modelo.
    Retorna número de módulos modificados.
    """
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'act_fn'):
            old_act = type(module.act_fn).__name__
            module.act_fn = new_act
            count += 1
            if count <= 3:  # Log primeiros 3
                logger.info(f"  {name}: {old_act} → {new_act}")
    return count


def benchmark_generate(
    model, tokenizer, prompt: str, max_new_tokens: int = 50, n_runs: int = 2
) -> Dict:
    """Benchmark de geração com chat template."""
    chat = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    except Exception:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    # Warmup
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(n_runs):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    gen_tokens = out.shape[-1] - input_len
    avg_time = sum(times) / len(times)
    decoded = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    vram = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "tokens": gen_tokens,
        "time_s": avg_time,
        "tok_per_s": gen_tokens / avg_time if avg_time > 0 else 0,
        "vram_gb": vram,
        "output": decoded,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1: ReLU² Activation Swap")
    parser.add_argument("--model-id", default="google/gemma-4-e4b-it")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline measurement (already have it)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("GPU necessária para Fase 1")
        return 1

    # ── Enviroment ────────────────────────────────────────────────────────
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n{'═'*60}")
    print(f"  FASE 1: ReLU² ACTIVATION SWAP")
    print(f"  GPU: {gpu} ({vram:.1f} GB)")
    print(f"{'═'*60}\n")

    # ── Load model ────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Carregando {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # ── Identify activation function ──────────────────────────────────────
    act_fn_name = "unknown"
    for name, module in model.named_modules():
        if hasattr(module, 'act_fn'):
            act_fn_name = type(module.act_fn).__name__
            break
    logger.info(f"Ativação original: {act_fn_name}")

    # ══════════════════════════════════════════════════════════════════════
    # ETAPA 1: BASELINE (GELU original)
    # ══════════════════════════════════════════════════════════════════════

    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
    ]

    if not args.skip_baseline:
        print(f"\n{'─'*60}")
        print(f"  ETAPA 1: BASELINE (GELU original)")
        print(f"{'─'*60}")

        # Medir esparsidade pré-ativação (negatives = futuro ReLU² zeros)
        profiler = ActivationProfiler()
        profiler.attach(model)

        with torch.inference_mode():
            inputs = tokenizer(test_prompts[0], return_tensors="pt").to(model.device)
            _ = model(**inputs)

        profiler.detach()
        pre_summary = profiler.summary()

        print(f"\n  PRÉ-ATIVAÇÃO (gate_proj output, ANTES de GELU):")
        print(f"    Layers analisados:   {pre_summary.get('n_layers', 0)}")
        print(f"    Zeros exatos:        {pre_summary.get('avg_exact_zero', 0):.1%}")
        print(f"    Valores negativos:   {pre_summary.get('avg_negative', 0):.1%}")
        print(f"    → Com ReLU², esses negativos viram ZEROS:")
        print(f"      Esparsidade prevista: {pre_summary.get('avg_relu2_would_zero', 0):.1%}")

        # Medir esparsidade pós-ativação com GELU
        post_profiler = PostActivationProfiler()
        post_profiler.attach(model)

        with torch.inference_mode():
            _ = model(**inputs)

        post_profiler.detach()
        post_summary = post_profiler.summary()

        print(f"\n  PÓS-ATIVAÇÃO (após GELU):")
        print(f"    Zeros exatos:        {post_summary.get('avg_exact_zero', 0):.1%}")
        print(f"    Near-zero (<1e-6):   {post_summary.get('avg_near_zero', 0):.1%}")

        # Benchmark GELU
        print(f"\n  BENCHMARK GELU:")
        for prompt in test_prompts:
            result = benchmark_generate(model, tokenizer, prompt)
            short_prompt = prompt[:40] + "..." if len(prompt) > 40 else prompt
            print(f"    [{short_prompt}]")
            print(f"      {result['tok_per_s']:.1f} tok/s | {result['tokens']} tokens | {result['time_s']:.2f}s")
            print(f"      → {result['output'][:120]}")
            print()

    # ══════════════════════════════════════════════════════════════════════
    # ETAPA 2: SWAP GELU → ReLU²
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  ETAPA 2: SWAP {act_fn_name} → ReLU²")
    print(f"{'─'*60}")

    relu2 = ReLU2Activation()
    n_swapped = swap_activation(model, relu2)
    logger.info(f"Total: {n_swapped} módulos MLP convertidos para ReLU²")

    # Verificar que o swap funcionou
    for name, module in model.named_modules():
        if hasattr(module, 'act_fn'):
            assert isinstance(module.act_fn, ReLU2Activation), (
                f"{name} ainda tem {type(module.act_fn)}"
            )
            break

    # ══════════════════════════════════════════════════════════════════════
    # ETAPA 3: MEDIR ESPARSIDADE COM ReLU²
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  ETAPA 3: ESPARSIDADE COM ReLU²")
    print(f"{'─'*60}")

    post_profiler2 = PostActivationProfiler()
    post_profiler2.attach(model)

    with torch.inference_mode():
        inputs = tokenizer(test_prompts[0], return_tensors="pt").to(model.device)
        _ = model(**inputs)

    post_profiler2.detach()
    relu2_summary = post_profiler2.summary()

    print(f"\n  PÓS-ATIVAÇÃO (após ReLU²):")
    print(f"    Layers analisados:   {relu2_summary.get('n_layers', 0)}")
    print(f"    Zeros exatos:        {relu2_summary.get('avg_exact_zero', 0):.1%}")
    print(f"    Near-zero (<1e-6):   {relu2_summary.get('avg_near_zero', 0):.1%}")

    # Esparsidade por layer (detalhado)
    if post_profiler2.stats:
        text_stats = {k: v for k, v in sorted(post_profiler2.stats.items())
                      if "language_model" in k}
        if text_stats:
            print(f"\n    Esparsidade por layer:")
            for name, stats in list(text_stats.items())[:10]:
                short = name.split("layers.")[-1] if "layers." in name else name
                bar_len = int(stats["exact_zero_pct"] * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(f"      Layer {short:<8} {bar} {stats['exact_zero_pct']:.1%}")
            if len(text_stats) > 10:
                print(f"      ... ({len(text_stats) - 10} more)")

    # ══════════════════════════════════════════════════════════════════════
    # ETAPA 4: BENCHMARK COM ReLU²
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  ETAPA 4: BENCHMARK COM ReLU²")
    print(f"{'─'*60}")

    print(f"\n  QUALIDADE + VELOCIDADE:")
    for prompt in test_prompts:
        result = benchmark_generate(model, tokenizer, prompt)
        short_prompt = prompt[:40] + "..." if len(prompt) > 40 else prompt
        print(f"    [{short_prompt}]")
        print(f"      {result['tok_per_s']:.1f} tok/s | {result['tokens']} tokens | {result['time_s']:.2f}s")
        print(f"      → {result['output'][:150]}")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # RESUMO
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print(f"  RESUMO FASE 1")
    print(f"{'═'*60}")
    print(f"  Ativação: {act_fn_name} → ReLU²")
    print(f"  Módulos swapped: {n_swapped}")
    if not args.skip_baseline and pre_summary:
        print(f"  Valores negativos (pré-ativação): {pre_summary.get('avg_negative', 0):.1%}")
        print(f"  Esparsidade GELU:  {post_summary.get('avg_exact_zero', 0):.1%}")
    print(f"  Esparsidade ReLU²: {relu2_summary.get('avg_exact_zero', 0):.1%}")
    print(f"  Ganho esperado: {relu2_summary.get('avg_exact_zero', 0):.0%} dos neurônios MLP eliminados")
    print(f"{'═'*60}\n")

    return 0


if __name__ == "__main__":
    main()
