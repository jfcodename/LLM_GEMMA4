"""
Gemma 4 E4B — Fase 1b: Top-K MLP Masking (Zero-Training Sparsity)
===================================================================
Mantém GELU original, mas aplica máscara top-K nos neurônios MLP
após ativação. Esparsidade controlável sem quebrar representações.

Fluxo original:  down_proj( GELU(gate_proj(x)) * up_proj(x) )
Fluxo com mask:  down_proj( topk(GELU(gate_proj(x))) * up_proj(x) )

Testa vários níveis de esparsidade (30%, 50%, 65%) e mede:
- Qualidade de output
- Throughput
- Degradação vs baseline

Uso no Kaggle:
    %cd /kaggle/working/LLM_GEMMA4
    !python unified/phase1b_topk.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
# TOP-K MASKED MLP WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class TopKMaskedMLP(nn.Module):
    """
    Wrapper que intercepta o forward do MLP e aplica top-K masking.

    Mantém GELU original → preserva representações aprendidas.
    Aplica máscara DEPOIS de GELU, zerando neurônios de baixa magnitude.

    Args:
        original_mlp: Módulo MLP original do Gemma4
        keep_ratio: Fração de neurônios a manter (1.0 = sem masking)
    """

    def __init__(self, original_mlp: nn.Module, keep_ratio: float = 0.35):
        super().__init__()
        self.mlp = original_mlp
        self.keep_ratio = keep_ratio
        self._sparsity_stats = {"total": 0, "zeros": 0, "calls": 0}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computar gate e up normalmente
        gate_output = self.mlp.act_fn(self.mlp.gate_proj(x))
        up_output = self.mlp.up_proj(x)

        # Top-K masking no gate_output
        if self.keep_ratio < 1.0:
            gate_output = self._apply_topk(gate_output)

        # Hadamard + down
        intermediate = gate_output * up_output
        output = self.mlp.down_proj(intermediate)
        return output

    def _apply_topk(self, x: torch.Tensor) -> torch.Tensor:
        """Mantém top-K neurônios por magnitude, zera o resto."""
        B, T, D = x.shape
        k = max(1, int(D * self.keep_ratio))

        # Top-k por magnitude (abs) — mantém sinal original
        _, topk_indices = torch.topk(x.abs(), k, dim=-1, sorted=False)

        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_indices, 1.0)

        result = x * mask

        # Stats
        with torch.no_grad():
            self._sparsity_stats["total"] += x.numel()
            self._sparsity_stats["zeros"] += (result == 0).sum().item()
            self._sparsity_stats["calls"] += 1

        return result

    @property
    def actual_sparsity(self) -> float:
        if self._sparsity_stats["total"] == 0:
            return 0.0
        return self._sparsity_stats["zeros"] / self._sparsity_stats["total"]

    def reset_stats(self):
        self._sparsity_stats = {"total": 0, "zeros": 0, "calls": 0}


# ─────────────────────────────────────────────────────────────────────────────
# PATCHING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def patch_mlps(model: nn.Module, keep_ratio: float, text_only: bool = True) -> List[TopKMaskedMLP]:
    """
    Substitui módulos MLP por TopKMaskedMLP.

    Args:
        model: Modelo HuggingFace
        keep_ratio: Fração de neurônios a manter
        text_only: Se True, só aplica no text decoder (não vision/audio)

    Returns:
        Lista dos wrappers instalados (para acessar stats)
    """
    wrappers = []
    patched = 0

    for name, module in model.named_modules():
        if not hasattr(module, 'act_fn') or not hasattr(module, 'gate_proj'):
            continue

        # Filtrar: só text decoder se text_only
        if text_only and ("vision" in name or "audio" in name or "embed" in name):
            continue

        # Encontra o parent e o atributo
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name

        wrapper = TopKMaskedMLP(module, keep_ratio=keep_ratio)
        setattr(parent, attr_name, wrapper)
        wrappers.append(wrapper)
        patched += 1

    logger.info(f"Patched {patched} MLPs com keep_ratio={keep_ratio:.0%}")
    return wrappers


def unpatch_mlps(model: nn.Module):
    """Remove wrappers, restaura MLPs originais."""
    unpatched = 0
    for name, module in model.named_modules():
        if isinstance(module, TopKMaskedMLP):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                attr_name = name
            setattr(parent, attr_name, module.mlp)
            unpatched += 1
    logger.info(f"Unpatched {unpatched} MLPs → restaurado GELU original")


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(model, tokenizer, prompts: List[str], max_new_tokens: int = 50) -> List[Dict]:
    """Benchmark com chat template."""
    results = []
    for prompt in prompts:
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
            model.generate(**inputs, max_new_tokens=3, do_sample=False)
        torch.cuda.synchronize()

        # Measure
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen_tokens = out.shape[-1] - input_len
        decoded = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        results.append({
            "prompt": prompt[:50],
            "tokens": gen_tokens,
            "time_s": elapsed,
            "tok_per_s": gen_tokens / elapsed if elapsed > 0 else 0,
            "output": decoded,
        })
    return results


def print_results(label: str, results: List[Dict], sparsity: float = None):
    """Pretty-print benchmark results."""
    avg_tps = sum(r["tok_per_s"] for r in results) / len(results)
    print(f"\n  {label}")
    if sparsity is not None:
        print(f"    Esparsidade real: {sparsity:.1%}")
    print(f"    Throughput médio: {avg_tps:.1f} tok/s")
    print()
    for r in results:
        print(f"    [{r['prompt']}...]")
        print(f"      {r['tok_per_s']:.1f} tok/s | {r['tokens']} tok | {r['time_s']:.2f}s")
        # Mostra output limitado e detecta garbage
        out = r["output"][:150]
        has_garbage = any(ord(c) > 0x3000 for c in out[:50])  # CJK/special chars
        quality = "⚠️ GARBAGE" if has_garbage else "✅ OK"
        print(f"      {quality} → {out}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1b: Top-K MLP Masking")
    parser.add_argument("--model-id", default="google/gemma-4-e4b-it")
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.50, 0.35, 0.25],
                        help="Keep ratios to test (1.0=dense, 0.35=65%% sparse)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("GPU necessária")
        return 1

    gpu = torch.cuda.get_device_name(0)
    print(f"\n{'═'*60}")
    print(f"  FASE 1b: TOP-K MLP MASKING (Zero-Training)")
    print(f"  GPU: {gpu}")
    print(f"  Keep ratios a testar: {args.ratios}")
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

    # ── Test prompts ──────────────────────────────────────────────────────
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
    ]

    # ══════════════════════════════════════════════════════════════════════
    # BASELINE (dense, GELU original)
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  BASELINE: GELU DENSO (keep=100%)")
    print(f"{'─'*60}")

    baseline = benchmark(model, tokenizer, prompts)
    print_results("GELU DENSO (baseline)", baseline, sparsity=0.009)

    baseline_tps = sum(r["tok_per_s"] for r in baseline) / len(baseline)

    # ══════════════════════════════════════════════════════════════════════
    # TEST EACH KEEP RATIO
    # ══════════════════════════════════════════════════════════════════════

    all_results = {"baseline": {"tps": baseline_tps, "sparsity": 0.009}}

    for ratio in sorted(args.ratios, reverse=True):  # Menos agressivo primeiro
        sparsity_target = 1.0 - ratio

        print(f"\n{'─'*60}")
        print(f"  TOP-K: keep={ratio:.0%} (esparsidade alvo: {sparsity_target:.0%})")
        print(f"{'─'*60}")

        # Patch
        wrappers = patch_mlps(model, keep_ratio=ratio, text_only=True)

        # Reset stats
        for w in wrappers:
            w.reset_stats()

        # Benchmark
        results = benchmark(model, tokenizer, prompts)

        # Collect sparsity
        total_sp = sum(w.actual_sparsity * w._sparsity_stats["total"]
                       for w in wrappers if w._sparsity_stats["total"] > 0)
        total_n = sum(w._sparsity_stats["total"] for w in wrappers
                      if w._sparsity_stats["total"] > 0)
        actual_sparsity = total_sp / total_n if total_n > 0 else 0

        print_results(f"TOP-K keep={ratio:.0%}", results, sparsity=actual_sparsity)

        avg_tps = sum(r["tok_per_s"] for r in results) / len(results)
        speedup = avg_tps / baseline_tps if baseline_tps > 0 else 1.0

        all_results[f"topk_{ratio:.0%}"] = {
            "tps": avg_tps,
            "sparsity": actual_sparsity,
            "speedup": speedup,
            "outputs": [r["output"][:100] for r in results],
        }

        # Unpatch para próximo teste
        unpatch_mlps(model)

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print(f"  RESUMO COMPARATIVO")
    print(f"{'═'*60}")
    print(f"  {'Config':<20} {'Sparsity':>10} {'tok/s':>8} {'Speedup':>8} {'Quality':>8}")
    print(f"  {'─'*56}")

    for name, data in all_results.items():
        # Check quality from outputs
        if "outputs" in data:
            has_garbage = any(
                any(ord(c) > 0x3000 for c in out[:30])
                for out in data["outputs"]
            )
            quality = "⚠️ BAD" if has_garbage else "✅ OK"
        else:
            quality = "✅ OK"

        speedup = data.get("speedup", 1.0)
        print(f"  {name:<20} {data['sparsity']:>9.1%} {data['tps']:>7.1f} {speedup:>7.2f}× {quality:>8}")

    print(f"\n  Nota: Speedup real requer kernel esparso (não implementado)")
    print(f"  O ganho virá quando usarmos sparse matmul para pular zeros.")
    print(f"{'═'*60}\n")

    return 0


if __name__ == "__main__":
    main()
