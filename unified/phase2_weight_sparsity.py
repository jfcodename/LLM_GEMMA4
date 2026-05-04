"""
Gemma 4 E4B — Fase 2: 6:8 Structured Weight Sparsity
======================================================
Aplica máscaras 6:8 SlideSparse nos pesos MLP do E4B real.
Testa isoladamente e combinado com Top-K activation masking.

Configurações testadas:
1. 6:8 weight sparsity sozinha (25% zeros nos pesos)
2. 6:8 + Top-K 50% ativação (esparsidade combinada)
3. Comparativo com baseline denso

Uso no Kaggle:
    %cd /kaggle/working/LLM_GEMMA4
    !python unified/phase2_weight_sparsity.py
"""

import argparse
import copy
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from unified.modules import magnitude_mask_68, magnitude_mask_24, check_dim_eligibility
from unified.phase1b_topk import TopKMaskedMLP, patch_mlps, unpatch_mlps, benchmark, print_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT SPARSITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def apply_weight_sparsity(
    model: nn.Module,
    mode: str = "6:8",
    text_only: bool = True,
    skip_patterns: List[str] = None,
) -> Dict:
    """
    Aplica esparsidade estruturada nos pesos das camadas lineares.

    Args:
        model: Modelo HuggingFace
        mode: "6:8" (25% zeros) ou "2:4" (50% zeros)
        text_only: Se True, só aplica no text decoder
        skip_patterns: Padrões de nome a pular (e.g., "per_layer", "embed")

    Returns:
        Estatísticas de esparsidade por módulo
    """
    if skip_patterns is None:
        skip_patterns = [
            "per_layer",        # Bottleneck gate — crítico, não esparsificar
            "embed",            # Embeddings
            "norm",             # RMSNorm
            "lm_head",          # Output head
            "pooler",           # Vision pooler
        ]

    mask_fn = magnitude_mask_68 if mode == "6:8" else magnitude_mask_24
    expected_sparsity = 0.25 if mode == "6:8" else 0.50

    stats = {
        "total_params": 0,
        "sparse_params": 0,
        "modules_modified": 0,
        "modules_skipped": 0,
        "per_module": {},
    }

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Filtrar
        if text_only and ("vision" in name or "audio" in name):
            stats["modules_skipped"] += 1
            continue

        if any(skip in name for skip in skip_patterns):
            stats["modules_skipped"] += 1
            continue

        # Verificar elegibilidade dimensional
        eligible, reason = check_dim_eligibility(module.weight.data, mode)
        if not eligible:
            stats["modules_skipped"] += 1
            continue

        # Aplicar máscara
        with torch.no_grad():
            mask = mask_fn(module.weight.data)
            # Converter mask bool para mesmo dtype que weight
            mask_float = mask.to(module.weight.dtype)
            module.weight.data *= mask_float

        n_params = module.weight.numel()
        n_zeros = (module.weight == 0).sum().item()
        actual_sp = n_zeros / n_params

        stats["total_params"] += n_params
        stats["sparse_params"] += n_zeros
        stats["modules_modified"] += 1

        # Guardar short name
        short = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
        stats["per_module"][short] = {
            "params": n_params,
            "zeros": n_zeros,
            "sparsity": actual_sp,
        }

    stats["global_sparsity"] = (
        stats["sparse_params"] / stats["total_params"]
        if stats["total_params"] > 0 else 0
    )

    return stats


def print_sparsity_stats(stats: Dict, mode: str):
    """Pretty-print estatísticas de esparsidade."""
    print(f"\n  WEIGHT SPARSITY ({mode}):")
    print(f"    Módulos modificados: {stats['modules_modified']}")
    print(f"    Módulos pulados:     {stats['modules_skipped']}")
    print(f"    Parâmetros totais:   {stats['total_params']:,}")
    print(f"    Zeros inseridos:     {stats['sparse_params']:,}")
    print(f"    Sparsidade global:   {stats['global_sparsity']:.1%}")

    # Breakdown por tipo de projeção
    gate_sp = [v for k, v in stats["per_module"].items() if "gate" in k]
    up_sp = [v for k, v in stats["per_module"].items() if "up" in k]
    down_sp = [v for k, v in stats["per_module"].items() if "down" in k]
    q_sp = [v for k, v in stats["per_module"].items() if "q_proj" in k]
    kv_sp = [v for k, v in stats["per_module"].items() if "k_proj" in k or "v_proj" in k]
    o_sp = [v for k, v in stats["per_module"].items() if "o_proj" in k]

    print(f"\n    Breakdown por tipo:")
    for label, items in [
        ("gate_proj", gate_sp), ("up_proj", up_sp), ("down_proj", down_sp),
        ("q_proj", q_sp), ("k/v_proj", kv_sp), ("o_proj", o_sp),
    ]:
        if items:
            avg = sum(v["sparsity"] for v in items) / len(items)
            print(f"      {label:<12} {avg:.1%} ({len(items)} layers)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Weight Sparsity")
    parser.add_argument("--model-id", default="google/gemma-4-e4b-it")
    parser.add_argument("--mode", default="6:8", choices=["6:8", "2:4"])
    parser.add_argument("--skip-combined", action="store_true",
                        help="Pular teste combinado (6:8 + Top-K)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("GPU necessária")
        return 1

    gpu = torch.cuda.get_device_name(0)
    print(f"\n{'═'*60}")
    print(f"  FASE 2: {args.mode} STRUCTURED WEIGHT SPARSITY")
    print(f"  GPU: {gpu}")
    print(f"{'═'*60}\n")

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

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
    ]

    # ══════════════════════════════════════════════════════════════════════
    # BASELINE
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  BASELINE: DENSO (sem esparsidade)")
    print(f"{'─'*60}")

    baseline = benchmark(model, tokenizer, prompts)
    print_results("DENSO baseline", baseline, sparsity=0.0)
    baseline_tps = sum(r["tok_per_s"] for r in baseline) / len(baseline)

    # ══════════════════════════════════════════════════════════════════════
    # APPLY 6:8 WEIGHT SPARSITY
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  APLICANDO {args.mode} WEIGHT SPARSITY")
    print(f"{'─'*60}")

    stats = apply_weight_sparsity(model, mode=args.mode, text_only=True)
    print_sparsity_stats(stats, args.mode)

    # Benchmark com pesos esparsos
    print(f"\n{'─'*60}")
    print(f"  BENCHMARK: {args.mode} WEIGHT SPARSITY")
    print(f"{'─'*60}")

    ws_results = benchmark(model, tokenizer, prompts)
    print_results(f"{args.mode} Weight Sparsity", ws_results, sparsity=stats["global_sparsity"])
    ws_tps = sum(r["tok_per_s"] for r in ws_results) / len(ws_results)

    # ══════════════════════════════════════════════════════════════════════
    # COMBINED: 6:8 WEIGHTS + TOP-K 50% ACTIVATIONS
    # ══════════════════════════════════════════════════════════════════════

    if not args.skip_combined:
        print(f"\n{'─'*60}")
        print(f"  COMBINADO: {args.mode} Pesos + Top-K 50% Ativações")
        print(f"{'─'*60}")

        # Pesos já estão esparsos, agora adiciona Top-K
        wrappers = patch_mlps(model, keep_ratio=0.50, text_only=True)
        for w in wrappers:
            w.reset_stats()

        combined_results = benchmark(model, tokenizer, prompts)

        total_sp = sum(w.actual_sparsity * w._sparsity_stats["total"]
                       for w in wrappers if w._sparsity_stats["total"] > 0)
        total_n = sum(w._sparsity_stats["total"] for w in wrappers
                      if w._sparsity_stats["total"] > 0)
        act_sparsity = total_sp / total_n if total_n > 0 else 0

        # Esparsidade efetiva combinada:
        # Pesos 6:8 = 25% zeros + Ativações 50% zeros
        # Efetiva ≈ 1 - (1-0.25)*(1-0.50) = 62.5%
        effective_sp = 1 - (1 - stats["global_sparsity"]) * (1 - act_sparsity)

        print_results(
            f"COMBINADO ({args.mode} + Top-K 50%)",
            combined_results,
            sparsity=effective_sp,
        )
        combined_tps = sum(r["tok_per_s"] for r in combined_results) / len(combined_results)

        unpatch_mlps(model)
    else:
        combined_tps = None
        effective_sp = None

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print(f"  RESUMO FASE 2")
    print(f"{'═'*60}")
    print(f"  {'Config':<30} {'Sparsity':>10} {'tok/s':>8} {'vs base':>8}")
    print(f"  {'─'*58}")
    print(f"  {'Baseline denso':<30} {'0.0%':>10} {baseline_tps:>7.1f} {'1.00×':>8}")
    print(f"  {f'{args.mode} weight':<30} {stats['global_sparsity']:>9.1%} {ws_tps:>7.1f} {ws_tps/baseline_tps:>7.2f}×")

    if combined_tps is not None:
        print(f"  {f'{args.mode} + Top-K 50%':<30} {effective_sp:>9.1%} {combined_tps:>7.1f} {combined_tps/baseline_tps:>7.2f}×")

    print(f"\n  Nota: Speedup real requer sparse matmul kernels.")
    print(f"  Com kernel esparso, expect ~1.3× ({args.mode}) a ~2× (combinado).")
    print(f"{'═'*60}\n")

    return 0


if __name__ == "__main__":
    main()
