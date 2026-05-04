"""
Gemma 4 E4B — INT4 Quantization + Sparsity
=============================================
Carrega modelo diretamente em INT4/NF4 e aplica Top-K 50%.
Sem modelo bf16 em memória (resolve erro de VRAM).

Uso no Kaggle:
    !pip install bitsandbytes -q
    !python unified/phase4b_int4_sparse.py
"""

import logging, sys, time, gc
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from unified.phase1b_topk import (
    TopKMaskedMLP, patch_mlps, unpatch_mlps, benchmark, print_results
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    if not torch.cuda.is_available():
        logger.error("GPU necessária")
        return 1

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n{'═'*60}")
    print(f"  INT4/NF4 QUANTIZATION + SPARSITY")
    print(f"  GPU: {gpu} ({vram:.1f} GB)")
    print(f"{'═'*60}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "google/gemma-4-e4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
    ]

    # ══════════════════════════════════════════════════════════════════
    # LOAD INT4
    # ══════════════════════════════════════════════════════════════════

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Carregando {model_id} direto em INT4/NF4...")
    torch.cuda.empty_cache()
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    vram_used = torch.cuda.max_memory_allocated() / (1024**3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n    Parâmetros: {total_params/1e9:.2f}B")
    print(f"    VRAM INT4:  {vram_used:.2f} GB (vs ~16 GB bf16)")
    print(f"    Compressão: {16/max(vram_used, 0.1):.1f}×")

    # ══════════════════════════════════════════════════════════════════
    # BENCHMARK INT4 DENSO
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  INT4/NF4 DENSO")
    print(f"{'─'*60}")

    int4_dense = benchmark(model, tokenizer, prompts)
    print_results("INT4 denso", int4_dense, sparsity=0.0)
    int4_tps = sum(r["tok_per_s"] for r in int4_dense) / len(int4_dense)

    # ══════════════════════════════════════════════════════════════════
    # INT4 + TOP-K 50%
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  INT4 + TOP-K 50% ATIVAÇÕES")
    print(f"{'─'*60}")

    wrappers = patch_mlps(model, keep_ratio=0.50, text_only=True)
    for w in wrappers:
        w.reset_stats()

    int4_topk = benchmark(model, tokenizer, prompts)

    total_sp = sum(w.actual_sparsity * w._sparsity_stats["total"]
                   for w in wrappers if w._sparsity_stats["total"] > 0)
    total_n = sum(w._sparsity_stats["total"] for w in wrappers
                  if w._sparsity_stats["total"] > 0)
    act_sparsity = total_sp / total_n if total_n > 0 else 0

    print_results("INT4 + Top-K 50%", int4_topk, sparsity=act_sparsity)
    int4_topk_tps = sum(r["tok_per_s"] for r in int4_topk) / len(int4_topk)

    unpatch_mlps(model)

    # ══════════════════════════════════════════════════════════════════
    # INT4 + 6:8 WEIGHT + TOP-K 50%
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  INT4 + 6:8 WEIGHT + TOP-K 50% (FULL STACK)")
    print(f"{'─'*60}")

    # Aplicar 6:8 nos pesos
    from unified.modules import magnitude_mask_68, check_dim_eligibility

    ws_modified = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "vision" in name or "audio" in name:
            continue
        if any(skip in name for skip in ["per_layer", "embed", "norm", "lm_head"]):
            continue
        # bitsandbytes Linear4bit: weight pode ser quantizado
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        try:
            w = module.weight.data
            if w.dim() != 2:
                continue
            eligible, _ = check_dim_eligibility(w, "6:8")
            if not eligible:
                continue
            mask = magnitude_mask_68(w)
            module.weight.data *= mask.to(w.dtype)
            ws_modified += 1
        except Exception:
            continue

    print(f"    6:8 aplicado em {ws_modified} módulos")

    # Top-K 50% em cima
    wrappers = patch_mlps(model, keep_ratio=0.50, text_only=True)
    for w in wrappers:
        w.reset_stats()

    full_stack = benchmark(model, tokenizer, prompts)

    total_sp2 = sum(w.actual_sparsity * w._sparsity_stats["total"]
                    for w in wrappers if w._sparsity_stats["total"] > 0)
    total_n2 = sum(w._sparsity_stats["total"] for w in wrappers
                   if w._sparsity_stats["total"] > 0)
    act_sp2 = total_sp2 / total_n2 if total_n2 > 0 else 0
    effective_sp = 1 - (1 - 0.25) * (1 - act_sp2)

    print_results("INT4 + 6:8 + Top-K 50%", full_stack, sparsity=effective_sp)
    full_tps = sum(r["tok_per_s"] for r in full_stack) / len(full_stack)

    vram_final = torch.cuda.max_memory_allocated() / (1024**3)

    unpatch_mlps(model)

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print(f"  RESUMO — INT4 + ESPARSIDADE")
    print(f"{'═'*60}")
    print(f"  {'Config':<30} {'Sparsity':>10} {'tok/s':>8} {'VRAM':>8}")
    print(f"  {'─'*58}")
    print(f"  {'bf16 denso (ref Phase2)':<30} {'0.0%':>10} {'8.5':>8} {'~16 GB':>8}")
    print(f"  {'INT4 denso':<30} {'0.0%':>10} {int4_tps:>7.1f} {vram_used:>6.1f} GB")
    print(f"  {'INT4 + Top-K 50%':<30} {act_sparsity:>9.1%} {int4_topk_tps:>7.1f} {vram_final:>6.1f} GB")
    print(f"  {'INT4 + 6:8 + Top-K 50%':<30} {effective_sp:>9.1%} {full_tps:>7.1f} {vram_final:>6.1f} GB")
    print(f"{'═'*60}\n")

    return 0


if __name__ == "__main__":
    main()
