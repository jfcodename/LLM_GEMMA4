"""
Gemma 4 E4B — INT4 Quantization + Sparsity
=============================================
Usa torchao (nativo PyTorch 2.11+) para quantização INT4.
Sem dependência de bitsandbytes ou CUDA libs externas.

Fallback para INT8 dynamic se INT4 falhar.

Uso no Kaggle:
    !python unified/phase4b_int4_sparse.py
"""

import logging, sys, time, gc
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def benchmark_generate(model, tokenizer, prompts, max_new_tokens=50):
    """Benchmark com chat template."""
    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen_ids = out[0][input_len:]
        n_tok = len(gen_ids)
        tps = n_tok / elapsed if elapsed > 0 else 0
        text_out = tokenizer.decode(gen_ids, skip_special_tokens=True)

        results.append({
            "prompt": prompt[:50],
            "tok_per_s": tps,
            "n_tokens": n_tok,
            "elapsed": elapsed,
            "output": text_out[:120],
        })

        short = prompt[:50].ljust(50)
        print(f"      {tps:.1f} tok/s | {n_tok} tok | {elapsed:.2f}s")
        print(f"      → {text_out[:100]}\n")

    return results


def try_torchao_int4(model):
    """Tenta quantização INT4 via torchao."""
    try:
        import torchao
        from torchao.quantization import int4_weight_only
        logger.info("torchao encontrado, aplicando INT4 weight-only...")
        torchao.quantize_(model, int4_weight_only(group_size=128))
        return True, "torchao INT4"
    except ImportError:
        logger.info("torchao não disponível")
        return False, None
    except Exception as e:
        logger.warning(f"torchao INT4 falhou: {e}")
        return False, None


def try_torch_dynamic_int8(model):
    """Fallback: quantização dinâmica INT8 nativa do PyTorch."""
    try:
        logger.info("Tentando quantização dinâmica INT8 (torch.ao)...")
        quantized = torch.ao.quantization.quantize_dynamic(
            model.language_model if hasattr(model, 'language_model') else model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        if hasattr(model, 'language_model'):
            model.language_model = quantized
        return True, "dynamic INT8"
    except Exception as e:
        logger.warning(f"INT8 dinâmico falhou: {e}")
        return False, None


def try_manual_fp16_quant(model):
    """Fallback 2: converter bf16 → fp16 (menor precisão, sem lib extra)."""
    try:
        logger.info("Convertendo bf16 → fp16...")
        model.half()
        return True, "fp16"
    except Exception as e:
        logger.warning(f"fp16 falhou: {e}")
        return False, None


def try_quanto(model):
    """Tenta quantização via optimum-quanto."""
    try:
        from optimum.quanto import quantize, qint4
        logger.info("quanto encontrado, aplicando qint4...")
        quantize(model, weights=qint4)
        return True, "quanto INT4"
    except ImportError:
        logger.info("optimum-quanto não disponível")
        return False, None
    except Exception as e:
        logger.warning(f"quanto falhou: {e}")
        return False, None


def main():
    if not torch.cuda.is_available():
        logger.error("GPU necessária")
        return 1

    gpu = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n{'═'*60}")
    print(f"  INT4/INT8 QUANTIZATION + SPARSITY")
    print(f"  GPU: {gpu} ({vram_total:.1f} GB)")
    print(f"  PyTorch: {torch.__version__}")
    print(f"{'═'*60}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "google/gemma-4-e4b-it"

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
    ]

    # ═══════════════════════════════════════════════════════════════
    # BASELINE bf16
    # ═══════════════════════════════════════════════════════════════

    logger.info(f"Carregando {model_id} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()

    vram_bf16 = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\n{'─'*60}")
    print(f"  BASELINE bf16 (VRAM: {vram_bf16:.2f} GB)")
    print(f"{'─'*60}")

    bf16_results = benchmark_generate(model, tokenizer, prompts)
    bf16_tps = sum(r["tok_per_s"] for r in bf16_results) / len(bf16_results)

    # ═══════════════════════════════════════════════════════════════
    # TRY QUANTIZATION (cascata de fallbacks)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  TENTANDO QUANTIZAÇÃO...")
    print(f"{'─'*60}")

    # Tentar em ordem de preferência
    quant_success = False
    quant_method = None

    for try_fn in [try_torchao_int4, try_quanto, try_torch_dynamic_int8, try_manual_fp16_quant]:
        success, method = try_fn(model)
        if success:
            quant_success = True
            quant_method = method
            break

    if not quant_success:
        print("  ❌ Nenhum método de quantização funcionou")
        return 1

    torch.cuda.empty_cache()
    vram_quant = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\n  ✅ Quantização: {quant_method}")
    print(f"  VRAM: {vram_quant:.2f} GB (vs {vram_bf16:.2f} bf16)")
    print(f"  Compressão VRAM: {vram_bf16/max(vram_quant,0.1):.1f}×")

    # Benchmark quantizado denso
    print(f"\n{'─'*60}")
    print(f"  {quant_method.upper()} DENSO")
    print(f"{'─'*60}")

    quant_results = benchmark_generate(model, tokenizer, prompts)
    quant_tps = sum(r["tok_per_s"] for r in quant_results) / len(quant_results)

    # ═══════════════════════════════════════════════════════════════
    # QUANTIZADO + TOP-K 50%
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  {quant_method.upper()} + TOP-K 50%")
    print(f"{'─'*60}")

    try:
        from unified.phase1b_topk import TopKMaskedMLP

        wrappers = []
        named_mods = dict(model.named_modules())
        for name, mod in list(named_mods.items()):
            if not hasattr(mod, 'act_fn') or not hasattr(mod, 'gate_proj'):
                continue
            if "vision" in name or "audio" in name:
                continue
            wrapper = TopKMaskedMLP(mod, keep_ratio=0.50)
            parent_parts = name.rsplit(".", 1)
            if len(parent_parts) == 2:
                parent = named_mods.get(parent_parts[0])
                if parent:
                    setattr(parent, parent_parts[1], wrapper)
                    wrappers.append(wrapper)

        logger.info(f"Top-K patched {len(wrappers)} MLPs (keep=50%)")

        for w in wrappers:
            w.reset_stats()

        quant_topk_results = benchmark_generate(model, tokenizer, prompts)
        quant_topk_tps = sum(r["tok_per_s"] for r in quant_topk_results) / len(quant_topk_results)

        # Unpatch
        named_mods2 = dict(model.named_modules())
        for name, mod in list(named_mods2.items()):
            if isinstance(mod, TopKMaskedMLP):
                parent_parts = name.rsplit(".", 1)
                if len(parent_parts) == 2:
                    parent = named_mods2.get(parent_parts[0])
                    if parent:
                        setattr(parent, parent_parts[1], mod.original_mlp)
        logger.info(f"Unpatched {len(wrappers)} MLPs")

    except Exception as e:
        logger.error(f"Top-K patch falhou: {e}")
        quant_topk_tps = 0

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print(f"  RESUMO — QUANTIZAÇÃO + ESPARSIDADE")
    print(f"{'═'*60}")
    print(f"  {'Config':<30} {'tok/s':>8} {'vs bf16':>8} {'VRAM':>10}")
    print(f"  {'─'*58}")
    print(f"  {'bf16 denso':<30} {bf16_tps:>7.1f} {'1.00×':>8} {vram_bf16:>8.1f} GB")
    print(f"  {f'{quant_method} denso':<30} {quant_tps:>7.1f} {quant_tps/bf16_tps:>7.2f}× {vram_quant:>8.1f} GB")
    if quant_topk_tps > 0:
        print(f"  {f'{quant_method} + Top-K 50%':<30} {quant_topk_tps:>7.1f} {quant_topk_tps/bf16_tps:>7.2f}× {vram_quant:>8.1f} GB")
    print(f"\n  Referências anteriores (bf16):")
    print(f"  {'bf16 + Top-K 50%':<30} {'7.1':>8} {'0.84×':>8} {vram_bf16:>8.1f} GB")
    print(f"  {'bf16 + 6:8 + Top-K 50%':<30} {'7.2':>8} {'0.85×':>8} {vram_bf16:>8.1f} GB")
    print(f"{'═'*60}\n")

    return 0


if __name__ == "__main__":
    main()
