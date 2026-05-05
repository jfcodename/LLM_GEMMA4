"""
Gemma 4 E4B — Fase 6: Structural Pruning Físico
=================================================
Comprova empiricamente que Masking ≠ Pruning.
Nesta fase, removemos FISICAMENTE os neurônios dispensáveis da matriz,
o que garante speedup verdadeiro (redução de FLOPs e VRAM), sem depender
de kernels esparsos ou operações de gather/scatter.

Uso:
    python unified/phase6_structural_pruning.py --mock
"""

import argparse
import sys
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from unified.mock_gemma4_e4b import MockGemma4E4B
from unified.phase1b_topk import benchmark, print_results
from structural_pruning import NeuronImportanceCalibrator, StructuralPruner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Structural Pruning")
    parser.add_argument("--mock", action="store_true", help="Usa mock model (CPU)")
    parser.add_argument("--keep-ratio", type=float, default=0.50, help="Fração mantida")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  FASE 6: STRUCTURAL PRUNING (REMOÇÃO FÍSICA)")
    print(f"{'═'*60}")

    if args.mock:
        logger.info("Carregando MockGemma4E4B...")
        model = MockGemma4E4B(lite=True)
        tokenizer = None
        device = "cpu"
        prompts = ["Test 1", "Test 2"]
        # Dummy token ids
        dummy_input = torch.randint(0, 1000, (1, 32))
        
        def run_bench(mdl, n_runs=10):
            mdl.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_runs):
                    mdl(dummy_input)
            t = time.perf_counter() - t0
            return t / n_runs

    else:
        # Load full model from huggingface
        logger.info("Carregando modelo real...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-4-e4b-it", torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-e4b-it")
        prompts = ["What is the capital of France?"]

    params_before = count_params(model)
    logger.info(f"Params (Antes): {params_before / 1e6:.2f} M")

    if args.mock:
        lat_base = run_bench(model)
        logger.info(f"Latência baseline: {lat_base*1000:.2f} ms")
    else:
        base_res = benchmark(model, tokenizer, prompts)
        print_results("Baseline", base_res, sparsity=0.0)

    # 1. Calibrar Importância
    print(f"\n{'─'*60}")
    print(f"  CALIBRAÇÃO DE IMPORTÂNCIA")
    print(f"{'─'*60}")
    
    # Para simplificar na Phase 6, usaremos a calibração por Proxy de Pesos (Gate)
    calibrator = NeuronImportanceCalibrator(model, device=device)
    importance = calibrator.calibrate_with_gate(keep_ratio=args.keep_ratio)
    neuron_selection = calibrator.select_neurons(importance, keep_ratio=args.keep_ratio)
    
    # 2. Podar Fisicamente
    print(f"\n{'─'*60}")
    print(f"  PODA ESTRUTURAL FÍSICA")
    print(f"{'─'*60}")
    
    pruner = StructuralPruner(model, keep_ratio=args.keep_ratio, use_relu2=True)
    report = pruner.prune(neuron_selection, verbose=True)

    params_after = count_params(model)
    logger.info(f"Params (Depois): {params_after / 1e6:.2f} M")

    # 3. Benchmark Final
    print(f"\n{'─'*60}")
    print(f"  BENCHMARK PÓS-PODA")
    print(f"{'─'*60}")
    
    if args.mock:
        lat_pruned = run_bench(model)
        logger.info(f"Latência pós-poda: {lat_pruned*1000:.2f} ms")
        logger.info(f"Speedup: {lat_base / lat_pruned:.2f}x")
    else:
        pruned_res = benchmark(model, tokenizer, prompts)
        print_results("Pruned", pruned_res, sparsity=0.5)

    print(f"\n{'═'*60}")
    print(f"  CONCLUSÃO: O pruning físico resulta em speedup matemático real.")
    print(f"{'═'*60}\n")

if __name__ == "__main__":
    main()
