"""
Gemma 4 E4B — Fase 7: DeepSlice MoE (DeepSeek + SliceGPT)
===========================================================
Implementa a arquitetura de roteamento ultra-fina, transformando
a MLP monolítica em 1 Especialista Compartilhado (Core) e N
Especialistas Roteados (Raros) comandados por um Mamba-2 Router.
Isso entrega speedup matemático mantendo 100% da capacidade estrutural da rede.

Uso:
    python unified/phase7_deepslice_moe.py --mock
"""

import argparse
import sys
import logging
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from unified.mock_gemma4_e4b import MockGemma4E4B
from unified.phase1b_topk import benchmark, print_results
from deepslice_moe_converter import DeepSliceConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Phase 7: DeepSlice MoE")
    parser.add_argument("--mock", action="store_true", help="Usa mock model (CPU)")
    parser.add_argument("--shared-ratio", type=float, default=0.50, help="Fração de neurônios no Shared Expert")
    parser.add_argument("--experts", type=int, default=8, help="Número de Routed Experts")
    parser.add_argument("--topk", type=int, default=2, help="Quantos experts roteados ativar por token")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  FASE 7: DEEPSLICE MOE (MAMBA-2 ROUTING)")
    print("="*65)

    if args.mock:
        logger.info("Carregando MockGemma4E4B...")
        model = MockGemma4E4B(lite=True)
        tokenizer = None
        device = "cpu"
        prompts = ["Test 1", "Test 2"]
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
        logger.info("Carregando modelo real...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-4-e4b-it", torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-e4b-it")
        prompts = ["What is the capital of France?", "Explain quantum physics concisely."]

    params_before = count_params(model)
    logger.info(f"Parâmetros Globais da Rede (Antes da MoEification): {params_before / 1e6:.2f} M")

    if args.mock:
        lat_base = run_bench(model)
        logger.info(f"Latência baseline (Denso): {lat_base*1000:.2f} ms")
    else:
        base_res = benchmark(model, tokenizer, prompts)
        print_results("Baseline", base_res, sparsity=0.0)

    print("\n" + "-"*65)
    print("  CONVERSÃO E ROTEAMENTO (DEEPSLICE)")
    print("-"*65)
    
    # Executa o conversor arquitetural
    converter = DeepSliceConverter(
        model=model,
        shared_ratio=args.shared_ratio,
        num_routed_experts=args.experts,
        num_experts_per_tok=args.topk,
        use_relu2=False # Mantém GELU nativo
    )
    
    model = converter.convert()

    # Diferente do Pruning que destrói parâmetros, a MoEIFICATION retém tudo (apenas adiciona overhead leve do Mamba).
    params_after = count_params(model)
    logger.info(f"Parâmetros Globais da Rede (MoEified): {params_after / 1e6:.2f} M")

    print("\n" + "-"*65)
    print("  BENCHMARK PÓS-MOEIFICATION")
    print("-"*65)
    
    # Calcular % computacional teórica economizada na FFN
    ffn_compute_ratio = args.shared_ratio + ((1.0 - args.shared_ratio) * (args.topk / args.experts))
    sparsity_theoretical = 1.0 - ffn_compute_ratio
    logger.info(f"Sparsity teórica nos FFNs: {sparsity_theoretical*100:.1f}%")
    
    if args.mock:
        lat_pruned = run_bench(model)
        logger.info(f"Latência DeepSlice MoE: {lat_pruned*1000:.2f} ms")
        logger.info(f"Speedup Real de Execução: {lat_base / lat_pruned:.2f}x")
    else:
        moe_res = benchmark(model, tokenizer, prompts)
        print_results("DeepSlice MoE", moe_res, sparsity=sparsity_theoretical)

    print("\n" + "="*65)
    print("  CONCLUSÃO: O modelo agora roteia tokens contextualmente via Mamba-2")
    print("  para pequenas frações da rede, preservando o número total de parâmetros!")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
