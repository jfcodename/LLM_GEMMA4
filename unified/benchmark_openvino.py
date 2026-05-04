"""
Gemma 4 E4B — OpenVINO INT4 CPU Benchmark
=========================================
Script focado na latência máxima em CPUs Intel.
Usa as otimizações nativas AVX2/AVX-512 do OpenVINO e modelo comprimido via NNCF.

Para rodar no Intel i5 (dentro do ambiente Brain/.venv):
    pip install "optimum-intel[openvino]" transformers
    python benchmark_openvino.py
"""

import time
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def main():
    print(f"\n{'═'*60}")
    print(f"  OPENVINO INT4 BENCHMARK (CPU LOCAL)")
    print(f"{'═'*60}\n")

    try:
        from optimum.intel.openvino import OVModelForVisualCausalLM
        from transformers import AutoProcessor
    except ImportError:
        logger.error("optimum-intel não encontrado! Rode: pip install optimum-intel[openvino] transformers")
        return

    # Modelo já comprimido com NNCF INT4 otimizado para OpenVINO
    model_id = "OpenVINO/gemma-4-E4B-it-int4-ov"

    logger.info(f"Baixando/Carregando {model_id}...")
    t0 = time.time()
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # PERFORMANCE_HINT="LATENCY" prioriza time-to-first-token e geração rápida em single-batch
    # CACHE_DIR evita recompilação dos kernels C++ a cada nova execução
    model = OVModelForVisualCausalLM.from_pretrained(
        model_id, 
        device="CPU",
        ov_config={
            "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": "./ov_cache"
        }
    )
    logger.info(f"Modelo carregado em {time.time() - t0:.1f}s")

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers."
    ]

    print(f"\n{'─'*60}")
    print(f"  BENCHMARK")
    print(f"{'─'*60}")

    for prompt in prompts:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt")
        
        input_len = inputs["input_ids"].shape[-1]

        logger.info(f"Iniciando geração para: '{prompt[:40]}...'")
        
        t_gen_start = time.perf_counter()
        
        # Geração
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        elapsed = time.perf_counter() - t_gen_start
        gen_ids = outputs[0][input_len:]
        n_tok = len(gen_ids)
        tps = n_tok / elapsed if elapsed > 0 else 0
        
        text_out = processor.decode(gen_ids, skip_special_tokens=True)

        print(f"      {tps:.1f} tok/s | {n_tok} tok | {elapsed:.2f}s")
        print(f"      → {text_out[:100].replace(chr(10), ' ')}...\n")

if __name__ == "__main__":
    main()
