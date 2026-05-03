"""
sparse_gemma4/benchmark/run_benchmark.py
==========================================
Pipeline de benchmark completo: carrega Gemma 4, aplica esparsidade,
mede e compara todas as métricas.

Execução:
  python benchmark/run_benchmark.py \
    --model_path google/gemma-4-pt-2b \
    --policy conservative \
    --max_new_tokens 200 \
    --num_runs 3 \
    --output_dir ./results

Ou em código:
  from benchmark.run_benchmark import BenchmarkRunner
  runner = BenchmarkRunner(model_path="...", policy="conservative")
  results = runner.run_all()
  runner.print_full_report(results)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Adiciona parent ao path para imports relativos funcionarem como script standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.sparsity_policy import (
    CONSERVATIVE_POLICY,
    AGGRESSIVE_POLICY,
    SparsityMode,
)
from core.sparsifier import Gemma4Sparsifier, SparsificationReport
from monitor.profiler import Gemma4Profiler, RunMetrics, estimate_gemma4_flops

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS DE BENCHMARK
# Diversificados para capturar reasoning, geração longa e inferência simples
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_PROMPTS = {
    "short_factual": "What is the capital of France?",
    "math_reasoning": (
        "Solve step by step: A train travels 120km in 1.5 hours. "
        "Another train travels 200km in 2.5 hours. Which is faster and by how much?"
    ),
    "long_generation": (
        "Write a detailed technical explanation of how transformers work, "
        "covering attention mechanisms, positional encoding, and layer normalization."
    ),
    "code_generation": (
        "Write a Python implementation of a binary search tree with insert, "
        "delete, and search operations, including proper error handling."
    ),
    "complex_reasoning": (
        "You have 3 boxes. Box A has 2 red and 3 blue balls. Box B has 4 red and 1 blue ball. "
        "Box C has 1 red and 5 blue balls. You pick a box at random and draw a ball. "
        "It's red. What is the probability it came from Box B? Show all steps."
    ),
}


@dataclass
class BenchmarkResult:
    prompt_name: str
    baseline_metrics: Optional[RunMetrics]
    sparse_24_metrics: Optional[RunMetrics]
    sparse_68_metrics: Optional[RunMetrics]
    sparsification_report: Optional[SparsificationReport]
    
    def speedup_24(self) -> float:
        if self.baseline_metrics and self.sparse_24_metrics:
            if self.sparse_24_metrics.total_time_s > 0:
                return self.baseline_metrics.total_time_s / self.sparse_24_metrics.total_time_s
        return 1.0
    
    def speedup_68(self) -> float:
        if self.baseline_metrics and self.sparse_68_metrics:
            if self.sparse_68_metrics.total_time_s > 0:
                return self.baseline_metrics.total_time_s / self.sparse_68_metrics.total_time_s
        return 1.0


class BenchmarkRunner:
    """
    Orquestra o benchmark completo do Gemma 4 dense vs sparse.
    
    Sequência de execução:
      1. Carrega modelo denso original
      2. Mede baseline (dense)
      3. Aplica política 6:8 — mede sparse_68 (recomendado para reasoning)
      4. Aplica política 2:4 — mede sparse_24 (máximo speedup, risco em reasoning)
      5. Gera relatório comparativo completo
    """

    def __init__(
        self,
        model_path: str,
        policy: str = "conservative",
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 150,
        num_runs: int = 3,           # Runs para média (reduz variância)
        output_dir: str = "./results",
        prompts: Optional[dict] = None,
        skip_dense: bool = False,    # Para testes rápidos
        skip_24: bool = False,
        skip_68: bool = False,
    ):
        self.model_path = model_path
        self.policy_name = policy
        self.policy = CONSERVATIVE_POLICY if policy == "conservative" else AGGRESSIVE_POLICY
        self.device_str = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.num_runs = num_runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = prompts or BENCHMARK_PROMPTS
        self.skip_dense = skip_dense
        self.skip_24 = skip_24
        self.skip_68 = skip_68

        # Lazy load
        self._model = None
        self._tokenizer = None
        self._profiler = None

    def _load_model(self):
        """Carrega modelo e tokenizer com configuração otimizada."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers não instalado: pip install transformers>=4.50")

        logger.info(f"Carregando modelo: {self.model_path}")
        
        device_map = self.device_str if self.device_str != "auto" else "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            # attn_implementation="sdpa",  # Habilitar se suportado
        )
        model.eval()

        try:
            from transformers import AutoProcessor
            tokenizer = AutoProcessor.from_pretrained(self.model_path)
        except Exception:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        return model, tokenizer

    def _warmup(self, model, tokenizer, n_tokens: int = 20) -> None:
        """Aquece o modelo para estabilizar medições de performance."""
        logger.info("Warmup...")
        inputs = tokenizer(
            "Warmup pass.",
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("Warmup concluído.")

    def _run_single(
        self,
        model: nn.Module,
        tokenizer,
        profiler: Gemma4Profiler,
        label: str,
        prompt: str,
        prompt_name: str,
    ) -> RunMetrics:
        """Executa uma única medição com o modelo."""
        full_label = f"{label}_{prompt_name}"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        profiler.begin_run(full_label)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        
        metrics = profiler.end_run(full_label, inputs, outputs)
        profiler.print_summary(full_label)
        return metrics

    def _average_metrics(self, metrics_list: list[RunMetrics], label: str) -> RunMetrics:
        """Calcula média de múltiplos runs (reduz variância de medição)."""
        if len(metrics_list) == 1:
            return metrics_list[0]
        
        avg = RunMetrics(label=label)
        for field_name in [
            "prefill_tps", "decode_tps", "total_time_s",
            "gpu_time_ms", "peak_memory_mb", "model_memory_mb",
            "estimated_flops_gflops", "avg_activation_sparsity",
            "prompt_tokens", "generated_tokens",
        ]:
            vals = [getattr(m, field_name) for m in metrics_list]
            setattr(avg, field_name, sum(vals) / len(vals))
        
        avg.device_name = metrics_list[0].device_name
        avg.torch_version = metrics_list[0].torch_version
        avg.activation_sparsity = metrics_list[-1].activation_sparsity  # Último run
        return avg

    def run_all(self) -> dict[str, BenchmarkResult]:
        """Executa o benchmark completo para todos os prompts e configurações."""
        results: dict[str, BenchmarkResult] = {}
        
        # ── Carrega modelo ────────────────────────────────────────────────────
        model, tokenizer = self._load_model()
        self._model = model
        self._tokenizer = tokenizer

        gpu_id = 0 if torch.cuda.is_available() else None
        
        # ── Warmup ────────────────────────────────────────────────────────────
        self._warmup(model, tokenizer)

        # ── Para cada prompt ──────────────────────────────────────────────────
        for prompt_name, prompt_text in self.prompts.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"BENCHMARK: {prompt_name}")
            logger.info(f"{'='*50}")

            result = BenchmarkResult(
                prompt_name=prompt_name,
                baseline_metrics=None,
                sparse_24_metrics=None,
                sparse_68_metrics=None,
                sparsification_report=None,
            )

            # ── BASELINE DENSO ────────────────────────────────────────────────
            if not self.skip_dense:
                logger.info(f"\n[1/3] Dense baseline — {prompt_name}")
                profiler = Gemma4Profiler(model, tokenizer, device=gpu_id or 0)
                run_metrics_list = []
                for i in range(self.num_runs):
                    logger.info(f"  Run {i+1}/{self.num_runs}")
                    m = self._run_single(model, tokenizer, profiler, "dense", prompt_text, prompt_name)
                    run_metrics_list.append(m)
                result.baseline_metrics = self._average_metrics(run_metrics_list, "dense")

            # ── SPARSE 6:8 (SlideSparse — recomendado) ────────────────────────
            if not self.skip_68:
                logger.info(f"\n[2/3] Sparse 6:8 (SlideSparse conservative) — {prompt_name}")
                
                # Aplica esparsidade 6:8 (in-place no modelo)
                from copy import deepcopy
                model_68 = deepcopy(model)
                
                # Força modo 6:8 para todas as text layers
                policy_68 = {
                    k: v for k, v in self.policy.items()
                    if v.mode != SparsityMode.SKIP
                }
                # Override: todas as eligible → 6:8
                from configs.sparsity_policy import LayerPolicy
                for k in policy_68:
                    if policy_68[k].mode != SparsityMode.SKIP:
                        policy_68[k] = LayerPolicy(
                            mode=SparsityMode.SEMI_68,
                            sparsity_ratio=0.25,
                            note=policy_68[k].note + " [6:8 override]"
                        )

                sparsifier_68 = Gemma4Sparsifier(
                    model_68,
                    policy=policy_68,
                    policy_name="6:8_slidesparse",
                    dtype=self.dtype,
                    use_native_sparse=False,   # 6:8 não tem suporte nativo ainda
                )
                report_68 = sparsifier_68.apply()
                result.sparsification_report = report_68
                logger.info(report_68.summary())

                profiler_68 = Gemma4Profiler(model_68, tokenizer, device=gpu_id or 0)
                run_metrics_68 = []
                for i in range(self.num_runs):
                    logger.info(f"  Run {i+1}/{self.num_runs}")
                    m = self._run_single(model_68, tokenizer, profiler_68, "sparse_68", prompt_text, prompt_name)
                    run_metrics_68.append(m)
                result.sparse_68_metrics = self._average_metrics(run_metrics_68, "sparse_68")

            # ── SPARSE 2:4 (NVIDIA Sparse TC — máximo speedup) ────────────────
            if not self.skip_24:
                logger.info(f"\n[3/3] Sparse 2:4 (NVIDIA Sparse TC) — {prompt_name}")
                
                from copy import deepcopy
                model_24 = deepcopy(model)
                
                sparsifier_24 = Gemma4Sparsifier(
                    model_24,
                    policy=self.policy,
                    policy_name="2:4_nvidia",
                    dtype=self.dtype,
                    use_native_sparse=True,   # Usa to_sparse_semi_structured
                )
                report_24 = sparsifier_24.apply()
                logger.info(report_24.summary())

                profiler_24 = Gemma4Profiler(model_24, tokenizer, device=gpu_id or 0)
                run_metrics_24 = []
                for i in range(self.num_runs):
                    logger.info(f"  Run {i+1}/{self.num_runs}")
                    m = self._run_single(model_24, tokenizer, profiler_24, "sparse_24", prompt_text, prompt_name)
                    run_metrics_24.append(m)
                result.sparse_24_metrics = self._average_metrics(run_metrics_24, "sparse_24")

            results[prompt_name] = result
            
            # Salva progresso parcial
            self._save_partial(results, prompt_name)

        return results

    def _save_partial(self, results: dict, prompt_name: str) -> None:
        """Salva resultados parciais a cada prompt processado."""
        out_path = self.output_dir / f"partial_{prompt_name}.json"
        data = {}
        r = results.get(prompt_name)
        if r:
            for tag, metrics in [
                ("dense", r.baseline_metrics),
                ("sparse_68", r.sparse_68_metrics),
                ("sparse_24", r.sparse_24_metrics),
            ]:
                if metrics:
                    data[tag] = {
                        "decode_tps": metrics.decode_tps,
                        "prefill_tps": metrics.prefill_tps,
                        "total_time_s": metrics.total_time_s,
                        "peak_memory_mb": metrics.peak_memory_mb,
                        "model_memory_mb": metrics.model_memory_mb,
                        "avg_activation_sparsity": metrics.avg_activation_sparsity,
                        "generated_tokens": metrics.generated_tokens,
                        "prompt_tokens": metrics.prompt_tokens,
                    }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    def print_full_report(self, results: dict[str, BenchmarkResult]) -> None:
        """Imprime relatório final consolidado de todos os benchmarks."""
        print(f"\n{'█'*70}")
        print(f"  GEMMA 4 SPARSITY BENCHMARK — RELATÓRIO FINAL")
        print(f"{'█'*70}")

        print(f"\n{'─'*70}")
        print(f"  {'Prompt':<22} {'Dense tok/s':>12} {'6:8 tok/s':>12} {'2:4 tok/s':>12} {'Speedup 6:8':>12} {'Speedup 2:4':>12}")
        print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

        for pname, result in results.items():
            b_tps = result.baseline_metrics.decode_tps if result.baseline_metrics else 0
            s68_tps = result.sparse_68_metrics.decode_tps if result.sparse_68_metrics else 0
            s24_tps = result.sparse_24_metrics.decode_tps if result.sparse_24_metrics else 0
            sp68 = result.speedup_68()
            sp24 = result.speedup_24()
            print(
                f"  {pname:<22} {b_tps:>11.1f} {s68_tps:>11.1f} {s24_tps:>11.1f} "
                f"  {sp68:>10.2f}x   {sp24:>10.2f}x"
            )

        print(f"\n{'─'*70}")
        print("  MEMÓRIA (média)")
        print(f"  {'Prompt':<22} {'Dense MB':>10} {'6:8 MB':>10} {'2:4 MB':>10} {'Δ 6:8':>10} {'Δ 2:4':>10}")
        print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

        for pname, result in results.items():
            b_mem  = result.baseline_metrics.peak_memory_mb if result.baseline_metrics else 0
            s68_mem = result.sparse_68_metrics.peak_memory_mb if result.sparse_68_metrics else 0
            s24_mem = result.sparse_24_metrics.peak_memory_mb if result.sparse_24_metrics else 0
            d68 = ((s68_mem / b_mem) - 1) * 100 if b_mem > 0 else 0
            d24 = ((s24_mem / b_mem) - 1) * 100 if b_mem > 0 else 0
            print(
                f"  {pname:<22} {b_mem:>9.0f} {s68_mem:>9.0f} {s24_mem:>9.0f} "
                f" {d68:>+9.1f}%  {d24:>+9.1f}%"
            )

        # Salva JSON final
        final_path = self.output_dir / "benchmark_final.json"
        self._save_final_json(results, final_path)
        print(f"\n  Resultados salvos em: {final_path}")
        print(f"{'█'*70}\n")

    def _save_final_json(self, results: dict, path: Path) -> None:
        data = {}
        for pname, r in results.items():
            data[pname] = {
                "dense": self._metrics_to_dict(r.baseline_metrics),
                "sparse_68": self._metrics_to_dict(r.sparse_68_metrics),
                "sparse_24": self._metrics_to_dict(r.sparse_24_metrics),
                "speedup_68": r.speedup_68(),
                "speedup_24": r.speedup_24(),
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _metrics_to_dict(self, m: Optional[RunMetrics]) -> Optional[dict]:
        if m is None:
            return None
        return {
            "decode_tps": round(m.decode_tps, 2),
            "prefill_tps": round(m.prefill_tps, 2),
            "total_time_s": round(m.total_time_s, 4),
            "gpu_time_ms": round(m.gpu_time_ms, 2),
            "peak_memory_mb": round(m.peak_memory_mb, 1),
            "model_memory_mb": round(m.model_memory_mb, 1),
            "estimated_flops_gflops": round(m.estimated_flops_gflops, 2),
            "avg_activation_sparsity": round(m.avg_activation_sparsity, 4),
            "prompt_tokens": m.prompt_tokens,
            "generated_tokens": m.generated_tokens,
            "device": m.device_name,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Sparsity Benchmark")
    parser.add_argument("--model_path", type=str, default="google/gemma-4-pt-2b",
                        help="Path ou HuggingFace model ID do Gemma 4")
    parser.add_argument("--policy", choices=["conservative", "aggressive"],
                        default="conservative", help="Política de esparsidade")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Número de runs por configuração para média")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--prompts", type=str, default=None,
                        help="JSON file com prompts customizados")
    parser.add_argument("--skip_dense", action="store_true")
    parser.add_argument("--skip_24", action="store_true")
    parser.add_argument("--skip_68", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Análise sem modificar modelo")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"],
                        default="fp16")
    args = parser.parse_args()

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    custom_prompts = None
    if args.prompts:
        with open(args.prompts) as f:
            custom_prompts = json.load(f)

    runner = BenchmarkRunner(
        model_path=args.model_path,
        policy=args.policy,
        dtype=dtype_map[args.dtype],
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
        output_dir=args.output_dir,
        prompts=custom_prompts,
        skip_dense=args.skip_dense,
        skip_24=args.skip_24,
        skip_68=args.skip_68,
    )

    results = runner.run_all()
    runner.print_full_report(results)


if __name__ == "__main__":
    main()
