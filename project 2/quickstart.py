"""
sparse_gemma4/quickstart.py
=============================
Script de análise dry-run: não requer GPU, não requer o modelo completo.
Demonstra o pipeline completo e estima métricas a partir da arquitetura.

Execução:
  python quickstart.py                    # Análise teórica completa
  python quickstart.py --model_path ...   # Com modelo real (requer GPU)
"""

import sys
import logging
import argparse
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# SIMULAÇÃO DA ARQUITETURA GEMMA 4 (para análise sem o modelo real)
# ─────────────────────────────────────────────────────────────────────────────

class MockGemma4ClippableLinear(nn.Module):
    """Simula o Gemma4ClippableLinear do vision/audio towers."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)
    def forward(self, x):
        return self.linear(x)


class MockGemma4TextDecoderLayer(nn.Module):
    """Simula uma layer do text decoder do Gemma 4."""
    def __init__(self, hidden=2560, intermediate=10240, q_dim=2048, kv_dim=512):
        super().__init__()
        # Atenção
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(hidden, q_dim, bias=False),
            "k_proj": nn.Linear(hidden, kv_dim, bias=False),
            "v_proj": nn.Linear(hidden, kv_dim, bias=False),
            "o_proj": nn.Linear(q_dim, hidden, bias=False),
        })
        # FFN
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(hidden, intermediate, bias=False),
            "up_proj":   nn.Linear(hidden, intermediate, bias=False),
            "down_proj": nn.Linear(intermediate, hidden, bias=False),
        })
        # per_layer (AltUp/LAuReL)
        self.per_layer_input_gate  = nn.Linear(hidden, 256, bias=False)
        self.per_layer_projection  = nn.Linear(256, hidden, bias=False)
        # Norms (sem pesos significativos)
        self.input_layernorm = nn.LayerNorm(hidden)


class MockGemma4VisionLayer(nn.Module):
    """Simula uma VisionEncoderLayer com ClippableLinear."""
    def __init__(self, dim=768, intermediate=3072):
        super().__init__()
        self.self_attn = nn.ModuleDict({
            "q_proj": MockGemma4ClippableLinear(dim, dim),
            "k_proj": MockGemma4ClippableLinear(dim, dim),
            "v_proj": MockGemma4ClippableLinear(dim, dim),
            "o_proj": MockGemma4ClippableLinear(dim, dim),
        })
        self.mlp = nn.ModuleDict({
            "gate_proj": MockGemma4ClippableLinear(dim, intermediate),
            "up_proj":   MockGemma4ClippableLinear(dim, intermediate),
            "down_proj": MockGemma4ClippableLinear(intermediate, dim),
        })


class MockGemma4AudioLayer(nn.Module):
    """Simula uma AudioLayer (Conformer-like) com ClippableLinear."""
    def __init__(self, dim=1024, intermediate=4096):
        super().__init__()
        for i in (1, 2):
            ff = nn.ModuleDict({
                "ffw_layer_1": MockGemma4ClippableLinear(dim, intermediate),
                "ffw_layer_2": MockGemma4ClippableLinear(intermediate, dim),
            })
            setattr(self, f"feed_forward{i}", ff)
        
        self.self_attn = nn.ModuleDict({
            "q_proj": MockGemma4ClippableLinear(dim, dim),
            "k_proj": MockGemma4ClippableLinear(dim, dim),
            "v_proj": MockGemma4ClippableLinear(dim, dim),
            "post":   MockGemma4ClippableLinear(dim, dim),
        })
        # Conv não esparsificável
        self.lconv1d = nn.Conv1d(dim, dim, kernel_size=5, groups=dim, padding=2, bias=False)


class MockGemma4(nn.Module):
    """Modelo mock completo do Gemma 4 para análise sem pesos reais."""
    def __init__(self):
        super().__init__()
        HIDDEN = 2560
        INTER  = 10240
        VOCAB  = 262144

        # Embeddings
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head      = nn.Linear(HIDDEN, VOCAB, bias=False)
        
        # Text decoder: 42 layers (padrão: 5 local + 1 global, repetido)
        layers = []
        for i in range(42):
            is_global = (i % 6 == 5)
            q_dim  = 4096 if is_global else 2048
            kv_dim = 1024 if is_global else 512
            layers.append(MockGemma4TextDecoderLayer(HIDDEN, INTER, q_dim, kv_dim))
        
        self.language_model = nn.ModuleDict({
            "layers": nn.ModuleList(layers)
        })

        # Vision tower: 16 layers
        self.vision_tower = nn.ModuleDict({
            "encoder": nn.ModuleDict({
                "layers": nn.ModuleList([MockGemma4VisionLayer() for _ in range(16)])
            })
        })

        # Audio tower: 12 layers
        self.audio_tower = nn.ModuleDict({
            "layers": nn.ModuleList([MockGemma4AudioLayer() for _ in range(12)])
        })

    def forward(self, x):
        return x  # Mock — sem forward real


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISE E DEMONSTRAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(model: nn.Module, policy_name: str = "conservative") -> None:
    """Executa análise completa de esparsidade no modelo."""
    from configs.sparsity_policy import CONSERVATIVE_POLICY
    from core.sparsifier import Gemma4Sparsifier
    from monitor.profiler import estimate_gemma4_flops

    print("\n" + "═"*65)
    print("  GEMMA 4 — ANÁLISE DE ESPARSIDADE (DRY RUN)")
    print("═"*65)

    # ── Parâmetros totais por módulo ──────────────────────────────────────────
    print("\n  DISTRIBUIÇÃO DE PARÂMETROS:")
    modules = {
        "Text decoder (42 layers)": model.language_model,
        "Vision tower (16 layers)": model.vision_tower,
        "Audio tower (12 layers)":  model.audio_tower,
        "Embeddings + LM head":     nn.ModuleDict({"embed": model.embed_tokens, "head": model.lm_head}),
    }
    total_params = sum(p.numel() for p in model.parameters())
    for name, mod in modules.items():
        params = sum(p.numel() for p in mod.parameters())
        pct = params / total_params * 100
        print(f"    {name:<35} {params/1e6:>8.1f}M  ({pct:5.1f}%)")
    print(f"    {'TOTAL':<35} {total_params/1e6:>8.1f}M")

    # ── Dry run do sparsifier ─────────────────────────────────────────────────
    print("\n  APLICANDO SPARSIFIER (dry_run=True) ...")
    sparsifier = Gemma4Sparsifier(
        model,
        policy=CONSERVATIVE_POLICY,
        policy_name=policy_name,
        dry_run=True,
        use_native_sparse=False,
    )
    report = sparsifier.apply()
    print(report.summary())

    # ── Estimativa de FLOPs ───────────────────────────────────────────────────
    print("  ESTIMATIVA DE FLOPs (text decoder):")
    for seq in [128, 512, 2048, 8192]:
        info = estimate_gemma4_flops(seq_len=seq)
        print(
            f"    seq_len={seq:5d} | Total={info['total_gflops']:8.1f} GFLOPs "
            f"| FFN={info['ffn_pct']:.0f}% | Attn={100-info['ffn_pct']:.0f}%"
        )

    # ── Estimativa de speedup por abordagem ───────────────────────────────────
    print("\n  SPEEDUP TEÓRICO ESPERADO (Ampere A100/H100):")
    print(f"    {'Abordagem':<30} {'Sparsidade':>12} {'Speedup Kernel':>15} {'Impacto Acurácia':>18}")
    print(f"    {'─'*30} {'─'*12} {'─'*15} {'─'*18}")
    approaches = [
        ("Dense (baseline)",         "0%",    "1.00x",  "—"),
        ("2:4 NVIDIA Sparse TC",     "50%",   "~2.00x", "Alto risco (reasoning)"),
        ("6:8 SlideSparse",          "25%",   "~1.33x", "Mínimo (~1-5%)"),
        ("8:16 (near-future HW)",    "50%",   "~1.5x",  "Baixo"),
        ("Unstructured L1 (>90%)",   ">90%",  "var.*",  "Mínimo"),
    ]
    for name, sp, su, acc in approaches:
        print(f"    {name:<30} {sp:>12} {su:>15} {acc:>18}")
    print("    * Requer kernels CUDA customizados (Sakana AI)")

    # ── Próximos passos ───────────────────────────────────────────────────────
    print("\n  PRÓXIMOS PASSOS RECOMENDADOS:")
    steps = [
        "1. Aplicar esparsidade 6:8 ao Gemma 4 treinado (política conservative)",
        "2. Medir degradação de acurácia em benchmarks: MMLU, GSM8K, HumanEval",
        "3. Se degradação > 3%: aplicar fine-tuning de 100-500 steps com máscara fixa",
        "4. Para 2:4: usar apenas nas vision/audio towers (mais tolerantes)",
        "5. Integrar SlideSparse quando código open-source disponível (arxiv 2603.05232)",
        "6. Considerar quantização FP8 combinada com 6:8 para multiplicação de ganhos",
    ]
    for s in steps:
        print(f"    {s}")

    print("\n" + "═"*65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Sparsity Quickstart")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path HuggingFace do Gemma 4 (opcional; usa mock se omitido)")
    parser.add_argument("--policy", choices=["conservative", "aggressive"],
                        default="conservative")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    if args.model_path:
        # Carrega modelo real
        logger.info(f"Carregando modelo real: {args.model_path}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            run_analysis(model, args.policy)

            # Roda benchmark rápido
            from benchmark.run_benchmark import BenchmarkRunner
            runner = BenchmarkRunner(
                model_path=args.model_path,
                policy=args.policy,
                max_new_tokens=args.max_new_tokens,
                num_runs=1,
                prompts={"quick_test": "Explain what sparsity means in neural networks."},
                skip_24=False,
                skip_68=False,
            )
            results = runner.run_all()
            runner.print_full_report(results)

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            logger.info("Usando modelo mock para análise")
            model = MockGemma4()
            run_analysis(model, args.policy)
    else:
        # Usa mock — não requer GPU nem modelo baixado
        logger.info("Nenhum model_path fornecido — usando MockGemma4 para análise teórica")
        model = MockGemma4()
        run_analysis(model, args.policy)


if __name__ == "__main__":
    main()
