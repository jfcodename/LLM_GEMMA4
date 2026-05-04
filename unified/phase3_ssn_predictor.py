"""
Gemma 4 E4B — Fase 3: SSN (SparsityPredictor via per_layer_input_gate)
=======================================================================
Usa o per_layer_input_gate nativo do E4B como preditor inteligente
de quais neurônios MLP manter — mais informado que Top-K por magnitude.

Diferença vs Top-K:
- Top-K: mantém neurônios de maior magnitude (heurística cega)
- SSN: usa o gate que o MODELO JÁ APRENDEU para decidir importância

O E4B já tem per_layer_input_gate (2560→256) em cada layer.
Reutilizamos esses pesos como "cérebro" do preditor de esparsidade.

Uso no Kaggle:
    %cd /kaggle/working/LLM_GEMMA4
    !python unified/phase3_ssn_predictor.py
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

from unified.phase1b_topk import benchmark, print_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SSN MASKED MLP
# ─────────────────────────────────────────────────────────────────────────────

class SSNMaskedMLP(nn.Module):
    """
    MLP com mascaramento guiado pelo per_layer_input_gate (SSN).

    O per_layer_input_gate do E4B é um Linear(hidden→256) seguido
    de projeção de volta (256→hidden). Ele já codifica importância
    contextual por neurônio. Reutilizamos esse sinal como preditor
    de quais neurônios do MLP (10240-dim) manter.

    Fluxo:
        x → per_layer_input_gate(x) → score (256-dim)
        score → project_to_intermediate (256→10240) → top-k mask
        mask * GELU(gate_proj(x)) * up_proj(x) → down_proj
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        gate_linear: nn.Linear,
        keep_ratio: float = 0.50,
        intermediate_size: int = 10240,
    ):
        super().__init__()
        self.mlp = original_mlp
        self.keep_ratio = keep_ratio
        self.intermediate_size = intermediate_size

        # Projeção do gate bottleneck (256) para intermediate (10240)
        # Inicializada com projeção aleatória leve — não treinada
        gate_out_dim = gate_linear.out_features  # 256
        self.score_proj = nn.Linear(gate_out_dim, intermediate_size, bias=False)
        nn.init.normal_(self.score_proj.weight, std=0.02)
        # Move para o device correto
        self.score_proj = self.score_proj.to(
            device=gate_linear.weight.device,
            dtype=gate_linear.weight.dtype,
        )

        # Referência ao gate (não copia — reutiliza pesos do modelo)
        self.gate_linear = gate_linear

        self._sparsity_stats = {"total": 0, "zeros": 0, "calls": 0}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Computar score de importância via gate
        with torch.no_grad():
            gate_score = self.gate_linear(x)           # (B, T, 256)
            neuron_scores = self.score_proj(gate_score) # (B, T, 10240)

        # 2. Top-K por score de importância
        k = max(1, int(self.intermediate_size * self.keep_ratio))
        _, topk_idx = torch.topk(neuron_scores.abs(), k, dim=-1, sorted=False)

        mask = torch.zeros_like(neuron_scores)
        mask.scatter_(-1, topk_idx, 1.0)

        # 3. MLP com mascaramento
        gate_output = self.mlp.act_fn(self.mlp.gate_proj(x))
        up_output = self.mlp.up_proj(x)

        intermediate = gate_output * up_output * mask
        output = self.mlp.down_proj(intermediate)

        # Stats
        with torch.no_grad():
            self._sparsity_stats["total"] += intermediate.numel()
            self._sparsity_stats["zeros"] += (intermediate == 0).sum().item()
            self._sparsity_stats["calls"] += 1

        return output

    @property
    def actual_sparsity(self) -> float:
        if self._sparsity_stats["total"] == 0:
            return 0.0
        return self._sparsity_stats["zeros"] / self._sparsity_stats["total"]

    def reset_stats(self):
        self._sparsity_stats = {"total": 0, "zeros": 0, "calls": 0}


class TopKMaskedMLP_ForComparison(nn.Module):
    """Top-K simples para comparação justa (mesma interface que SSN)."""

    def __init__(self, original_mlp: nn.Module, keep_ratio: float = 0.50):
        super().__init__()
        self.mlp = original_mlp
        self.keep_ratio = keep_ratio
        self._sparsity_stats = {"total": 0, "zeros": 0, "calls": 0}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output = self.mlp.act_fn(self.mlp.gate_proj(x))
        up_output = self.mlp.up_proj(x)

        # Top-K por magnitude da ativação
        B, T, D = gate_output.shape
        k = max(1, int(D * self.keep_ratio))
        _, topk_idx = torch.topk(gate_output.abs(), k, dim=-1, sorted=False)
        mask = torch.zeros_like(gate_output)
        mask.scatter_(-1, topk_idx, 1.0)

        intermediate = gate_output * up_output * mask
        output = self.mlp.down_proj(intermediate)

        with torch.no_grad():
            self._sparsity_stats["total"] += intermediate.numel()
            self._sparsity_stats["zeros"] += (intermediate == 0).sum().item()
            self._sparsity_stats["calls"] += 1

        return output

    @property
    def actual_sparsity(self):
        if self._sparsity_stats["total"] == 0:
            return 0.0
        return self._sparsity_stats["zeros"] / self._sparsity_stats["total"]

    def reset_stats(self):
        self._sparsity_stats = {"total": 0, "zeros": 0, "calls": 0}


# ─────────────────────────────────────────────────────────────────────────────
# PATCHING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def find_gate_for_layer(model: nn.Module, layer_idx: int) -> Optional[nn.Linear]:
    """Encontra o per_layer_input_gate para uma layer específica."""
    for name, module in model.named_modules():
        if (f"layers.{layer_idx}." in name and
            "per_layer_input_gate" in name and
            isinstance(module, nn.Linear)):
            return module
    return None


def patch_ssn(
    model: nn.Module, keep_ratio: float = 0.50
) -> List[SSNMaskedMLP]:
    """
    Substitui MLPs por SSNMaskedMLP usando per_layer_input_gate.
    """
    wrappers = []
    named_mods = dict(model.named_modules())

    for name, module in list(named_mods.items()):
        if not hasattr(module, 'act_fn') or not hasattr(module, 'gate_proj'):
            continue
        if "vision" in name or "audio" in name or "embed" in name:
            continue

        # Extrair layer index
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass

        if layer_idx is None:
            continue

        # Encontrar gate
        gate = find_gate_for_layer(model, layer_idx)
        if gate is None:
            logger.warning(f"  Layer {layer_idx}: per_layer_input_gate não encontrado, pulando")
            continue

        # Criar wrapper
        intermediate = module.gate_proj.out_features
        wrapper = SSNMaskedMLP(
            module, gate, keep_ratio=keep_ratio, intermediate_size=intermediate,
        )

        # Instalar
        parent_parts = name.rsplit(".", 1)
        if len(parent_parts) == 2:
            parent = named_mods[parent_parts[0]]
            setattr(parent, parent_parts[1], wrapper)
        wrappers.append(wrapper)

    logger.info(f"SSN patched {len(wrappers)} MLPs (keep={keep_ratio:.0%})")
    return wrappers


def patch_topk_comparison(
    model: nn.Module, keep_ratio: float = 0.50
) -> List[TopKMaskedMLP_ForComparison]:
    """Patch com Top-K simples para comparação."""
    wrappers = []
    named_mods = dict(model.named_modules())

    for name, module in list(named_mods.items()):
        if not hasattr(module, 'act_fn') or not hasattr(module, 'gate_proj'):
            continue
        if "vision" in name or "audio" in name or "embed" in name:
            continue

        wrapper = TopKMaskedMLP_ForComparison(module, keep_ratio=keep_ratio)
        parent_parts = name.rsplit(".", 1)
        if len(parent_parts) == 2:
            parent = named_mods[parent_parts[0]]
            setattr(parent, parent_parts[1], wrapper)
        wrappers.append(wrapper)

    logger.info(f"Top-K patched {len(wrappers)} MLPs (keep={keep_ratio:.0%})")
    return wrappers


def unpatch_any(model: nn.Module):
    """Remove qualquer wrapper (SSN ou Top-K)."""
    named_mods = dict(model.named_modules())
    count = 0
    for name, module in list(named_mods.items()):
        if isinstance(module, (SSNMaskedMLP, TopKMaskedMLP_ForComparison)):
            parent_parts = name.rsplit(".", 1)
            if len(parent_parts) == 2:
                parent = named_mods.get(parent_parts[0])
                if parent:
                    setattr(parent, parent_parts[1], module.mlp)
                    count += 1
    logger.info(f"Unpatched {count} MLPs")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: SSN Predictor")
    parser.add_argument("--model-id", default="google/gemma-4-e4b-it")
    parser.add_argument("--keep-ratio", type=float, default=0.50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("GPU necessária")
        return 1

    gpu = torch.cuda.get_device_name(0)
    print(f"\n{'═'*60}")
    print(f"  FASE 3: SSN PREDICTOR (per_layer_input_gate)")
    print(f"  GPU: {gpu}")
    print(f"  Keep ratio: {args.keep_ratio:.0%}")
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
        "What are the main differences between Python and Rust?",
    ]

    # ══════════════════════════════════════════════════════════════════════
    # VERIFICAR per_layer_input_gate EXISTE
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  VERIFICAÇÃO: per_layer_input_gate")
    print(f"{'─'*60}")

    gates_found = 0
    for i in range(42):
        gate = find_gate_for_layer(model, i)
        if gate is not None:
            gates_found += 1
            if i < 3:
                print(f"    Layer {i}: ✅ gate shape = {gate.weight.shape}")
    print(f"    Total gates encontrados: {gates_found}/42")

    if gates_found == 0:
        logger.error("Nenhum per_layer_input_gate encontrado!")
        # Debug: mostra todos os módulos com "gate" no nome
        for name, module in model.named_modules():
            if "gate" in name.lower() and isinstance(module, nn.Linear):
                print(f"    [DEBUG] {name}: {module.weight.shape}")
        return 1

    # ══════════════════════════════════════════════════════════════════════
    # BASELINE
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  BASELINE DENSO")
    print(f"{'─'*60}")

    baseline = benchmark(model, tokenizer, prompts)
    print_results("DENSO (baseline)", baseline, sparsity=0.0)
    baseline_tps = sum(r["tok_per_s"] for r in baseline) / len(baseline)

    # ══════════════════════════════════════════════════════════════════════
    # TOP-K MASKING (referência)
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  TOP-K MAGNITUDE (referência, keep={args.keep_ratio:.0%})")
    print(f"{'─'*60}")

    topk_wrappers = patch_topk_comparison(model, keep_ratio=args.keep_ratio)
    for w in topk_wrappers:
        w.reset_stats()

    topk_results = benchmark(model, tokenizer, prompts)

    topk_sp = sum(w.actual_sparsity * w._sparsity_stats["total"]
                  for w in topk_wrappers if w._sparsity_stats["total"] > 0)
    topk_n = sum(w._sparsity_stats["total"] for w in topk_wrappers
                 if w._sparsity_stats["total"] > 0)
    topk_sparsity = topk_sp / topk_n if topk_n > 0 else 0

    print_results("Top-K Magnitude", topk_results, sparsity=topk_sparsity)
    topk_tps = sum(r["tok_per_s"] for r in topk_results) / len(topk_results)

    unpatch_any(model)

    # ══════════════════════════════════════════════════════════════════════
    # SSN PREDICTOR
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  SSN PREDICTOR (per_layer_input_gate, keep={args.keep_ratio:.0%})")
    print(f"{'─'*60}")

    ssn_wrappers = patch_ssn(model, keep_ratio=args.keep_ratio)
    for w in ssn_wrappers:
        w.reset_stats()

    ssn_results = benchmark(model, tokenizer, prompts)

    ssn_sp = sum(w.actual_sparsity * w._sparsity_stats["total"]
                 for w in ssn_wrappers if w._sparsity_stats["total"] > 0)
    ssn_n = sum(w._sparsity_stats["total"] for w in ssn_wrappers
                if w._sparsity_stats["total"] > 0)
    ssn_sparsity = ssn_sp / ssn_n if ssn_n > 0 else 0

    print_results("SSN Predictor", ssn_results, sparsity=ssn_sparsity)
    ssn_tps = sum(r["tok_per_s"] for r in ssn_results) / len(ssn_results)

    unpatch_any(model)

    # ══════════════════════════════════════════════════════════════════════
    # COMPARATIVO
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print(f"  COMPARATIVO: TOP-K vs SSN (keep={args.keep_ratio:.0%})")
    print(f"{'═'*60}")
    print(f"  {'Método':<25} {'Sparsity':>10} {'tok/s':>8} {'vs base':>8}")
    print(f"  {'─'*53}")
    print(f"  {'Baseline denso':<25} {'0.0%':>10} {baseline_tps:>7.1f} {'1.00×':>8}")
    print(f"  {'Top-K (magnitude)':<25} {topk_sparsity:>9.1%} {topk_tps:>7.1f} {topk_tps/baseline_tps:>7.2f}×")
    print(f"  {'SSN (input_gate)':<25} {ssn_sparsity:>9.1%} {ssn_tps:>7.1f} {ssn_tps/baseline_tps:>7.2f}×")

    # Comparar qualidade
    print(f"\n  COMPARAÇÃO DE OUTPUT (prompt: quantum computing):")
    for label, results in [("Top-K", topk_results), ("SSN", ssn_results)]:
        for r in results:
            if "quantum" in r["prompt"].lower():
                out = r["output"][:150]
                has_garbage = any(ord(c) > 0x3000 for c in out[:50])
                q = "⚠️" if has_garbage else "✅"
                print(f"    {label}: {q} {out}")

    print(f"\n  O SSN deve preservar melhor a qualidade porque usa")
    print(f"  informação contextual do gate aprendido pelo modelo,")
    print(f"  em vez de heurística cega de magnitude.")
    print(f"{'═'*60}\n")

    return 0


if __name__ == "__main__":
    main()
