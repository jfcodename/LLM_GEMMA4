"""
sparse_gemma4/utils/layer_analysis.py
=======================================
Utilitários para análise dimensional e de compatibilidade das layers.
Usado antes de aplicar esparsidade para detectar problemas.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class LayerInfo:
    name: str
    module_type: str
    in_features: Optional[int]
    out_features: Optional[int]
    num_params: int
    is_clippable: bool          # Gemma4ClippableLinear wrapper
    is_conv: bool               # Convoluções — não esparsificáveis via Sparse TC
    dim_ok_24: bool             # Compatível com 2:4
    dim_ok_68: bool             # Compatível com 6:8
    bottleneck: bool            # min(in, out) < 512 — risco alto
    notes: list[str] = field(default_factory=list)


def analyze_all_layers(model: nn.Module) -> list[LayerInfo]:
    """
    Varre o modelo e classifica cada layer Linear/Conv quanto à elegibilidade.
    Retorna lista ordenada por número de parâmetros (maior primeiro).
    """
    results = []

    for name, module in model.named_modules():
        # ── Linear direta ─────────────────────────────────────────────────────
        if isinstance(module, nn.Linear):
            in_f  = module.in_features
            out_f = module.out_features
            is_clip = _is_inside_clippable(model, name)
            notes = []

            ok_24 = (in_f % 4 == 0) and (out_f % 8 == 0) and (in_f >= 64) and (out_f >= 64)
            ok_68 = (in_f % 8 == 0) and (out_f >= 64)
            bottleneck = min(in_f, out_f) < 512

            if not ok_24:
                notes.append(f"2:4 inelegível: in={in_f}%4={in_f%4}, out={out_f}%8={out_f%8}")
            if bottleneck:
                notes.append(f"⚠ Bottleneck: min_dim={min(in_f,out_f)} < 512")
            if "embed_tokens" in name or "lm_head" in name:
                notes.append("Embedding/LM head — skip recomendado")

            results.append(LayerInfo(
                name=name,
                module_type="Linear (ClippableLinear)" if is_clip else "Linear",
                in_features=in_f,
                out_features=out_f,
                num_params=module.weight.numel(),
                is_clippable=is_clip,
                is_conv=False,
                dim_ok_24=ok_24,
                dim_ok_68=ok_68,
                bottleneck=bottleneck,
                notes=notes,
            ))

        # ── Convoluções ────────────────────────────────────────────────────────
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            results.append(LayerInfo(
                name=name,
                module_type=f"Conv{module.weight.dim()-2}d (groups={module.groups})",
                in_features=None,
                out_features=None,
                num_params=module.weight.numel(),
                is_clippable=False,
                is_conv=True,
                dim_ok_24=False,
                dim_ok_68=False,
                bottleneck=False,
                notes=["Conv — excluída de Sparse TC. Não aplicar esparsidade."],
            ))

    # Ordena por tamanho (maior primeiro)
    results.sort(key=lambda x: x.num_params, reverse=True)
    return results


def _is_inside_clippable(model: nn.Module, linear_name: str) -> bool:
    """Verifica se o Linear está dentro de um Gemma4ClippableLinear."""
    parent_name = ".".join(linear_name.split(".")[:-1])
    for name, mod in model.named_modules():
        if name == parent_name:
            return type(mod).__name__ == "Gemma4ClippableLinear"
    return False


def print_layer_table(layers: list[LayerInfo], top_n: int = 40) -> None:
    """Imprime tabela formatada das layers para inspeção rápida."""
    print(f"\n{'═'*100}")
    print(f"  ANÁLISE DE LAYERS — TOP {top_n} POR PARÂMETROS")
    print(f"{'═'*100}")
    print(
        f"  {'#':>3}  {'Nome':<55} {'Tipo':<22} {'in×out':>15} "
        f"{'Params':>10} {'2:4':>5} {'6:8':>5} {'BN':>4}"
    )
    print(f"  {'─'*3}  {'─'*55} {'─'*22} {'─'*15} {'─'*10} {'─'*5} {'─'*5} {'─'*4}")

    for i, l in enumerate(layers[:top_n]):
        dim_str = f"{l.in_features}×{l.out_features}" if l.in_features else "conv"
        ok24 = "✓" if l.dim_ok_24 else ("✗" if not l.is_conv else "—")
        ok68 = "✓" if l.dim_ok_68 else ("✗" if not l.is_conv else "—")
        bn   = "⚠" if l.bottleneck else ""
        print(
            f"  {i+1:>3}. {l.name:<55} {l.module_type:<22} {dim_str:>15} "
            f"{l.num_params/1e6:>9.2f}M {ok24:>5} {ok68:>5} {bn:>4}"
        )
        for note in l.notes:
            print(f"       ↳ {note}")

    total = sum(l.num_params for l in layers)
    eligible_24 = sum(l.num_params for l in layers if l.dim_ok_24)
    eligible_68 = sum(l.num_params for l in layers if l.dim_ok_68)
    print(f"\n  Total params nas layers analisadas: {total/1e6:.1f}M")
    print(f"  Elegíveis para 2:4: {eligible_24/1e6:.1f}M ({eligible_24/total*100:.1f}%)")
    print(f"  Elegíveis para 6:8: {eligible_68/1e6:.1f}M ({eligible_68/total*100:.1f}%)")
    print(f"{'═'*100}\n")


def count_sparse_params(model: nn.Module) -> dict[str, int]:
    """
    Conta parâmetros que já estão em formato esparso (SparseSemiStructuredTensor).
    Útil para verificar se a esparsidade foi aplicada corretamente.
    """
    try:
        from torch.sparse import SparseSemiStructuredTensor
        HAVE_SPARSE = True
    except ImportError:
        HAVE_SPARSE = False

    counts = {"dense": 0, "sparse_24": 0, "other": 0}
    for name, param in model.named_parameters():
        if HAVE_SPARSE and isinstance(param.data, SparseSemiStructuredTensor):
            counts["sparse_24"] += param.numel()
        elif param.is_sparse:
            counts["other"] += param.numel()
        else:
            # Conta zeros reais (esparsidade de peso, mesmo sem formato comprimido)
            counts["dense"] += param.numel()
    return counts


def compute_actual_weight_sparsity(model: nn.Module) -> dict[str, float]:
    """
    Computa a fração real de zeros nos pesos de cada layer Linear.
    Útil para validar que a máscara foi aplicada corretamente.
    """
    result = {}
    for name, module in model.named_modules():
        weight = None
        if isinstance(module, nn.Linear):
            weight = module.weight
        elif hasattr(module, "linear") and isinstance(getattr(module, "linear", None), nn.Linear):
            weight = module.linear.weight

        if weight is not None and isinstance(weight, torch.Tensor):
            zeros = (weight == 0).float().mean().item()
            result[name] = zeros
    return result


def sparsity_histogram(model: nn.Module, bins: int = 10) -> None:
    """
    Imprime histograma de esparsidade de pesos por layer.
    Ajuda a identificar layers naturalmente esparsas (candidatos a maiores taxas).
    """
    sparsities = compute_actual_weight_sparsity(model)
    if not sparsities:
        print("Nenhuma layer Linear encontrada.")
        return

    values = list(sparsities.values())
    min_s, max_s = min(values), max(values)
    step = (max_s - min_s) / bins if max_s > min_s else 0.1

    print("\n  HISTOGRAMA DE ESPARSIDADE DE PESOS:")
    print(f"  min={min_s:.1%}  max={max_s:.1%}  média={sum(values)/len(values):.1%}")
    print()

    for b in range(bins):
        lo = min_s + b * step
        hi = lo + step
        count = sum(1 for v in values if lo <= v < hi)
        bar   = "█" * count
        print(f"  [{lo:.0%}–{hi:.0%}]  {bar:<40} ({count})")
    print()
