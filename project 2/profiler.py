"""
sparse_gemma4/monitor/profiler.py
===================================
Sistema de monitoramento de performance para comparação dense vs sparse.

Métricas coletadas:
  - Memória GPU: allocated, reserved, peak (por layer e global)
  - FLOPs: contagem analítica por operação Linear/Attention/FFN
  - Latência: wall-clock e GPU time via CUDA Events
  - Throughput: tokens/s prefill e decode separados
  - Ativação de esparsidade: % de ativações zero (dinâmica) por layer
  - Energia: estimada via TDP e duty cycle (proxy)

Uso:
  profiler = Gemma4Profiler(model, tokenizer)
  with profiler.measure("dense_baseline"):
      output = model.generate(...)
  report = profiler.report()
  profiler.compare("dense_baseline", "sparse_24")
"""

import gc
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs ANALÍTICOS
# ─────────────────────────────────────────────────────────────────────────────

def flops_linear(in_f: int, out_f: int, batch_tokens: int, sparse_ratio: float = 0.0) -> int:
    """
    FLOPs de uma camada Linear: 2 * batch_tokens * in_f * out_f
    (fator 2: multiplicação + adição por elemento)
    Com esparsidade: multiply-add count reduzido pelo fator (1 - sparse_ratio)
    """
    dense_flops = 2 * batch_tokens * in_f * out_f
    return int(dense_flops * (1.0 - sparse_ratio))


def flops_attention(
    seq_len: int, head_dim: int, num_heads: int, batch: int = 1
) -> dict[str, int]:
    """
    FLOPs de um bloco de atenção (QKV proj + scores + output proj).
    Retorna breakdown por componente.
    """
    hidden = head_dim * num_heads
    qkv_proj  = 3 * flops_linear(hidden, hidden, batch * seq_len)
    attn_score = 2 * batch * num_heads * seq_len * seq_len * head_dim  # QK^T
    attn_softmax = batch * num_heads * seq_len * seq_len * 5            # exp, sum, div
    attn_value = 2 * batch * num_heads * seq_len * seq_len * head_dim   # score * V
    out_proj   = flops_linear(hidden, hidden, batch * seq_len)
    return {
        "qkv_proj": qkv_proj,
        "attn_score": attn_score,
        "attn_softmax": attn_softmax,
        "attn_value": attn_value,
        "out_proj": out_proj,
        "total": qkv_proj + attn_score + attn_softmax + attn_value + out_proj
    }


def flops_ffn(hidden: int, intermediate: int, batch_tokens: int) -> dict[str, int]:
    """FLOPs do FFN (SwiGLU / GEGLUTanh: gate_proj + up_proj + element-wise + down_proj)."""
    gate  = flops_linear(hidden, intermediate, batch_tokens)
    up    = flops_linear(hidden, intermediate, batch_tokens)
    elem  = batch_tokens * intermediate           # Hadamard product (gate * up)
    down  = flops_linear(intermediate, hidden, batch_tokens)
    return {
        "gate_proj": gate,
        "up_proj": up,
        "elementwise": elem,
        "down_proj": down,
        "total": gate + up + elem + down
    }


def estimate_gemma4_flops(
    seq_len: int,
    num_text_layers: int = 42,
    hidden: int = 2560,
    intermediate: int = 10240,
    num_heads_local: int = 16,      # q heads para local attn
    num_heads_global: int = 32,     # q heads para global attn
    kv_heads_local: int = 4,
    kv_heads_global: int = 8,
    batch: int = 1,
) -> dict[str, Any]:
    """
    Estima FLOPs totais para um forward pass do text decoder do Gemma 4.
    Separa layers locais (local window attn) e globais (full attn).
    """
    # Gemma 4 padrão: 5 local : 1 global → ~35 local, ~7 global em 42 layers
    n_global = num_text_layers // 6
    n_local  = num_text_layers - n_global

    batch_tokens = batch * seq_len

    local_attn  = flops_attention(seq_len, hidden // num_heads_local, num_heads_local, batch)
    global_attn = flops_attention(seq_len, hidden // num_heads_global, num_heads_global, batch)
    ffn         = flops_ffn(hidden, intermediate, batch_tokens)

    per_layer_local  = local_attn["total"] + ffn["total"]
    per_layer_global = global_attn["total"] + ffn["total"]
    total_flops = n_local * per_layer_local + n_global * per_layer_global

    return {
        "total_gflops": total_flops / 1e9,
        "per_layer_local_gflops": per_layer_local / 1e9,
        "per_layer_global_gflops": per_layer_global / 1e9,
        "ffn_gflops": (num_text_layers * ffn["total"]) / 1e9,
        "attn_gflops": (n_local * local_attn["total"] + n_global * global_attn["total"]) / 1e9,
        "ffn_pct": ffn["total"] / (per_layer_local) * 100,
        "seq_len": seq_len,
        "num_layers": num_text_layers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MONITORAMENTO DE MEMÓRIA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    free_mb: float
    total_mb: float
    timestamp: float = field(default_factory=time.perf_counter)

    @classmethod
    def capture(cls, device: int = 0) -> "MemorySnapshot":
        if not torch.cuda.is_available():
            return cls(0, 0, 0, 0, 0, 0)
        
        stats = torch.cuda.memory_stats(device)
        mem_info = torch.cuda.mem_get_info(device)
        total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        
        return cls(
            allocated_mb=stats.get("allocated_bytes.all.current", 0) / (1024**2),
            reserved_mb=stats.get("reserved_bytes.all.current", 0) / (1024**2),
            peak_allocated_mb=stats.get("allocated_bytes.all.peak", 0) / (1024**2),
            peak_reserved_mb=stats.get("reserved_bytes.all.peak", 0) / (1024**2),
            free_mb=mem_info[0] / (1024**2),
            total_mb=total,
        )

    def diff(self, other: "MemorySnapshot") -> dict[str, float]:
        return {
            "delta_allocated_mb": self.allocated_mb - other.allocated_mb,
            "delta_reserved_mb": self.reserved_mb - other.reserved_mb,
            "peak_allocated_mb": self.peak_allocated_mb,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LAYER ACTIVATION SPARSITY TRACKER (dinâmico)
# ─────────────────────────────────────────────────────────────────────────────

class ActivationSparsityTracker:
    """
    Rastreia a esparsidade dinâmica das ativações (não dos pesos) durante inferência.
    Útil para entender onde a rede naturalmente produz zeros (candidates para ReLU-based pruning).
    """

    def __init__(self):
        self.stats: dict[str, dict] = defaultdict(lambda: {
            "calls": 0, "zero_fraction_sum": 0.0, "shape": None
        })
        self._hooks: list[RemovableHandle] = []

    def attach(self, model: nn.Module, target_types=(nn.Linear,)) -> None:
        """Registra hooks em todas as layers alvo."""
        for name, module in model.named_modules():
            if isinstance(module, target_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                zero_frac = (output == 0).float().mean().item()
                self.stats[name]["calls"] += 1
                self.stats[name]["zero_fraction_sum"] += zero_frac
                self.stats[name]["shape"] = tuple(output.shape)
        return hook

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self) -> dict[str, float]:
        return {
            name: s["zero_fraction_sum"] / max(s["calls"], 1)
            for name, s in self.stats.items()
        }

    def top_sparse_layers(self, n: int = 10) -> list[tuple[str, float]]:
        summary = self.summary()
        return sorted(summary.items(), key=lambda x: x[1], reverse=True)[:n]


# ─────────────────────────────────────────────────────────────────────────────
# PROFILER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    label: str
    # Timing
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    total_time_s: float = 0.0
    gpu_time_ms: float = 0.0
    # Throughput
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_tps: float = 0.0     # tokens/s durante prefill
    decode_tps: float = 0.0      # tokens/s durante decode
    # Memória
    memory_before: Optional[MemorySnapshot] = None
    memory_after: Optional[MemorySnapshot] = None
    peak_memory_mb: float = 0.0
    model_memory_mb: float = 0.0
    # FLOPs
    estimated_flops_gflops: float = 0.0
    measured_tflops: float = 0.0  # FLOPs/s efetivos
    # Ativação
    activation_sparsity: dict[str, float] = field(default_factory=dict)
    avg_activation_sparsity: float = 0.0
    # Meta
    device_name: str = ""
    torch_version: str = ""
    extra: dict = field(default_factory=dict)

    def tflops_effective(self) -> float:
        if self.total_time_s > 0:
            return self.estimated_flops_gflops / self.total_time_s / 1000
        return 0.0


class Gemma4Profiler:
    """
    Profiler completo para Gemma 4 — coleta todas as métricas de performance.
    
    Exemplo de uso:
    
        profiler = Gemma4Profiler(model, tokenizer, device=0)
        
        # Rodar baseline denso
        profiler.begin_run("dense")
        output = model.generate(**inputs, max_new_tokens=100)
        metrics = profiler.end_run("dense", inputs, output)
        
        # Rodar esparso
        profiler.begin_run("sparse_68")
        output = sparse_model.generate(**inputs, max_new_tokens=100)
        metrics = profiler.end_run("sparse_68", inputs, output)
        
        # Comparar
        profiler.compare("dense", "sparse_68")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        device: int = 0,
        track_activation_sparsity: bool = True,
        num_warmup_tokens: int = 10,  # Tokens de warmup antes de medir
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.track_activation_sparsity = track_activation_sparsity
        self.num_warmup = num_warmup_tokens
        self._runs: dict[str, RunMetrics] = {}
        self._current_label: Optional[str] = None
        self._act_tracker = ActivationSparsityTracker() if track_activation_sparsity else None
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
        self._t0: float = 0.0
        self._mem_before: Optional[MemorySnapshot] = None

    def begin_run(self, label: str) -> None:
        """Inicia uma sessão de medição."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            gc.collect()

        self._current_label = label
        self._mem_before = MemorySnapshot.capture(self.device)
        
        if self.track_activation_sparsity and self._act_tracker:
            self._act_tracker.stats.clear()
            self._act_tracker.attach(self.model)

        if torch.cuda.is_available():
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event   = torch.cuda.Event(enable_timing=True)
            self._start_event.record()

        self._t0 = time.perf_counter()

    def end_run(
        self,
        label: str,
        inputs: dict,
        outputs,
        seq_len: Optional[int] = None,
    ) -> RunMetrics:
        """
        Finaliza medição e retorna RunMetrics completo.
        
        Args:
            label: identificador da run
            inputs: dict de inputs (com 'input_ids' para contar tokens)
            outputs: saída do model.generate()
            seq_len: sequência de prompt (auto-detectada se None)
        """
        wall_time = time.perf_counter() - self._t0
        
        if torch.cuda.is_available():
            self._end_event.record()
            torch.cuda.synchronize(self.device)
            gpu_time_ms = self._start_event.elapsed_time(self._end_event)
        else:
            gpu_time_ms = wall_time * 1000

        mem_after = MemorySnapshot.capture(self.device)

        if self.track_activation_sparsity and self._act_tracker:
            self._act_tracker.detach()
            act_sparsity = self._act_tracker.summary()
        else:
            act_sparsity = {}

        # Contagem de tokens
        prompt_tokens = inputs.get("input_ids", torch.tensor([])).shape[-1] if "input_ids" in inputs else 0
        if seq_len:
            prompt_tokens = seq_len
        
        if hasattr(outputs, "sequences"):
            total_tokens = outputs.sequences.shape[-1]
        elif isinstance(outputs, torch.Tensor):
            total_tokens = outputs.shape[-1]
        else:
            total_tokens = prompt_tokens

        generated_tokens = max(0, total_tokens - prompt_tokens)
        
        # Estimativa de FLOPs
        flops_info = estimate_gemma4_flops(seq_len=prompt_tokens)
        
        # Memória do modelo
        model_mem = sum(
            p.nelement() * p.element_size()
            for p in self.model.parameters()
        ) / (1024**2)

        # Throughput
        prefill_tps = prompt_tokens / wall_time if wall_time > 0 else 0
        decode_tps  = generated_tokens / wall_time if wall_time > 0 else 0

        metrics = RunMetrics(
            label=label,
            total_time_s=wall_time,
            gpu_time_ms=gpu_time_ms,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            memory_before=self._mem_before,
            memory_after=mem_after,
            peak_memory_mb=mem_after.peak_allocated_mb,
            model_memory_mb=model_mem,
            estimated_flops_gflops=flops_info["total_gflops"],
            activation_sparsity=act_sparsity,
            avg_activation_sparsity=(
                sum(act_sparsity.values()) / len(act_sparsity) if act_sparsity else 0.0
            ),
            device_name=(
                torch.cuda.get_device_name(self.device)
                if torch.cuda.is_available() else "cpu"
            ),
            torch_version=torch.__version__,
        )
        
        self._runs[label] = metrics
        self._current_label = None
        return metrics

    def compare(self, baseline: str, target: str) -> str:
        """
        Gera relatório comparativo entre dois runs.
        Retorna string formatada para print.
        """
        if baseline not in self._runs or target not in self._runs:
            return f"Erro: runs '{baseline}' ou '{target}' não encontradas."

        b = self._runs[baseline]
        t = self._runs[target]

        def pct_change(a, b):
            if a == 0:
                return 0.0
            return (b - a) / a * 100

        def fmt(val, unit="", precision=2):
            return f"{val:.{precision}f}{unit}"

        lines = [
            f"\n{'═'*65}",
            f"  COMPARATIVO DE PERFORMANCE: {baseline} vs {target}",
            f"{'═'*65}",
            f"  {'Métrica':<35} {'Baseline':>10} {'Sparse':>10} {'Δ':>10}",
            f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10}",
        ]

        def row(name, bval, tval, unit="", is_lower_better=False):
            delta = pct_change(bval, tval)
            arrow = "↑" if delta > 0 else "↓"
            is_improvement = (delta > 0) != is_lower_better
            sign = "✓" if is_improvement else "✗"
            lines.append(
                f"  {sign} {name:<33} {fmt(bval):>10}{unit} {fmt(tval):>10}{unit} "
                f"{arrow}{abs(delta):.1f}%"
            )

        row("Tokens/s (prefill)",   b.prefill_tps,       t.prefill_tps,       " tok/s")
        row("Tokens/s (decode)",    b.decode_tps,        t.decode_tps,        " tok/s")
        row("Latência total",       b.total_time_s,      t.total_time_s,      "s",   is_lower_better=True)
        row("GPU time",             b.gpu_time_ms,       t.gpu_time_ms,       "ms",  is_lower_better=True)
        row("Memória peak",         b.peak_memory_mb,    t.peak_memory_mb,    "MB",  is_lower_better=True)
        row("Memória modelo",       b.model_memory_mb,   t.model_memory_mb,   "MB",  is_lower_better=True)
        row("FLOPs estimados",      b.estimated_flops_gflops, t.estimated_flops_gflops, "G")
        row("Ativação esparsidade", b.avg_activation_sparsity*100, t.avg_activation_sparsity*100, "%")

        # Top layers mais esparsas no target
        if t.activation_sparsity:
            top5 = sorted(t.activation_sparsity.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append(f"\n  {'─'*65}")
            lines.append(f"  Top 5 layers com maior esparsidade de ativação ({target}):")
            for layer_name, sparsity in top5:
                lines.append(f"    {layer_name[:55]:55s} {sparsity:.1%}")

        lines.append(f"\n  Device: {b.device_name} | PyTorch {b.torch_version}")
        lines.append(f"{'═'*65}\n")
        return "\n".join(lines)

    def to_json(self, path: str) -> None:
        """Serializa todos os runs para JSON."""
        data = {}
        for label, m in self._runs.items():
            data[label] = {
                "label": m.label,
                "prefill_tps": m.prefill_tps,
                "decode_tps": m.decode_tps,
                "total_time_s": m.total_time_s,
                "gpu_time_ms": m.gpu_time_ms,
                "peak_memory_mb": m.peak_memory_mb,
                "model_memory_mb": m.model_memory_mb,
                "estimated_flops_gflops": m.estimated_flops_gflops,
                "avg_activation_sparsity": m.avg_activation_sparsity,
                "prompt_tokens": m.prompt_tokens,
                "generated_tokens": m.generated_tokens,
                "device": m.device_name,
                "top_sparse_layers": sorted(
                    m.activation_sparsity.items(), key=lambda x: x[1], reverse=True
                )[:20],
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Métricas salvas em: {path}")

    def print_summary(self, label: str) -> None:
        if label not in self._runs:
            print(f"Run '{label}' não encontrada.")
            return
        m = self._runs[label]
        print(f"\n{'─'*50}")
        print(f"  Run: {label}")
        print(f"  Device: {m.device_name}")
        print(f"  Tokens gerados: {m.generated_tokens} ({m.decode_tps:.1f} tok/s)")
        print(f"  Tokens de prompt: {m.prompt_tokens} ({m.prefill_tps:.1f} tok/s)")
        print(f"  Latência total: {m.total_time_s:.3f}s | GPU: {m.gpu_time_ms:.1f}ms")
        print(f"  Memória peak: {m.peak_memory_mb:.1f}MB | Modelo: {m.model_memory_mb:.1f}MB")
        print(f"  FLOPs estimados: {m.estimated_flops_gflops:.1f} GFLOPs")
        print(f"  Esparsidade ativação: {m.avg_activation_sparsity:.1%} média")
        print(f"{'─'*50}\n")
