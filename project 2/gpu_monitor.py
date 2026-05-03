"""
sparse_gemma4/monitor/gpu_monitor.py
======================================
Monitoramento contínuo de GPU em thread separada:
  - Potência (Watts) — via NVML
  - Temperatura (°C)
  - Utilização SM (%)
  - Memória alocada / total
  - Bandwidth de memória estimado

Uso:
  with GpuMonitor(device=0) as mon:
      model.generate(...)
  stats = mon.get_stats()
  print(f"Potência média: {stats['power_avg_w']:.1f}W")
  print(f"Energia total:  {stats['energy_j']:.2f}J")
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

import torch


@dataclass
class GpuSample:
    timestamp: float
    power_w: float           # Potência instantânea (W)
    temp_c: float            # Temperatura (°C)
    sm_util_pct: float       # Utilização dos SMs (%)
    mem_util_pct: float      # Utilização de memória (%)
    mem_used_mb: float       # Memória usada (MB)
    mem_total_mb: float      # Memória total (MB)
    clock_sm_mhz: int        # Clock SM atual (MHz)
    clock_mem_mhz: int       # Clock memória atual (MHz)


@dataclass
class GpuStats:
    device_name: str
    device_idx: int
    samples: list[GpuSample] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def power_avg_w(self) -> float:
        return _mean([s.power_w for s in self.samples])

    @property
    def power_peak_w(self) -> float:
        return max((s.power_w for s in self.samples), default=0.0)

    @property
    def energy_j(self) -> float:
        """Energia total estimada (Joules) = integral de P(t) dt via trapézio."""
        if len(self.samples) < 2:
            return self.power_avg_w * self.duration_s
        energy = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i].timestamp - self.samples[i-1].timestamp
            avg_p = (self.samples[i].power_w + self.samples[i-1].power_w) / 2
            energy += avg_p * dt
        return energy

    @property
    def sm_util_avg(self) -> float:
        return _mean([s.sm_util_pct for s in self.samples])

    @property
    def temp_avg_c(self) -> float:
        return _mean([s.temp_c for s in self.samples])

    @property
    def mem_peak_mb(self) -> float:
        return max((s.mem_used_mb for s in self.samples), default=0.0)

    def tokens_per_joule(self, num_tokens: int) -> float:
        if self.energy_j > 0:
            return num_tokens / self.energy_j
        return 0.0

    def summary(self, num_tokens: int = 0) -> str:
        lines = [
            f"\n  GPU Monitor — {self.device_name} (idx={self.device_idx})",
            f"  {'─'*45}",
            f"  Duração          : {self.duration_s:.3f}s",
            f"  Potência média   : {self.power_avg_w:.1f}W  (peak: {self.power_peak_w:.1f}W)",
            f"  Energia total    : {self.energy_j:.3f}J",
            f"  Temperatura média: {self.temp_avg_c:.1f}°C",
            f"  Util. SM média   : {self.sm_util_avg:.1f}%",
            f"  Memória peak     : {self.mem_peak_mb:.0f}MB",
        ]
        if num_tokens > 0:
            lines.append(f"  Eficiência       : {self.tokens_per_joule(num_tokens):.2f} tokens/J")
        lines.append(f"  {'─'*45}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "device_name": self.device_name,
            "device_idx": self.device_idx,
            "duration_s": round(self.duration_s, 4),
            "power_avg_w": round(self.power_avg_w, 2),
            "power_peak_w": round(self.power_peak_w, 2),
            "energy_j": round(self.energy_j, 4),
            "sm_util_avg_pct": round(self.sm_util_avg, 2),
            "temp_avg_c": round(self.temp_avg_c, 1),
            "mem_peak_mb": round(self.mem_peak_mb, 1),
        }


def _mean(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


class GpuMonitor:
    """
    Monitor de GPU em background thread. Usa como context manager.

    with GpuMonitor(device=0, interval_s=0.05) as mon:
        model.generate(...)
    stats = mon.get_stats()
    print(stats.summary(num_tokens=200))
    """

    def __init__(self, device: int = 0, interval_s: float = 0.05):
        self.device_idx = device
        self.interval_s = interval_s
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: list[GpuSample] = []
        self._start_t: float = 0.0
        self._end_t: float = 0.0
        self._handle = None
        self._device_name = "unknown"

        if NVML_AVAILABLE:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                self._device_name = pynvml.nvmlDeviceGetName(self._handle)
                if isinstance(self._device_name, bytes):
                    self._device_name = self._device_name.decode()
            except Exception as e:
                print(f"[GpuMonitor] NVML init warning: {e}")
        elif torch.cuda.is_available():
            self._device_name = torch.cuda.get_device_name(device)

    def __enter__(self) -> "GpuMonitor":
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def start(self) -> None:
        self._stop_event.clear()
        self._samples.clear()
        self._start_t = time.perf_counter()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._end_t = time.perf_counter()

    def _collect_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample()
            if sample:
                self._samples.append(sample)
            time.sleep(self.interval_s)

    def _sample(self) -> Optional[GpuSample]:
        t = time.perf_counter()

        if NVML_AVAILABLE and self._handle:
            try:
                power_mw  = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                temp_c    = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
                util      = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem_info  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                clk_sm    = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
                clk_mem   = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM)

                return GpuSample(
                    timestamp=t,
                    power_w=power_mw / 1000.0,
                    temp_c=float(temp_c),
                    sm_util_pct=float(util.gpu),
                    mem_util_pct=float(util.memory),
                    mem_used_mb=mem_info.used / (1024**2),
                    mem_total_mb=mem_info.total / (1024**2),
                    clock_sm_mhz=int(clk_sm),
                    clock_mem_mhz=int(clk_mem),
                )
            except Exception:
                pass  # Falha silenciosa — NVML pode ter limitações de poll

        # Fallback: apenas memória via PyTorch
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(self.device_idx) / (1024**2)
            mem_total = torch.cuda.get_device_properties(self.device_idx).total_memory / (1024**2)
            return GpuSample(
                timestamp=t,
                power_w=0.0,     # Indisponível sem NVML
                temp_c=0.0,
                sm_util_pct=0.0,
                mem_util_pct=0.0,
                mem_used_mb=mem_alloc,
                mem_total_mb=mem_total,
                clock_sm_mhz=0,
                clock_mem_mhz=0,
            )

        return None

    def get_stats(self) -> GpuStats:
        stats = GpuStats(
            device_name=self._device_name,
            device_idx=self.device_idx,
            samples=self._samples.copy(),
            duration_s=self._end_t - self._start_t,
        )
        return stats
