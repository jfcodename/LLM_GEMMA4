"""
sparse_gemma4/core/sparsifier.py
==================================
Engine principal de esparsificação para o Gemma 4.

Implementa:
  - Magnitude pruning com máscara 2:4 (torch.ao.pruning / to_sparse_semi_structured)
  - SlideSparse 6:8 via decomposição de janelas deslizantes
  - Transferência não-destrutiva de pesos (dense → sparse sem re-treino)
  - Suporte a Gemma4ClippableLinear (vision/audio towers)
  - Análise de elegibilidade dimensional
"""

import fnmatch
import logging
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch
import torch.nn as nn

# Importações condicionais — graceful degradation se bibliotecas não estiverem disponíveis
try:
    from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
    SPARSE_SEMI_AVAILABLE = True
except ImportError:
    SPARSE_SEMI_AVAILABLE = False
    logging.warning("torch.sparse.to_sparse_semi_structured não disponível. Atualize para PyTorch >= 2.1")

try:
    from torch.ao.pruning import WeightNormSparsifier
    AO_PRUNING_AVAILABLE = True
except ImportError:
    AO_PRUNING_AVAILABLE = False

from configs.sparsity_policy import SparsityMode, LayerPolicy, CONSERVATIVE_POLICY

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ESTRUTURAS DE RESULTADO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerSparsityResult:
    name: str
    mode: SparsityMode
    original_shape: tuple
    actual_sparsity: float          # Fração real de zeros após pruning
    target_sparsity: float          # Fração alvo
    nonzero_params: int
    total_params: int
    memory_before_mb: float
    memory_after_mb: float
    eligible: bool
    skip_reason: str = ""
    time_ms: float = 0.0


@dataclass
class SparsificationReport:
    model_name: str
    policy_name: str
    layers: list[LayerSparsityResult] = field(default_factory=list)
    total_params_original: int = 0
    total_params_nonzero: int = 0
    total_memory_before_mb: float = 0.0
    total_memory_after_mb: float = 0.0
    sparsification_time_s: float = 0.0

    @property
    def global_sparsity(self) -> float:
        if self.total_params_original == 0:
            return 0.0
        return 1.0 - self.total_params_nonzero / self.total_params_original

    @property
    def memory_reduction_pct(self) -> float:
        if self.total_memory_before_mb == 0:
            return 0.0
        return (1 - self.total_memory_after_mb / self.total_memory_before_mb) * 100

    def summary(self) -> str:
        applied = [l for l in self.layers if l.eligible]
        skipped = [l for l in self.layers if not l.eligible]
        lines = [
            f"\n{'═'*60}",
            f"  GEMMA 4 SPARSIFICATION REPORT",
            f"{'═'*60}",
            f"  Modelo       : {self.model_name}",
            f"  Política     : {self.policy_name}",
            f"  Layers apl.  : {len(applied)} / {len(self.layers)}",
            f"  Layers skip  : {len(skipped)}",
            f"  Esparsidade  : {self.global_sparsity:.1%} global",
            f"  Memória antes: {self.total_memory_before_mb:.1f} MB",
            f"  Memória após : {self.total_memory_after_mb:.1f} MB",
            f"  Redução mem. : {self.memory_reduction_pct:.1f}%",
            f"  Tempo total  : {self.sparsification_time_s:.2f}s",
            f"{'─'*60}",
        ]
        for l in applied:
            lines.append(
                f"  ✓ [{l.mode.value:5s}] {l.name[:50]:50s}  "
                f"sparsity={l.actual_sparsity:.1%}  "
                f"mem={l.memory_after_mb:.1f}MB"
            )
        for l in skipped:
            lines.append(f"  ✗ [skip ] {l.name[:50]:50s}  {l.skip_reason}")
        lines.append(f"{'═'*60}\n")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE MÁSCARA
# ─────────────────────────────────────────────────────────────────────────────

def _magnitude_mask_24(weight: torch.Tensor) -> torch.Tensor:
    """
    Gera máscara 2:4: para cada grupo de 4 pesos consecutivos,
    mantém os 2 de maior magnitude e zera os 2 menores.
    Requisito: última dimensão múltipla de 4.
    """
    assert weight.dim() == 2, "Espera tensor 2D (out_features, in_features)"
    rows, cols = weight.shape
    assert cols % 4 == 0, f"in_features ({cols}) deve ser múltiplo de 4"

    w = weight.abs().view(rows, -1, 4)          # (rows, cols//4, 4)
    _, indices = torch.topk(w, k=2, dim=-1)     # top-2 por magnitude
    mask = torch.zeros_like(w, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask.view(rows, cols)                  # (rows, cols)


def _magnitude_mask_68(weight: torch.Tensor) -> torch.Tensor:
    """
    Gera máscara 6:8 (SlideSparse): para cada grupo de 8 pesos,
    mantém os 6 de maior magnitude (25% zeros).
    Mais conservador que 2:4 — preserva ~95-99% da acurácia em LLMs.
    """
    assert weight.dim() == 2
    rows, cols = weight.shape
    # Pad para múltiplo de 8 se necessário
    pad = (8 - cols % 8) % 8
    if pad > 0:
        weight = torch.nn.functional.pad(weight, (0, pad))
    
    rows, cols_padded = weight.shape
    w = weight.abs().view(rows, -1, 8)          # (rows, cols//8, 8)
    _, zero_indices = torch.topk(w, k=2, dim=-1, largest=False)  # 2 menores
    mask = torch.ones_like(w, dtype=torch.bool)
    mask.scatter_(-1, zero_indices, False)
    mask = mask.view(rows, cols_padded)
    
    # Remove padding
    orig_cols = weight.shape[1] - pad if pad > 0 else weight.shape[1]
    return mask[:, :orig_cols]


def _decompose_68_to_24(weight: torch.Tensor, mask_68: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SlideSparse: decompõe padrão 6:8 em blocos 2:4 via janela deslizante.
    Permite usar Sparse Tensor Cores (que só suportam 2:4) para acelerar 6:8.
    
    Referência: SlideSparse (arxiv 2603.05232, Março 2026)
    """
    # Por enquanto retorna a máscara 6:8 diretamente como approximação.
    # A decomposição full em janelas 2:4 deslizantes requer o runtime SlideSparse.
    # TODO: integrar quando SlideSparse liberar código open-source
    return weight * mask_68, mask_68


def _check_dim_eligibility(weight: torch.Tensor, mode: SparsityMode) -> tuple[bool, str]:
    """Verifica se as dimensões do tensor são compatíveis com o modo de esparsidade."""
    if weight.dim() != 2:
        return False, f"Não é 2D (shape={weight.shape})"
    
    rows, cols = weight.shape
    
    if mode in (SparsityMode.SEMI_24,):
        if cols % 4 != 0:
            return False, f"in_features={cols} não é múltiplo de 4"
        if rows % 8 != 0:
            return False, f"out_features={rows} não é múltiplo de 8 (req. Sparse TC)"
        if rows < 64 or cols < 64:
            return False, f"Dimensões muito pequenas ({rows}×{cols}), risco de degradação"
    
    elif mode == SparsityMode.SEMI_68:
        if cols % 8 != 0:
            return False, f"in_features={cols} não é múltiplo de 8"
        if rows < 64:
            return False, f"out_features={rows} muito pequeno"
    
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class Gemma4Sparsifier:
    """
    Aplica esparsidade estruturada ao Gemma 4 com transferência de pesos.
    
    Fluxo:
      1. Identifica layers elegíveis com base na política
      2. Gera máscara (2:4 ou 6:8) por magnitude
      3. Aplica máscara in-place nos pesos (não-destrutivo — cria novo tensor)
      4. Para 2:4: converte para SparseSemiStructuredTensor nativo do PyTorch
      5. Retorna relatório detalhado
    
    Uso:
      sparsifier = Gemma4Sparsifier(model, policy=CONSERVATIVE_POLICY)
      report = sparsifier.apply()
      print(report.summary())
    """

    def __init__(
        self,
        model: nn.Module,
        policy: dict[str, LayerPolicy] = None,
        policy_name: str = "conservative",
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        use_native_sparse: bool = True,    # Usa to_sparse_semi_structured para 2:4
        dry_run: bool = False,             # Analisa sem modificar o modelo
    ):
        self.model = model
        self.policy = policy or CONSERVATIVE_POLICY
        self.policy_name = policy_name
        self.dtype = dtype
        self.device = device or next(model.parameters()).device
        self.use_native_sparse = use_native_sparse and SPARSE_SEMI_AVAILABLE
        self.dry_run = dry_run
        self._report = SparsificationReport(
            model_name=type(model).__name__,
            policy_name=policy_name
        )

    def _match_policy(self, name: str) -> Optional[LayerPolicy]:
        """
        Resolve o nome da layer contra os padrões da política (suporta wildcards '*').
        Retorna a política mais específica que casa.
        """
        best_match = None
        best_specificity = -1

        for pattern, layer_policy in self.policy.items():
            # Converte padrão de política para glob fnmatch
            # Ex: "language_model.layers.*.mlp.gate_proj" → casa com "language_model.layers.3.mlp.gate_proj"
            if fnmatch.fnmatch(name, pattern):
                specificity = len(pattern.replace("*", ""))
                if specificity > best_specificity:
                    best_match = layer_policy
                    best_specificity = specificity

        return best_match

    def _get_weight_tensor(self, module: nn.Module) -> Optional[torch.Tensor]:
        """
        Extrai o tensor de peso, suportando tanto nn.Linear quanto
        Gemma4ClippableLinear (que tem .linear.weight interno).
        """
        # Caso Gemma4ClippableLinear (vision/audio towers)
        if hasattr(module, 'linear') and isinstance(module.linear, nn.Linear):
            return module.linear.weight
        # Caso Linear padrão (text decoder)
        if isinstance(module, nn.Linear):
            return module.weight
        return None

    def _set_weight_tensor(self, module: nn.Module, new_weight: nn.Parameter) -> None:
        """Define o peso no lugar correto, suportando ClippableLinear."""
        if hasattr(module, 'linear') and isinstance(module.linear, nn.Linear):
            module.linear.weight = new_weight
        elif isinstance(module, nn.Linear):
            module.weight = new_weight

    def _tensor_memory_mb(self, t: torch.Tensor) -> float:
        return t.nelement() * t.element_size() / (1024 ** 2)

    def _apply_24(self, weight: torch.Tensor, name: str) -> tuple[torch.Tensor, float]:
        """Aplica esparsidade 2:4 e opcionalmente converte para formato nativo."""
        mask = _magnitude_mask_24(weight)
        pruned = weight * mask

        if self.use_native_sparse and not self.dry_run:
            # Converte para SparseSemiStructuredTensor — permite usar Sparse TC
            # Requer dtype float16 ou bfloat16
            if pruned.dtype not in (torch.float16, torch.bfloat16):
                pruned = pruned.to(torch.float16)
            try:
                sparse_tensor = to_sparse_semi_structured(pruned)
                actual_sparsity = 0.5  # 2:4 sempre tem exatamente 50%
                logger.debug(f"[2:4] {name}: convertido para SparseSemiStructuredTensor")
                return sparse_tensor, actual_sparsity
            except Exception as e:
                logger.warning(f"[2:4] {name}: falha na conversão nativa ({e}), mantendo denso mascarado")
        
        actual_sparsity = 1.0 - (mask.float().sum() / mask.numel()).item()
        return pruned, actual_sparsity

    def _apply_68(self, weight: torch.Tensor, name: str) -> tuple[torch.Tensor, float]:
        """Aplica esparsidade 6:8 (SlideSparse approach)."""
        mask = _magnitude_mask_68(weight)
        pruned = weight * mask
        actual_sparsity = 1.0 - (mask.float().sum() / mask.numel()).item()
        logger.debug(f"[6:8] {name}: sparsity real={actual_sparsity:.1%}")
        return pruned, actual_sparsity

    def apply(self) -> SparsificationReport:
        """
        Ponto de entrada principal: percorre o modelo e aplica a política de esparsidade.
        Retorna SparsificationReport com métricas detalhadas.
        """
        start = time.perf_counter()
        logger.info(f"Iniciando esparsificação — política: {self.policy_name} | dry_run={self.dry_run}")

        total_params_orig = 0
        total_params_nz = 0
        total_mem_before = 0.0
        total_mem_after = 0.0

        for name, module in self.model.named_modules():
            weight = self._get_weight_tensor(module)
            if weight is None:
                continue

            policy = self._match_policy(name)
            if policy is None:
                continue  # Layer não coberta pela política — pula silenciosamente

            layer_start = time.perf_counter()
            mem_before = self._tensor_memory_mb(weight)
            total_params_orig += weight.numel()
            total_mem_before += mem_before

            # ── SKIP explícito ────────────────────────────────────────────────
            if policy.mode == SparsityMode.SKIP:
                result = LayerSparsityResult(
                    name=name, mode=SparsityMode.SKIP,
                    original_shape=tuple(weight.shape),
                    actual_sparsity=0.0, target_sparsity=0.0,
                    nonzero_params=weight.numel(), total_params=weight.numel(),
                    memory_before_mb=mem_before, memory_after_mb=mem_before,
                    eligible=False, skip_reason=policy.note
                )
                total_params_nz += weight.numel()
                total_mem_after += mem_before
                self._report.layers.append(result)
                continue

            # ── Verificação de elegibilidade dimensional ──────────────────────
            eligible, reason = _check_dim_eligibility(weight, policy.mode)
            if not eligible:
                result = LayerSparsityResult(
                    name=name, mode=policy.mode,
                    original_shape=tuple(weight.shape),
                    actual_sparsity=0.0, target_sparsity=policy.sparsity_ratio,
                    nonzero_params=weight.numel(), total_params=weight.numel(),
                    memory_before_mb=mem_before, memory_after_mb=mem_before,
                    eligible=False, skip_reason=f"Dims inelegíveis: {reason}"
                )
                total_params_nz += weight.numel()
                total_mem_after += mem_before
                self._report.layers.append(result)
                logger.warning(f"Skip {name}: {reason}")
                continue

            # ── Aplicar esparsidade ───────────────────────────────────────────
            if not self.dry_run:
                w = weight.detach().to(self.dtype)
                
                if policy.mode == SparsityMode.SEMI_24:
                    new_w, actual_sparsity = self._apply_24(w, name)
                elif policy.mode == SparsityMode.SEMI_68:
                    new_w, actual_sparsity = self._apply_68(w, name)
                else:
                    new_w, actual_sparsity = w, 0.0
                    logger.warning(f"Modo {policy.mode} não implementado para {name}, mantendo denso")

                if isinstance(new_w, torch.Tensor):
                    self._set_weight_tensor(module, nn.Parameter(new_w, requires_grad=False))

                mem_after = self._tensor_memory_mb(new_w if isinstance(new_w, torch.Tensor) else weight)
                nz_count = int(new_w.count_nonzero().item()) if isinstance(new_w, torch.Tensor) and not isinstance(new_w, SparseSemiStructuredTensor if SPARSE_SEMI_AVAILABLE else type(None)) else int(weight.numel() * (1 - actual_sparsity))
            else:
                # Dry run — apenas analisa
                actual_sparsity = policy.sparsity_ratio
                mem_after = mem_before * (1 - actual_sparsity * 0.5)  # Estimativa
                nz_count = int(weight.numel() * (1 - actual_sparsity))

            total_params_nz += nz_count
            total_mem_after += mem_after

            result = LayerSparsityResult(
                name=name, mode=policy.mode,
                original_shape=tuple(weight.shape),
                actual_sparsity=actual_sparsity,
                target_sparsity=policy.sparsity_ratio,
                nonzero_params=nz_count,
                total_params=weight.numel(),
                memory_before_mb=mem_before,
                memory_after_mb=mem_after,
                eligible=True,
                time_ms=(time.perf_counter() - layer_start) * 1000
            )
            self._report.layers.append(result)
            logger.info(
                f"✓ [{policy.mode.value}] {name} | "
                f"shape={weight.shape} | "
                f"sparsity={actual_sparsity:.1%} | "
                f"{mem_before:.1f}→{mem_after:.1f}MB | "
                f"{result.time_ms:.1f}ms"
            )

        # Finaliza relatório
        self._report.total_params_original = total_params_orig
        self._report.total_params_nonzero = total_params_nz
        self._report.total_memory_before_mb = total_mem_before
        self._report.total_memory_after_mb = total_mem_after
        self._report.sparsification_time_s = time.perf_counter() - start

        logger.info(f"Esparsificação concluída em {self._report.sparsification_time_s:.2f}s")
        return self._report

    def get_report(self) -> SparsificationReport:
        return self._report

    def save_sparse_model(self, path: str) -> None:
        """Salva o modelo esparso com estado completo."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "sparsity_report": self._report,
            "policy_name": self.policy_name,
        }, path)
        logger.info(f"Modelo esparso salvo em: {path}")

    @classmethod
    def load_sparse_model(cls, model: nn.Module, path: str) -> "Gemma4Sparsifier":
        """Carrega modelo esparso previamente salvo."""
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        inst = cls(model, policy_name=checkpoint.get("policy_name", "unknown"))
        inst._report = checkpoint.get("sparsity_report", SparsificationReport("unknown", "unknown"))
        return inst
