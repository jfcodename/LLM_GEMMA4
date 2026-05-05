"""
Gemma 4 Neo — Structural Pruning Engine
=========================================

O insight central (provado pelos experimentos):
  Top-K masking com keep=50% → qualidade perfeita, ZERO speedup.
  Por quê? GPU não pula zeros em matmul denso.
  Solução: remover fisicamente as colunas — W(2560×10240) → W(2560×5120)

Fluxo do MLP original:
  gate_proj(x): (B,T,2560) @ (2560,10240)ᵀ → (B,T,10240)
  up_proj(x):   (B,T,2560) @ (2560,10240)ᵀ → (B,T,10240)
  hidden:       ReLU²(gate) ⊙ up → (B,T,10240)
  down_proj:    (B,T,10240) @ (10240,2560)ᵀ → (B,T,2560)

Fluxo do MLP após structural pruning (keep_ratio=0.50):
  gate_proj(x): (B,T,2560) @ (2560,5120)ᵀ → (B,T,5120)   ← 2× menos FLOPs
  up_proj(x):   (B,T,2560) @ (2560,5120)ᵀ → (B,T,5120)   ← 2× menos FLOPs
  hidden:       ReLU²(gate) ⊙ up → (B,T,5120)
  down_proj:    (B,T,5120) @ (5120,2560)ᵀ → (B,T,2560)   ← 2× menos FLOPs

Total: ~2× menos FLOPs no MLP. MLP é ~70% do compute total → ~1.6× speedup global.

COMO DECIDIR QUAIS NEURÔNIOS MANTER:
  Usa calibração com dados reais para calcular a importância média de cada
  neurônio. Importância = média da magnitude da ativação pós-ReLU² em amostras.
  Os top-k neurônios com maior importância são mantidos permanentemente.
  Estratégia alternativa: usar per_layer_input_gate como proxy (sem dados).
"""

import os
import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO MLP PODADO
# Substitui Gemma4TextMLP — interface idêntica, weights menores
# ─────────────────────────────────────────────────────────────────────────────

class PrunedMLP(nn.Module):
    """
    MLP com dimensão intermediate fisicamente reduzida.

    Pesos carregados são SUBCONJUNTOS dos pesos originais:
      gate_proj.weight: (keep_neurons, hidden_size)    ← linhas selecionadas
      up_proj.weight:   (keep_neurons, hidden_size)    ← linhas selecionadas
      down_proj.weight: (hidden_size, keep_neurons)    ← colunas selecionadas

    Resultado: forward pass faz matmuls menores → speedup real de hardware.
    """

    def __init__(
        self,
        hidden_size: int,
        kept_neurons: int,          # intermediate_size após pruning
        gate_proj_weight: torch.Tensor,   # (kept_neurons, hidden_size)
        up_proj_weight: torch.Tensor,     # (kept_neurons, hidden_size)
        down_proj_weight: torch.Tensor,   # (hidden_size, kept_neurons)
        use_relu2: bool = True,
        neuron_indices: Optional[torch.Tensor] = None,  # índices mantidos (debug)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = kept_neurons
        self.use_relu2 = use_relu2

        # Pesos fisicamente menores — o único segredo do speedup
        self.gate_proj = nn.Linear(hidden_size, kept_neurons, bias=False)
        self.up_proj   = nn.Linear(hidden_size, kept_neurons, bias=False)
        self.down_proj = nn.Linear(kept_neurons, hidden_size, bias=False)

        self.gate_proj.weight.data.copy_(gate_proj_weight)
        self.up_proj.weight.data.copy_(up_proj_weight)
        self.down_proj.weight.data.copy_(down_proj_weight)

        # Registra índices para auditoria/análise (não afeta compute)
        if neuron_indices is not None:
            self.register_buffer("kept_neuron_indices", neuron_indices, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)   # (B, T, kept_neurons)
        if self.use_relu2:
            gate = F.relu(gate).square()   # ReLU² — zeros exatos adicionais
        else:
            gate = F.gelu(gate, approximate='tanh')
        up = self.up_proj(x)       # (B, T, kept_neurons)
        return self.down_proj(gate * up)   # (B, T, hidden_size)

    @property
    def reduction_ratio(self) -> float:
        """Fração do compute original usada."""
        original = self.hidden_size * 10240 * 3  # gate + up + down (aprox)
        pruned = self.hidden_size * self.intermediate_size * 3
        return pruned / original


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRAÇÃO DE IMPORTÂNCIA
# Decide quais neurônios manter usando dados reais
# ─────────────────────────────────────────────────────────────────────────────

class NeuronImportanceCalibrator:
    """
    Mede a importância de cada neurônio MLP usando dados de calibração.

    Importância de um neurônio j na layer i:
        importance[i][j] = média( |ReLU²(gate_proj_i(x))_j| ) sobre amostras

    Neurônios com alta importância média são selecionados para manter.
    Os outros são removidos permanentemente.

    Alternativa leve (sem dados): usa o per_layer_input_gate como proxy.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self._hooks: List = []
        self._importance: Dict[int, torch.Tensor] = {}  # layer_idx → (intermediate,)
        self._n_samples: Dict[int, int] = {}

    def _get_lm_layers(self):
        """Acessa as decoder layers do language model."""
        try:
            return self.model.model.language_model.layers
        except AttributeError:
            try:
                return self.model.language_model.layers
            except AttributeError:
                return self.model.model.layers

    def calibrate_with_data(
        self,
        tokenizer,
        texts: List[str],
        max_length: int = 512,
        batch_size: int = 4,
    ) -> Dict[int, torch.Tensor]:
        """
        Roda forward passes com textos reais e acumula importâncias.

        Retorna dict {layer_idx: tensor(intermediate_size,)} com scores.
        """
        layers = self._get_lm_layers()
        n_layers = len(layers)

        # Inicializa acumuladores
        for i in range(n_layers):
            layer = layers[i]
            inter = layer.mlp.gate_proj.out_features
            self._importance[i] = torch.zeros(inter, device="cpu")
            self._n_samples[i] = 0

        # Hook: captura saída do gate_proj (pré-ativação) por layer
        def make_gate_hook(layer_idx: int):
            def hook(module: nn.Module, inp, out: torch.Tensor):
                # out: (B, T, intermediate) — pré-ativação do gate_proj
                # Importância = magnitude pós-ReLU² (simula ativação real)
                with torch.no_grad():
                    activated = F.relu(out).square()  # ReLU²
                    # Soma magnitude por neurônio (dim 0=batch, 1=seq)
                    importance = activated.float().abs().mean(dim=(0, 1)).cpu()
                    self._importance[layer_idx] += importance
                    self._n_samples[layer_idx] += 1
            return hook

        # Registra hooks em todos os gate_proj
        for i, layer in enumerate(layers):
            h = layer.mlp.gate_proj.register_forward_hook(make_gate_hook(i))
            self._hooks.append(h)

        # Forward passes com os dados de calibração
        self.model.eval()
        total_batches = 0
        print(f"  Calibrando {n_layers} layers com {len(texts)} textos...")

        with torch.no_grad():
            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start:batch_start + batch_size]
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                ).to(self.device)

                self.model(**inputs)
                total_batches += 1

                if total_batches % 5 == 0:
                    print(f"    Batch {total_batches}/{len(texts)//batch_size + 1}...")

        # Remove hooks
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # Normaliza por número de amostras
        for i in range(n_layers):
            if self._n_samples[i] > 0:
                self._importance[i] /= self._n_samples[i]

        print(f"  ✓ Calibração concluída: {total_batches} batches processados")
        return self._importance

    def calibrate_with_gate(self, keep_ratio: float = 0.5) -> Dict[int, torch.Tensor]:
        """
        Alternativa SEM dados: usa o per_layer_input_gate como proxy de importância.

        O per_layer_input_gate (2560→256) aprende a comprimir hidden states.
        A norma de cada dimensão da projeção indica quais neurônios MLP são
        mais frequentemente ativados via a correlação entre as projeções.

        Estratégia: usa magnitude dos pesos de gate_proj como proxy de importância.
        Neurônios com maior norma de peso tendem a ter maiores ativações.
        """
        layers = self._get_lm_layers()
        importance = {}

        for i, layer in enumerate(layers):
            # Norma L2 de cada linha de gate_proj = importância do neurônio
            # gate_proj.weight: (intermediate, hidden) → norma por linha
            weight_norms = layer.mlp.gate_proj.weight.data.float().norm(dim=1).cpu()
            importance[i] = weight_norms

        return importance

    def select_neurons(
        self,
        importance: Dict[int, torch.Tensor],
        keep_ratio: float = 0.5,
    ) -> Dict[int, torch.Tensor]:
        """
        Seleciona os top-k neurônios mais importantes por layer.

        Retorna dict {layer_idx: LongTensor(kept_neurons,)} com índices.
        """
        selection = {}
        for layer_idx, scores in importance.items():
            n_keep = max(1, int(len(scores) * keep_ratio))
            # Arredonda para múltiplo de 8 (melhor para hardware)
            n_keep = max(8, (n_keep // 8) * 8)
            _, top_indices = torch.topk(scores, n_keep, sorted=True)
            # Ordena índices (preserva ordem original → sem permutação de ativações)
            selection[layer_idx] = top_indices.sort().values
        return selection

    def analyze_importance_distribution(
        self,
        importance: Dict[int, torch.Tensor],
    ) -> None:
        """Imprime estatísticas de distribuição de importância por layer."""
        print(f"\n  {'Layer':>6} {'Min':>8} {'Max':>8} {'Mean':>8} {'Top10%':>8} {'Ratio':>8}")
        print(f"  {'─'*50}")
        for idx in sorted(importance.keys())[:10]:
            scores = importance[idx]
            top10 = scores.topk(len(scores) // 10).values.mean().item()
            rest = scores.mean().item()
            ratio = top10 / max(rest, 1e-8)
            print(f"  {idx:>6}  {scores.min():>8.4f}  {scores.max():>8.4f}  "
                  f"{scores.mean():>8.4f}  {top10:>8.4f}  {ratio:>7.1f}×")
        print(f"  ... ({len(importance)} layers total)")


# ─────────────────────────────────────────────────────────────────────────────
# PRUNER PRINCIPAL
# Aplica a remoção estrutural de neurônios no modelo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PruningReport:
    keep_ratio: float
    n_layers_pruned: int
    original_intermediate: int
    kept_intermediate: int
    params_before: int
    params_after: int
    flops_reduction: float
    layer_reports: List[dict] = field(default_factory=list)

    @property
    def param_reduction(self) -> float:
        return 1.0 - self.params_after / max(self.params_before, 1)

    def summary(self) -> str:
        lines = [
            f"\n{'═'*60}",
            f"  STRUCTURAL PRUNING REPORT",
            f"{'═'*60}",
            f"  Keep ratio:          {self.keep_ratio:.0%}",
            f"  Layers podados:      {self.n_layers_pruned}",
            f"  Intermediate orig:   {self.original_intermediate:,}",
            f"  Intermediate após:   {self.kept_intermediate:,}",
            f"  Params antes:        {self.params_before/1e9:.3f}B",
            f"  Params após:         {self.params_after/1e9:.3f}B",
            f"  Redução de params:   {self.param_reduction:.1%}",
            f"  Redução de FLOPs:    {self.flops_reduction:.1%}",
            f"  Speedup estimado:    {1/(1-self.flops_reduction*0.7):.2f}×",
            f"{'═'*60}",
        ]
        return "\n".join(lines)


class StructuralPruner:
    """
    Aplica structural pruning permanente no Gemma 4.

    Processo:
      1. Calibrar importância de neurônios (ou usar proxy de pesos)
      2. Selecionar top-k neurônios por layer
      3. Criar PrunedMLP com weights fisicamente menores
      4. Substituir o mlp original in-place
      5. Salvar modelo podado como checkpoint HuggingFace

    O modelo resultante tem MENOS parâmetros e roda MAIS RÁPIDO
    sem nenhuma modificação de runtime ou kernel especial.
    """

    def __init__(
        self,
        model: nn.Module,
        keep_ratio: float = 0.5,
        use_relu2: bool = True,
    ):
        self.model = model
        self.keep_ratio = keep_ratio
        self.use_relu2 = use_relu2
        self._report = None

    def _get_layers(self):
        try:
            return self.model.model.language_model.layers
        except AttributeError:
            try:
                return self.model.language_model.layers
            except AttributeError:
                return self.model.model.layers

    def prune(
        self,
        neuron_selection: Dict[int, torch.Tensor],
        verbose: bool = True,
    ) -> PruningReport:
        """
        Aplica a remoção estrutural baseada nos índices selecionados.

        Args:
            neuron_selection: {layer_idx: LongTensor(kept_indices,)}
            verbose: imprime progresso

        Returns:
            PruningReport com métricas
        """
        layers = self._get_layers()
        params_before = sum(p.numel() for p in self.model.parameters())

        layer_reports = []
        n_pruned = 0
        original_inter = 0
        total_kept = 0

        for layer_idx, layer in enumerate(layers):
            if layer_idx not in neuron_selection:
                continue

            kept_idx = neuron_selection[layer_idx].to(
                device=layer.mlp.gate_proj.weight.device
            )
            n_original = layer.mlp.gate_proj.out_features
            n_kept = len(kept_idx)
            original_inter = n_original
            total_kept += n_kept

            # Extrai subconjunto de pesos
            # gate_proj: (n_original, hidden) → seleciona LINHAS (neurônios)
            gate_w = layer.mlp.gate_proj.weight.data[kept_idx].clone()
            up_w   = layer.mlp.up_proj.weight.data[kept_idx].clone()
            # down_proj: (hidden, n_original) → seleciona COLUNAS (neurônios)
            down_w = layer.mlp.down_proj.weight.data[:, kept_idx].clone()

            hidden_size = layer.mlp.gate_proj.in_features
            dtype = layer.mlp.gate_proj.weight.dtype
            device = layer.mlp.gate_proj.weight.device

            # Cria PrunedMLP com os pesos selecionados
            pruned_mlp = PrunedMLP(
                hidden_size=hidden_size,
                kept_neurons=n_kept,
                gate_proj_weight=gate_w,
                up_proj_weight=up_w,
                down_proj_weight=down_w,
                use_relu2=self.use_relu2,
                neuron_indices=kept_idx.cpu(),
            ).to(device=device, dtype=dtype)

            # Substitui in-place
            layer.mlp = pruned_mlp
            n_pruned += 1

            report = {
                "layer": layer_idx,
                "original_neurons": n_original,
                "kept_neurons": n_kept,
                "prune_ratio": 1.0 - n_kept / n_original,
            }
            layer_reports.append(report)

            if verbose and (layer_idx % 10 == 0 or layer_idx == len(layers) - 1):
                print(f"  Layer {layer_idx:02d}: {n_original} → {n_kept} neurônios "
                      f"(-{1-n_kept/n_original:.0%})")

        params_after = sum(p.numel() for p in self.model.parameters())
        avg_kept = total_kept / max(n_pruned, 1)
        flops_reduction = 1.0 - (avg_kept / original_inter) if original_inter > 0 else 0

        self._report = PruningReport(
            keep_ratio=self.keep_ratio,
            n_layers_pruned=n_pruned,
            original_intermediate=original_inter,
            kept_intermediate=int(avg_kept),
            params_before=params_before,
            params_after=params_after,
            flops_reduction=flops_reduction,
            layer_reports=layer_reports,
        )

        if verbose:
            print(self._report.summary())

        return self._report

    def save(self, output_dir: str, tokenizer=None) -> None:
        """Salva o modelo podado como checkpoint HuggingFace padrão."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Salva modelo
        self.model.save_pretrained(output_dir, safe_serialization=True)

        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Salva metadados do pruning
        if self._report is not None:
            meta = {
                "pruning_type": "structural_mlp",
                "keep_ratio": self._report.keep_ratio,
                "original_intermediate": self._report.original_intermediate,
                "kept_intermediate": self._report.kept_intermediate,
                "param_reduction": self._report.param_reduction,
                "flops_reduction": self._report.flops_reduction,
                "use_relu2": self.use_relu2,
                "n_layers": self._report.n_layers_pruned,
            }
            with open(os.path.join(output_dir, "pruning_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

        print(f"  ✓ Modelo podado salvo em: {output_dir}")
