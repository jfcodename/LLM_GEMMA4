"""
Gemma 4 E4B — Fase 4: SSN Calibration + Sparse Kernel + INT4
==============================================================
Pipeline completo de otimização:
  1. Calibra SSN (score_proj) com dados reais
  2. Gather/Scatter MLP para speedup real
  3. INT4 quantização + esparsidade combinada

Uso no Kaggle:
    %cd /kaggle/working/LLM_GEMMA4
    !pip install datasets bitsandbytes -q
    !python unified/phase4_full_pipeline.py
"""

import argparse, logging, sys, time, gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from unified.phase1b_topk import benchmark, print_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: SSN CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

class CalibratedSSNMaskedMLP(nn.Module):
    """MLP com mascaramento SSN calibrado via dados reais."""

    def __init__(self, original_mlp, gate_linear, keep_ratio=0.50):
        super().__init__()
        self.mlp = original_mlp
        self.gate_linear = gate_linear
        self.keep_ratio = keep_ratio
        self.intermediate_size = original_mlp.gate_proj.out_features
        gate_dim = gate_linear.out_features

        # score_proj: será calibrado (não random!)
        self.score_proj = nn.Linear(gate_dim, self.intermediate_size, bias=False)
        self.score_proj = self.score_proj.to(
            device=gate_linear.weight.device, dtype=gate_linear.weight.dtype,
        )
        self.calibrated = False
        self._stats = {"total": 0, "zeros": 0}

    def calibrate(self, gate_outputs: torch.Tensor, activation_importances: torch.Tensor):
        """
        Calibra score_proj via regressão linear.
        gate_outputs: (N, gate_dim) — saída do per_layer_input_gate
        activation_importances: (N, intermediate_size) — magnitude das ativações
        """
        with torch.no_grad():
            # Least squares: W = (G^T G)^{-1} G^T A
            G = gate_outputs.float()
            A = activation_importances.float()
            try:
                W = torch.linalg.lstsq(G, A).solution  # (gate_dim, intermediate)
                self.score_proj.weight.data = W.T.to(self.score_proj.weight.dtype)
                self.calibrated = True
            except Exception as e:
                logger.warning(f"lstsq falhou, usando pseudo-inverse: {e}")
                G_pinv = torch.linalg.pinv(G)
                W = G_pinv @ A
                self.score_proj.weight.data = W.T.to(self.score_proj.weight.dtype)
                self.calibrated = True

    def forward(self, x):
        gate_output = self.mlp.act_fn(self.mlp.gate_proj(x))
        up_output = self.mlp.up_proj(x)

        # Score de importância via gate calibrado
        with torch.no_grad():
            gate_score = self.gate_linear(x)
            neuron_scores = self.score_proj(gate_score)

        k = max(1, int(self.intermediate_size * self.keep_ratio))
        _, topk_idx = torch.topk(neuron_scores.abs(), k, dim=-1, sorted=False)
        mask = torch.zeros_like(gate_output)
        mask.scatter_(-1, topk_idx, 1.0)

        intermediate = gate_output * up_output * mask
        output = self.mlp.down_proj(intermediate)

        with torch.no_grad():
            self._stats["total"] += intermediate.numel()
            self._stats["zeros"] += (intermediate == 0).sum().item()
        return output

    @property
    def actual_sparsity(self):
        return self._stats["zeros"] / max(self._stats["total"], 1)

    def reset_stats(self):
        self._stats = {"total": 0, "zeros": 0}


def collect_calibration_data(model, tokenizer, n_samples=50):
    """Coleta dados de calibração usando WikiText ou prompts sintéticos."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t for t in ds["text"] if len(t) > 100][:n_samples]
        logger.info(f"Usando WikiText-2 ({len(texts)} amostras)")
    except Exception:
        logger.info("WikiText indisponível, usando prompts sintéticos")
        texts = [
            "The theory of relativity was developed by Albert Einstein in the early 20th century.",
            "Python is a high-level programming language known for its simplicity and readability.",
            "Quantum computing leverages quantum mechanical phenomena to process information.",
            "The capital of France is Paris, which is known for the Eiffel Tower and the Louvre.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn.",
            "The mitochondria is the powerhouse of the cell, producing ATP through oxidative phosphorylation.",
            "Neural networks are computing systems inspired by biological neural networks in the brain.",
            "The Fibonacci sequence starts with 0 and 1, where each subsequent number is the sum.",
        ] * (n_samples // 8 + 1)
        texts = texts[:n_samples]

    # Coletar gate outputs e activation importances por layer
    layer_data = {}  # {layer_idx: {"gates": [], "importances": []}}

    gate_hooks = []
    mlp_hooks = []

    def find_text_layers(model):
        layers = {}
        for name, mod in model.named_modules():
            if hasattr(mod, 'act_fn') and hasattr(mod, 'gate_proj'):
                if "vision" not in name and "audio" not in name:
                    # Extract layer index
                    for part_i, part in enumerate(name.split(".")):
                        if part == "layers" and part_i + 1 < len(name.split(".")):
                            try:
                                idx = int(name.split(".")[part_i + 1])
                                layers[idx] = (name, mod)
                            except ValueError:
                                pass
        return layers

    text_layers = find_text_layers(model)
    logger.info(f"Encontradas {len(text_layers)} text layers para calibração")

    # Hook para capturar gate outputs
    def find_gate(model, layer_idx):
        for name, mod in model.named_modules():
            if (f"layers.{layer_idx}." in name and
                "per_layer_input_gate" in name and
                isinstance(mod, nn.Linear)):
                return mod
        return None

    for idx in sorted(text_layers.keys()):
        layer_data[idx] = {"gates": [], "importances": []}

    # Forward hooks
    hooks = []
    for idx, (name, mlp_mod) in text_layers.items():
        gate_mod = find_gate(model, idx)
        if gate_mod is None:
            continue

        def make_gate_hook(layer_idx):
            def hook(module, input, output):
                layer_data[layer_idx]["gates"].append(output.detach().float().cpu().reshape(-1, output.shape[-1]))
            return hook

        def make_mlp_hook(layer_idx, mlp):
            def hook(module, input, output):
                x = input[0] if isinstance(input, tuple) else input
                with torch.no_grad():
                    act = mlp.act_fn(mlp.gate_proj(x))
                    importance = act.abs().detach().float().cpu().reshape(-1, act.shape[-1])
                    layer_data[layer_idx]["importances"].append(importance)
            return hook

        hooks.append(gate_mod.register_forward_hook(make_gate_hook(idx)))
        hooks.append(mlp_mod.register_forward_hook(make_mlp_hook(idx, mlp_mod)))

    # Forward passes
    logger.info(f"Rodando {len(texts)} forward passes para calibração...")
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.inference_mode():
            _ = model(**inputs)
        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(texts)}")

    for h in hooks:
        h.remove()

    # Concatenar
    result = {}
    for idx in sorted(layer_data.keys()):
        if layer_data[idx]["gates"] and layer_data[idx]["importances"]:
            gates = torch.cat(layer_data[idx]["gates"], dim=0)
            imps = torch.cat(layer_data[idx]["importances"], dim=0)
            # Limitar para não estourar memória
            max_samples = 2000
            if gates.shape[0] > max_samples:
                perm = torch.randperm(gates.shape[0])[:max_samples]
                gates = gates[perm]
                imps = imps[perm]
            result[idx] = (gates, imps)

    logger.info(f"Calibração coletada para {len(result)} layers")
    return result


def patch_calibrated_ssn(model, calibration_data, keep_ratio=0.50):
    """Instala SSN calibrado em todas as text layers."""
    wrappers = []
    named_mods = dict(model.named_modules())

    for name, mod in list(named_mods.items()):
        if not hasattr(mod, 'act_fn') or not hasattr(mod, 'gate_proj'):
            continue
        if "vision" in name or "audio" in name:
            continue

        # Find layer index
        layer_idx = None
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass

        if layer_idx is None or layer_idx not in calibration_data:
            continue

        # Find gate
        gate = None
        for gname, gmod in model.named_modules():
            if (f"layers.{layer_idx}." in gname and
                "per_layer_input_gate" in gname and
                isinstance(gmod, nn.Linear)):
                gate = gmod
                break
        if gate is None:
            continue

        wrapper = CalibratedSSNMaskedMLP(mod, gate, keep_ratio=keep_ratio)

        # Calibrate!
        gates, importances = calibration_data[layer_idx]
        gates = gates.to(wrapper.score_proj.weight.device)
        importances = importances.to(wrapper.score_proj.weight.device)
        wrapper.calibrate(gates, importances)

        # Install
        parent_parts = name.rsplit(".", 1)
        if len(parent_parts) == 2:
            parent = named_mods[parent_parts[0]]
            setattr(parent, parent_parts[1], wrapper)
        wrappers.append(wrapper)

    logger.info(f"SSN Calibrado patched {len(wrappers)} MLPs (keep={keep_ratio:.0%})")
    return wrappers


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: GATHER/SCATTER SPARSE MLP
# ═══════════════════════════════════════════════════════════════════════════

class GatherScatterMLP(nn.Module):
    """
    MLP com gather/scatter para speedup real no down_proj.
    gate_proj e up_proj rodam full (precisam determinar importância).
    down_proj usa gather para computar apenas sobre neurônios ativos.
    """

    def __init__(self, original_mlp, keep_ratio=0.50):
        super().__init__()
        self.mlp = original_mlp
        self.keep_ratio = keep_ratio
        self._stats = {"total": 0, "zeros": 0, "flops_saved_pct": 0, "calls": 0}

    def forward(self, x):
        B, T, H = x.shape
        gate = self.mlp.act_fn(self.mlp.gate_proj(x))  # (B, T, D)
        up = self.mlp.up_proj(x)                         # (B, T, D)

        D = gate.shape[-1]
        k = max(1, int(D * self.keep_ratio))

        # Top-K e gather
        _, idx = torch.topk(gate.abs(), k, dim=-1, sorted=False)
        gate_sparse = torch.gather(gate, -1, idx)  # (B, T, k)
        up_sparse = torch.gather(up, -1, idx)       # (B, T, k)

        intermediate = gate_sparse * up_sparse  # (B, T, k) — MENOR!

        # Sparse down_proj: gather columns then matmul
        # down_proj.weight: (hidden, D) — queremos apenas colunas em idx
        # Para cada (b,t) os índices podem variar, então flatten
        flat_inter = intermediate.reshape(-1, k)  # (B*T, k)
        flat_idx = idx.reshape(-1, k)              # (B*T, k)

        # Gather weight columns per-token
        W = self.mlp.down_proj.weight  # (hidden, D)
        # Expand W para batch
        W_expanded = W.unsqueeze(0).expand(flat_inter.shape[0], -1, -1)  # (B*T, hidden, D)
        flat_idx_exp = flat_idx.unsqueeze(1).expand(-1, W.shape[0], -1)  # (B*T, hidden, k)
        W_gathered = torch.gather(W_expanded, 2, flat_idx_exp)  # (B*T, hidden, k)

        # Matmul: (B*T, hidden, k) @ (B*T, k, 1) → (B*T, hidden, 1)
        output = torch.bmm(W_gathered, flat_inter.unsqueeze(-1)).squeeze(-1)

        if self.mlp.down_proj.bias is not None:
            output = output + self.mlp.down_proj.bias

        output = output.reshape(B, T, -1)

        with torch.no_grad():
            self._stats["total"] += D * B * T
            self._stats["zeros"] += (D - k) * B * T
            self._stats["calls"] += 1
            self._stats["flops_saved_pct"] = 1.0 - self.keep_ratio

        return output

    @property
    def actual_sparsity(self):
        return self._stats["zeros"] / max(self._stats["total"], 1)

    def reset_stats(self):
        self._stats = {"total": 0, "zeros": 0, "flops_saved_pct": 0, "calls": 0}


def patch_gather_scatter(model, keep_ratio=0.50):
    """Substitui MLPs por GatherScatterMLP."""
    wrappers = []
    named_mods = dict(model.named_modules())
    for name, mod in list(named_mods.items()):
        if not hasattr(mod, 'act_fn') or not hasattr(mod, 'gate_proj'):
            continue
        if "vision" in name or "audio" in name:
            continue
        wrapper = GatherScatterMLP(mod, keep_ratio=keep_ratio)
        parent_parts = name.rsplit(".", 1)
        if len(parent_parts) == 2:
            parent = named_mods[parent_parts[0]]
            setattr(parent, parent_parts[1], wrapper)
        wrappers.append(wrapper)
    logger.info(f"GatherScatter patched {len(wrappers)} MLPs (keep={keep_ratio:.0%})")
    return wrappers


def unpatch_all(model):
    """Remove qualquer wrapper."""
    named_mods = dict(model.named_modules())
    count = 0
    for name, mod in list(named_mods.items()):
        if isinstance(mod, (CalibratedSSNMaskedMLP, GatherScatterMLP)):
            parent_parts = name.rsplit(".", 1)
            if len(parent_parts) == 2:
                parent = named_mods.get(parent_parts[0])
                if parent:
                    setattr(parent, parent_parts[1], mod.mlp)
                    count += 1
    logger.info(f"Unpatched {count} MLPs")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Full Pipeline")
    parser.add_argument("--model-id", default="google/gemma-4-e4b-it")
    parser.add_argument("--keep-ratio", type=float, default=0.50)
    parser.add_argument("--cal-samples", type=int, default=50)
    parser.add_argument("--skip-int4", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("GPU necessária")
        return 1

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n{'═'*60}")
    print(f"  FASE 4: FULL OPTIMIZATION PIPELINE")
    print(f"  GPU: {gpu} ({vram:.1f} GB)")
    print(f"  Keep ratio: {args.keep_ratio:.0%}")
    print(f"{'═'*60}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
    ]

    # ══════════════════════════════════════════════════════════════════
    # STEP 0: LOAD MODEL (bf16)
    # ══════════════════════════════════════════════════════════════════

    logger.info(f"Carregando {args.model_id} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Baseline
    print(f"\n{'─'*60}")
    print(f"  BASELINE (bf16 denso)")
    print(f"{'─'*60}")
    baseline = benchmark(model, tokenizer, prompts)
    print_results("bf16 denso", baseline, sparsity=0.0)
    base_tps = sum(r["tok_per_s"] for r in baseline) / len(baseline)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: SSN CALIBRATION
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  STEP 1: SSN CALIBRATION ({args.cal_samples} amostras)")
    print(f"{'─'*60}")

    cal_data = collect_calibration_data(model, tokenizer, n_samples=args.cal_samples)
    wrappers = patch_calibrated_ssn(model, cal_data, keep_ratio=args.keep_ratio)
    n_calibrated = sum(1 for w in wrappers if w.calibrated)
    print(f"\n    Layers calibrados: {n_calibrated}/{len(wrappers)}")

    for w in wrappers:
        w.reset_stats()

    ssn_results = benchmark(model, tokenizer, prompts)
    ssn_sp = sum(w.actual_sparsity * w._stats["total"] for w in wrappers if w._stats["total"] > 0)
    ssn_n = sum(w._stats["total"] for w in wrappers if w._stats["total"] > 0)
    ssn_sparsity = ssn_sp / ssn_n if ssn_n > 0 else 0

    print_results("SSN Calibrado", ssn_results, sparsity=ssn_sparsity)
    ssn_tps = sum(r["tok_per_s"] for r in ssn_results) / len(ssn_results)

    unpatch_all(model)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: GATHER/SCATTER SPARSE MLP
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  STEP 2: GATHER/SCATTER SPARSE MLP")
    print(f"{'─'*60}")

    gs_wrappers = patch_gather_scatter(model, keep_ratio=args.keep_ratio)
    for w in gs_wrappers:
        w.reset_stats()

    gs_results = benchmark(model, tokenizer, prompts)
    gs_sp = sum(w.actual_sparsity * w._stats["total"] for w in gs_wrappers if w._stats["total"] > 0)
    gs_n = sum(w._stats["total"] for w in gs_wrappers if w._stats["total"] > 0)
    gs_sparsity = gs_sp / gs_n if gs_n > 0 else 0
    flops_saved = gs_wrappers[0]._stats["flops_saved_pct"] if gs_wrappers else 0

    print_results("Gather/Scatter", gs_results, sparsity=gs_sparsity)
    gs_tps = sum(r["tok_per_s"] for r in gs_results) / len(gs_results)
    print(f"    FLOPs saved (down_proj): {flops_saved:.0%}")

    unpatch_all(model)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: INT4 QUANTIZATION + SPARSITY
    # ══════════════════════════════════════════════════════════════════

    int4_tps = None
    int4_sparse_tps = None

    if not args.skip_int4:
        print(f"\n{'─'*60}")
        print(f"  STEP 3: INT4 QUANTIZATION + SPARSITY")
        print(f"{'─'*60}")

        # Liberar modelo bf16
        del model
        gc.collect()
        torch.cuda.empty_cache()

        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            logger.info(f"Carregando {args.model_id} (INT4/NF4)...")
            model_int4 = AutoModelForCausalLM.from_pretrained(
                args.model_id, quantization_config=bnb_config,
                device_map="auto", low_cpu_mem_usage=True,
            )
            model_int4.eval()

            vram_int4 = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"\n    VRAM com INT4: {vram_int4:.2f} GB")

            # Benchmark INT4 denso
            int4_results = benchmark(model_int4, tokenizer, prompts)
            print_results("INT4/NF4 denso", int4_results, sparsity=0.0)
            int4_tps = sum(r["tok_per_s"] for r in int4_results) / len(int4_results)

            # INT4 + Top-K 50%
            from unified.phase1b_topk import patch_mlps as patch_topk, unpatch_mlps
            topk_w = patch_topk(model_int4, keep_ratio=args.keep_ratio, text_only=True)
            for w in topk_w:
                w.reset_stats()

            int4_sparse_results = benchmark(model_int4, tokenizer, prompts)
            print_results("INT4 + Top-K 50%", int4_sparse_results, sparsity=0.50)
            int4_sparse_tps = sum(r["tok_per_s"] for r in int4_sparse_results) / len(int4_sparse_results)

            unpatch_mlps(model_int4)
            del model_int4

        except Exception as e:
            logger.error(f"INT4 falhou: {e}")
            logger.info("Instale: pip install bitsandbytes>=0.43.0")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print(f"  RESUMO COMPLETO — FASE 4")
    print(f"{'═'*60}")
    print(f"  {'Config':<30} {'tok/s':>8} {'vs base':>8} {'Notas':>20}")
    print(f"  {'─'*68}")
    print(f"  {'bf16 denso':<30} {base_tps:>7.1f} {'1.00×':>8} {'referência':>20}")
    print(f"  {'SSN calibrado (50%)':<30} {ssn_tps:>7.1f} {ssn_tps/base_tps:>7.2f}× {'gate inteligente':>20}")
    print(f"  {'Gather/Scatter (50%)':<30} {gs_tps:>7.1f} {gs_tps/base_tps:>7.2f}× {'speedup real':>20}")

    if int4_tps:
        print(f"  {'INT4/NF4 denso':<30} {int4_tps:>7.1f} {int4_tps/base_tps:>7.2f}× {'4× menos VRAM':>20}")
    if int4_sparse_tps:
        print(f"  {'INT4 + Top-K 50%':<30} {int4_sparse_tps:>7.1f} {int4_sparse_tps/base_tps:>7.2f}× {'quant + sparse':>20}")

    print(f"{'═'*60}\n")
    return 0


if __name__ == "__main__":
    main()
