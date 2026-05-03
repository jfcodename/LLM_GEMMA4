"""
Gemma 4 Neo — Conversor Principal

Carrega os pesos do Gemma 4 original e aplica as modificações arquiteturais
em sequência. Cada passo é independente, incremental e pode ser salvo/retomado.

Uso:
    python convert.py --model google/gemma-4-e2b-it --steps relu2 gated_attn quant
    python convert.py --model google/gemma-4-e2b-it --steps all --output ./neo_e2b

Sequência recomendada:
    Passo 1: quant         → W4A8 quantização (zero retreino, impacto imediato)
    Passo 2: relu2         → ReLU² MLP + SparsityPredictor (requer calibração)
    Passo 3: gated_attn   → Gated Attention nos layers globais (requer fine-tune)
    Passo 4: mamba2        → Mamba-2 SSM nos layers locais (requer destilação)
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import Gemma4NeoConfig, GLOBAL_LAYER_INDICES
from modules import (
    SparsityPredictor,
    ReLU2GatedMLP,
    GatedAttentionLayer,
    SnapKVCache,
    Mamba2Block,
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def get_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """Acessa um decoder layer pelo índice (compatível com Gemma 4)."""
    # Caminho: model.model.language_model.layers[i]
    lm = model.model.language_model
    return lm.layers[layer_idx]


def get_mlp(layer: nn.Module) -> nn.Module:
    return layer.mlp


def get_attn(layer: nn.Module) -> nn.Module:
    return layer.self_attn


def count_parameters(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{total/1e9:.2f}B total, {trainable/1e9:.2f}B trainable"


def get_calibration_batch(
    tokenizer,
    dataset_name: str = "wikitext",
    n_samples: int = 128,
    max_length: int = 512,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Gera um batch de calibração para estimar thresholds de ativação.
    Usa wikitext-2 por padrão (leve, diverso, representativo).
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split="train")
        texts = [t["text"] for t in dataset if len(t["text"]) > 200][:n_samples]
    except Exception:
        print("  ⚠ Dataset não disponível. Usando texto de fallback para calibração.")
        texts = ["The quick brown fox jumps over the lazy dog. " * 20] * n_samples

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return encodings.input_ids.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class Gemma4NeoConverter:
    """
    Orquestra a conversão do Gemma 4 original para a arquitetura Neo.

    Cada passo é aplicado in-place ao modelo carregado em memória.
    O modelo modificado pode ser salvo como checkpoint HuggingFace padrão.
    """

    def __init__(self, config: Gemma4NeoConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.applied_steps: List[str] = []
        self._log = []

    # ── Carregamento ─────────────────────────────────────────────────────────

    def load_base_model(self) -> "Gemma4NeoConverter":
        """
        Carrega o Gemma 4 original com pesos completos.
        Suporta carregamento em BF16 para economizar memória.
        """
        print(f"\n{'─'*60}")
        print(f" Carregando: {self.config.base_model_id}")
        print(f"{'─'*60}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_id,
            trust_remote_code=True,
        )

        dtype = getattr(torch, self.config.torch_dtype)

        # Se quantização BnB está habilitada, carrega já quantizado
        if (self.config.quantization.enabled
                and self.config.quantization.use_bnb
                and "quant" in self._pending_steps):
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                quantization_config=bnb_cfg,
                device_map=self.config.device_map,
                trust_remote_code=True,
            )
            print("  ✓ Carregado com quantização NF4 (bitsandbytes)")
            self.applied_steps.append("quant_bnb")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                torch_dtype=dtype,
                device_map=self.config.device_map,
                trust_remote_code=True,
            )
            print(f"  ✓ Carregado em {self.config.torch_dtype}")

        print(f"  Parâmetros: {count_parameters(self.model)}")
        self._log_step("load", {"model_id": self.config.base_model_id})
        return self

    # ── Passo 1: Quantização ─────────────────────────────────────────────────

    def apply_quantization(self) -> "Gemma4NeoConverter":
        """
        W4A8 com bitsandbytes (simples e robusto).
        Para AWQ completo, veja apply_awq_quantization() abaixo.
        """
        if "quant_bnb" in self.applied_steps:
            print("\n  ✓ Quantização BnB já aplicada no carregamento.")
            return self

        print(f"\n{'─'*60}")
        print(" Passo 1: Quantização W4A8")
        print(f"{'─'*60}")

        try:
            import bitsandbytes as bnb
        except ImportError:
            print("  ⚠ bitsandbytes não instalado. Pulando quantização.")
            print("    Instale com: pip install bitsandbytes")
            return self

        # Converte Linear → Linear4bit em todas as camadas exceto embedding e lm_head
        def _quantize_linear(module: nn.Module, name: str = ""):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                if isinstance(child, nn.Linear) and not any(
                    skip in full_name for skip in ["embed_tokens", "lm_head", "score_proj"]
                ):
                    new_linear = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.bfloat16,
                        compress_statistics=True,
                        quant_type="nf4",
                    )
                    # Copia pesos (quantiza on-the-fly)
                    new_linear.weight = bnb.nn.Params4bit(
                        child.weight.data,
                        requires_grad=False,
                        quant_type="nf4",
                    )
                    if child.bias is not None:
                        new_linear.bias = child.bias
                    setattr(module, child_name, new_linear)
                else:
                    _quantize_linear(child, full_name)

        _quantize_linear(self.model)
        self.applied_steps.append("quant")
        print("  ✓ Quantização NF4 aplicada em todos os Linear layers")
        self._log_step("quant", {"bits": 4, "type": "nf4"})
        return self

    def apply_awq_quantization(self, calibration_ids: torch.Tensor) -> "Gemma4NeoConverter":
        """
        Quantização AWQ (mais precisa que NF4, requer calibração).
        Instalação: pip install autoawq
        """
        print(f"\n{'─'*60}")
        print(" Passo 1b: Quantização AWQ W4")
        print(f"{'─'*60}")
        try:
            from awq import AutoAWQForCausalLM

            awq_model = AutoAWQForCausalLM.from_pretrained(
                self.config.base_model_id,
                torch_dtype=torch.bfloat16,
            )
            quant_config = {
                "zero_point": True,
                "q_group_size": self.config.quantization.group_size,
                "w_bit": self.config.quantization.weight_bits,
                "version": "GEMM",
            }
            # Calibração com exemplos reais
            calib_texts = self.tokenizer.batch_decode(calibration_ids)
            awq_model.quantize(
                self.tokenizer,
                quant_config=quant_config,
                calib_data=calib_texts,
            )
            # Substitui o modelo pelo AWQ
            self.model = awq_model
            self.applied_steps.append("quant_awq")
            print("  ✓ AWQ W4 aplicado com calibração")
        except ImportError:
            print("  ⚠ autoawq não instalado. Use: pip install autoawq")
        return self

    # ── Passo 2: ReLU² MLP + Sparsity Predictor ──────────────────────────────

    def apply_relu2_mlp(
        self,
        calibration_ids: Optional[torch.Tensor] = None,
    ) -> "Gemma4NeoConverter":
        """
        Substitui GELUTanh → ReLU² em todos os 42 layers MLP.
        Reutiliza pesos gate_proj, up_proj, down_proj do checkpoint.
        Cria SparsityPredictor reutilizando per_layer_input_gate.
        """
        print(f"\n{'─'*60}")
        print(f" Passo 2: ReLU² MLP + SparsityPredictor")
        print(f" Target sparsidade: {self.config.relu2.sparsity_target:.0%}")
        print(f"{'─'*60}")

        n_layers = self.config.num_hidden_layers
        cfg = self.config.relu2

        for layer_idx in range(n_layers):
            layer = get_layer(self.model, layer_idx)
            mlp = get_mlp(layer)

            # Extrai pesos originais do MLP
            gw = mlp.gate_proj.weight.data.clone()
            uw = mlp.up_proj.weight.data.clone()
            dw = mlp.down_proj.weight.data.clone()

            # Extrai pesos do per_layer_input_gate (já existente)
            gate_in_w = None
            gate_norm_w = None

            if hasattr(layer, "per_layer_input_gate"):
                gate_in_w = layer.per_layer_input_gate.weight.data.clone()
            if hasattr(layer, "post_per_layer_input_norm"):
                gate_norm_w = layer.post_per_layer_input_norm.weight.data.clone()

            # Cria SparsityPredictor reutilizando pesos do gate
            predictor = SparsityPredictor(
                hidden_size=self.config.hidden_size,
                gate_bottleneck_size=self.config.gate_bottleneck_size,
                intermediate_size=self.config.intermediate_size,
                config=cfg,
                gate_weight_in=gate_in_w,
                gate_norm_weight=gate_norm_w,
            ).to(gw.device).to(gw.dtype)

            # Cria ReLU² MLP (carrega pesos originais)
            new_mlp = ReLU2GatedMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                config=cfg,
                gate_proj_weight=gw,
                up_proj_weight=uw,
                down_proj_weight=dw,
                sparsity_predictor=predictor,
            ).to(gw.device).to(gw.dtype)

            # Substitui o MLP na layer
            layer.mlp = new_mlp
            # Registra predictor como atributo da layer para acesso fácil
            layer.sparsity_predictor = predictor

            if layer_idx % 10 == 0 or layer_idx == n_layers - 1:
                print(f"  Layer {layer_idx:02d}/{n_layers-1} ✓")

        # Calibração de thresholds (se dados disponíveis)
        if calibration_ids is not None and not cfg.use_gate_as_predictor:
            print("\n  Calibrando thresholds por layer...")
            self._calibrate_relu2_thresholds(calibration_ids)

        self.applied_steps.append("relu2")
        print(f"\n  ✓ ReLU² aplicado em {n_layers} layers")
        self._log_step("relu2", {
            "sparsity_target": cfg.sparsity_target,
            "use_gate_predictor": cfg.use_gate_as_predictor,
        })
        return self

    @torch.no_grad()
    def _calibrate_relu2_thresholds(self, calibration_ids: torch.Tensor):
        """Roda forward pass de calibração para estimar thresholds por layer."""
        self.model.eval()
        # Hooks para capturar ativações
        hooks = []
        calib_data_per_layer = {}

        def make_hook(idx):
            def hook(module, inp, out):
                if idx not in calib_data_per_layer:
                    calib_data_per_layer[idx] = []
                # Captura a entrada do MLP (hidden states)
                calib_data_per_layer[idx].append(inp[0].detach().cpu()[:, :32, :])
            return hook

        for i in range(self.config.num_hidden_layers):
            layer = get_layer(self.model, i)
            h = layer.mlp.register_forward_hook(make_hook(i))
            hooks.append(h)

        # Forward pass de calibração
        batch = calibration_ids[:32].to(self.device)
        with torch.no_grad():
            self.model(batch)

        for h in hooks:
            h.remove()

        # Calibra threshold em cada layer
        for i in range(self.config.num_hidden_layers):
            layer = get_layer(self.model, i)
            if isinstance(layer.mlp, ReLU2GatedMLP) and i in calib_data_per_layer:
                data = torch.cat(calib_data_per_layer[i], dim=0).to(self.device)
                data = data.to(next(layer.mlp.parameters()).dtype)
                sp = layer.mlp.calibrate_threshold(data, percentile=35.0)
                if i % 10 == 0:
                    print(f"    Layer {i:02d}: sparsidade={sp:.1%}")

    # ── Passo 3: Gated Attention nos layers globais ──────────────────────────

    def apply_gated_attention(self) -> "Gemma4NeoConverter":
        """
        Adiciona gate sigmoid após SDPA nos 7 layers globais.
        Pesos q/k/v/o do Gemma 4 são PRESERVADOS — apenas Wθ é novo.
        """
        print(f"\n{'─'*60}")
        print(f" Passo 3: Gated Attention nos layers globais")
        print(f" Layers: {sorted(GLOBAL_LAYER_INDICES)}")
        print(f"{'─'*60}")

        cfg = self.config.gated_attention

        for layer_idx in sorted(GLOBAL_LAYER_INDICES):
            layer = get_layer(self.model, layer_idx)
            attn = get_attn(layer)

            # Determina num_heads e head_dim do layer global (q=4096)
            q_dim = attn.q_proj.weight.shape[0]  # 4096
            h = self.config.num_global_heads      # 16
            head_dim = q_dim // h                 # 256

            # Cria SnapKV cache para este layer
            snap_kv = None
            if cfg.snap_kv_enabled:
                snap_kv = SnapKVCache(
                    max_capacity=cfg.snap_kv_max_capacity,
                    window=cfg.snap_kv_window,
                )

            # Cria módulo Gated Attention
            gated_attn = GatedAttentionLayer(
                hidden_size=self.config.hidden_size,
                num_heads=h,
                head_dim=head_dim,
                config=cfg,
                snap_kv_cache=snap_kv,
            ).to(next(attn.parameters()).device).to(next(attn.parameters()).dtype)

            # Armazena referência ao módulo de atenção original
            # (pesos q/k/v/o permanecem no original)
            layer._original_attn = attn
            layer._gated_attn = gated_attn

            # Patch no forward do layer para usar o gate
            _patch_layer_with_gated_attn(layer)

            print(f"  Layer {layer_idx:02d}: Gated Attention "
                  f"({h} heads, head_dim={head_dim}, "
                  f"SnapKV={'✓' if cfg.snap_kv_enabled else '✗'})")

        self.applied_steps.append("gated_attn")
        print(f"\n  ✓ Gated Attention aplicado em {len(GLOBAL_LAYER_INDICES)} layers")
        self._log_step("gated_attn", {
            "layers": sorted(GLOBAL_LAYER_INDICES),
            "snap_kv": cfg.snap_kv_enabled,
        })
        return self

    # ── Passo 4: Mamba-2 nos layers locais ───────────────────────────────────

    def apply_mamba2(self) -> "Gemma4NeoConverter":
        """
        Substitui os 35 layers locais (SWA) por blocos Mamba-2.

        ⚠ ATENÇÃO: Esta substituição REQUER destilação para recuperar accuracy.
        Use mamba2_distill.py para treinar os blocos Mamba-2 antes de chamar este
        método em produção. Sem destilação, o modelo degrada significativamente.

        Para testes rápidos sem destilação: os blocos Mamba-2 são inicializados
        aleatoriamente e o modelo precisará de fine-tuning.
        """
        print(f"\n{'─'*60}")
        print(f" Passo 4: Mamba-2 SSM (layers locais)")
        print(f"{'─'*60}")

        if not self.config.mamba2.enabled:
            print("  ⚠ Mamba-2 desabilitado no config. "
                  "Ative com: config.mamba2.enabled = True")
            return self

        cfg = self.config.mamba2
        local_layers = [i for i in range(self.config.num_hidden_layers)
                        if i not in GLOBAL_LAYER_INDICES]

        print(f"  Substituindo {len(local_layers)} layers locais...")
        print(f"  d_state={cfg.d_state}, expand={cfg.expand}, headdim={cfg.headdim}")

        if cfg.pretrained_mamba_path:
            print(f"  Carregando pesos pré-destilados de: {cfg.pretrained_mamba_path}")

        for layer_idx in local_layers:
            layer = get_layer(self.model, layer_idx)
            device = next(layer.parameters()).device
            dtype = next(layer.parameters()).dtype

            if cfg.pretrained_mamba_path:
                # Carrega pesos de destilação por layer
                layer_path = os.path.join(
                    cfg.pretrained_mamba_path, f"mamba2_layer_{layer_idx}.pt"
                )
                mamba_block = Mamba2Block.from_pretrained_path(
                    layer_path if os.path.exists(layer_path) else None,
                    hidden_size=self.config.hidden_size,
                    config=cfg,
                )
            else:
                mamba_block = Mamba2Block(
                    hidden_size=self.config.hidden_size,
                    config=cfg,
                )

            mamba_block = mamba_block.to(device).to(dtype)

            # Preserva o módulo original para possível rollback
            layer._original_attn_backup = layer.self_attn
            # Substitui atenção pelo SSM
            layer.self_attn = mamba_block
            # Patch do forward para ignorar position_ids/mask no SSM
            _patch_layer_with_mamba2(layer)

            if layer_idx % 10 == 0:
                print(f"  Layer {layer_idx:02d} ✓")

        self.applied_steps.append("mamba2")
        print(f"\n  ✓ Mamba-2 aplicado em {len(local_layers)} layers")
        print(f"  ⚠ LEMBRE: Execute destilação (mamba2_distill.py) antes de usar!")
        self._log_step("mamba2", {
            "local_layers": len(local_layers),
            "d_state": cfg.d_state,
            "pretrained": cfg.pretrained_mamba_path is not None,
        })
        return self

    # ── Salvar / Carregar ─────────────────────────────────────────────────────

    def save(self, output_dir: Optional[str] = None) -> "Gemma4NeoConverter":
        """
        Salva o modelo Neo modificado + tokenizer + manifest de modificações.
        """
        out = output_dir or self.config.output_dir
        Path(out).mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f" Salvando em: {out}")
        print(f"{'─'*60}")

        # Salva modelo (PyTorch state dict + HuggingFace config)
        self.model.save_pretrained(out, safe_serialization=True)
        self.tokenizer.save_pretrained(out)

        # Salva manifest
        manifest = {
            "base_model": self.config.base_model_id,
            "applied_steps": self.applied_steps,
            "global_layers": sorted(GLOBAL_LAYER_INDICES),
            "config": {
                "hidden_size": self.config.hidden_size,
                "intermediate_size": self.config.intermediate_size,
                "num_layers": self.config.num_hidden_layers,
                "sparsity_target": self.config.relu2.sparsity_target,
                "gated_attention": self.config.gated_attention.enabled,
                "mamba2": self.config.mamba2.enabled,
            },
            "conversion_log": self._log,
        }
        with open(os.path.join(out, "neo_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  ✓ Modelo salvo")
        print(f"  ✓ Modificações aplicadas: {', '.join(self.applied_steps)}")
        return self

    # ── Verificação / Report ──────────────────────────────────────────────────

    def report(self):
        """Exibe um resumo das modificações aplicadas e métricas estimadas."""
        print(f"\n{'═'*60}")
        print(f"  GEMMA 4 NEO — RELATÓRIO DE CONVERSÃO")
        print(f"{'═'*60}")
        print(f"\n  Modelo base:   {self.config.base_model_id}")
        print(f"  Parâmetros:    {count_parameters(self.model)}")
        print(f"\n  Modificações aplicadas:")
        for step in self.applied_steps:
            print(f"    ✓ {step}")
        not_applied = [s for s in ["quant", "relu2", "gated_attn", "mamba2"]
                       if not any(s in a for a in self.applied_steps)]
        for step in not_applied:
            print(f"    ○ {step} (não aplicado)")

        # Estima sparsidade MLP atual
        if "relu2" in self.applied_steps:
            sparsities = []
            for i in range(self.config.num_hidden_layers):
                layer = get_layer(self.model, i)
                if hasattr(layer, "sparsity_predictor"):
                    sparsities.append(layer.sparsity_predictor.actual_sparsity)
            if sparsities:
                avg_sp = sum(sparsities) / len(sparsities)
                print(f"\n  Sparsidade MLP média: {avg_sp:.1%}")

        print(f"\n  Layers globais (Gated Attention): {sorted(GLOBAL_LAYER_INDICES)}")
        print(f"  Layers locais ({'Mamba-2' if 'mamba2' in self.applied_steps else 'SWA original'}):"
              f" {35} layers")
        print(f"{'═'*60}\n")

    # ── Interno ───────────────────────────────────────────────────────────────

    def _log_step(self, step: str, params: dict):
        self._log.append({"step": step, "timestamp": time.time(), **params})

    @property
    def _pending_steps(self) -> List[str]:
        return []  # placeholder


# ─────────────────────────────────────────────────────────────────────────────
# PATCHES DE FORWARD PASS
# Modificam o __call__ dos decoder layers para usar os novos módulos
# sem re-escrever a classe inteira do Gemma 4.
# ─────────────────────────────────────────────────────────────────────────────

def _patch_layer_with_gated_attn(layer: nn.Module):
    """
    Substitui o forward do decoder layer para usar GatedAttentionLayer
    nos layers globais, preservando todos os outros componentes (MLP, norms, etc).
    """
    original_forward = layer.forward

    def gated_forward(
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        # Pre-norm
        x = layer.input_layernorm(hidden_states)

        # Gated Attention (substitui self_attn original)
        attn_out, pkv = layer._gated_attn.forward(
            original_attn_module=layer._original_attn,
            x=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        # Post-attn residual + norm
        hidden_states = residual + attn_out
        if hasattr(layer, 'post_attention_layernorm'):
            hidden_states = layer.post_attention_layernorm(hidden_states)

        # Per-layer gate (já existente no Gemma 4) → agora também como predictor
        predictor_mask = None
        if hasattr(layer, 'sparsity_predictor'):
            predictor_mask = layer.sparsity_predictor(hidden_states)

        # Pre-feedforward norm
        if hasattr(layer, 'pre_feedforward_layernorm'):
            mlp_input = layer.pre_feedforward_layernorm(hidden_states)
        else:
            mlp_input = hidden_states

        # MLP (ReLU² se já aplicado, senão original)
        residual2 = hidden_states
        if isinstance(layer.mlp, ReLU2GatedMLP):
            mlp_out = layer.mlp(mlp_input, predictor_mask=predictor_mask)
        else:
            mlp_out = layer.mlp(mlp_input)

        hidden_states = residual2 + mlp_out
        if hasattr(layer, 'post_feedforward_layernorm'):
            hidden_states = layer.post_feedforward_layernorm(hidden_states)

        if use_cache:
            return hidden_states, pkv
        return (hidden_states,)

    layer.forward = gated_forward


def _patch_layer_with_mamba2(layer: nn.Module):
    """
    Substitui o forward do decoder layer para usar Mamba2Block
    nos layers locais (ignora position_ids e attention_mask — SSM não usa).
    """
    def mamba2_forward(
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        # Pre-norm
        x = layer.input_layernorm(hidden_states)

        # Mamba-2 SSM (ignora mask/position — processa sequencialmente)
        ssm_out = layer.self_attn(x)  # self_attn agora é Mamba2Block
        hidden_states = residual + ssm_out

        # Per-layer gate como predictor
        predictor_mask = None
        if hasattr(layer, 'sparsity_predictor'):
            predictor_mask = layer.sparsity_predictor(hidden_states)

        # Pre-feedforward norm
        if hasattr(layer, 'pre_feedforward_layernorm'):
            mlp_input = layer.pre_feedforward_layernorm(hidden_states)
        else:
            mlp_input = hidden_states

        residual2 = hidden_states
        if isinstance(layer.mlp, ReLU2GatedMLP):
            mlp_out = layer.mlp(mlp_input, predictor_mask=predictor_mask)
        else:
            mlp_out = layer.mlp(mlp_input)

        hidden_states = residual2 + mlp_out
        if hasattr(layer, 'post_feedforward_layernorm'):
            hidden_states = layer.post_feedforward_layernorm(hidden_states)

        return (hidden_states,)

    layer.forward = mamba2_forward


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Converte Gemma 4 → Gemma 4 Neo")
    p.add_argument("--model", default="google/gemma-4-e2b-it",
                   help="Model ID HuggingFace ou caminho local")
    p.add_argument("--steps", nargs="+",
                   choices=["quant", "relu2", "gated_attn", "mamba2", "all"],
                   default=["quant", "relu2", "gated_attn"],
                   help="Passos de conversão a aplicar")
    p.add_argument("--output", default="./gemma4_neo_checkpoint",
                   help="Diretório de saída para o modelo convertido")
    p.add_argument("--sparsity", type=float, default=0.65,
                   help="Target de esparsidade MLP (0.0–0.95)")
    p.add_argument("--no-bnb", action="store_true",
                   help="Usa FP16 em vez de quantização BnB")
    p.add_argument("--calibrate", action="store_true",
                   help="Executa calibração de threshold ReLU²")
    p.add_argument("--mamba-weights", type=str, default=None,
                   help="Caminho para pesos Mamba-2 pré-destilados")
    return p.parse_args()


def main():
    args = parse_args()
    steps = args.steps
    if "all" in steps:
        steps = ["quant", "relu2", "gated_attn", "mamba2"]

    # Monta config
    cfg = Gemma4NeoConfig(
        base_model_id=args.model,
        output_dir=args.output,
    )
    cfg.relu2.sparsity_target = args.sparsity
    cfg.quantization.enabled = "quant" in steps and not args.no_bnb
    cfg.gated_attention.enabled = "gated_attn" in steps
    cfg.mamba2.enabled = "mamba2" in steps
    cfg.mamba2.pretrained_mamba_path = args.mamba_weights

    print(f"\n{'═'*60}")
    print(f"  GEMMA 4 NEO — CONVERSÃO")
    print(f"{'═'*60}")
    print(f"  Modelo: {cfg.base_model_id}")
    print(f"  Passos: {steps}")
    print(f"  Output: {cfg.output_dir}")
    print(f"  Modificações ativas: {', '.join(cfg.active_modifications())}")

    # Executa conversão
    converter = Gemma4NeoConverter(cfg)
    converter.load_base_model()

    # Prepara dados de calibração (se necessário)
    calib_ids = None
    if "relu2" in steps and args.calibrate:
        calib_ids = get_calibration_batch(converter.tokenizer, n_samples=128)

    if "relu2" in steps:
        converter.apply_relu2_mlp(calibration_ids=calib_ids)

    if "gated_attn" in steps:
        converter.apply_gated_attention()

    if "mamba2" in steps:
        converter.apply_mamba2()

    # Quant por último (ou no carregamento se BnB)
    if "quant" in steps and not args.no_bnb and "quant" not in converter.applied_steps:
        converter.apply_quantization()

    converter.report()
    converter.save()
    print(f"\n  Conversão concluída! Modelo salvo em: {cfg.output_dir}\n")


if __name__ == "__main__":
    main()
