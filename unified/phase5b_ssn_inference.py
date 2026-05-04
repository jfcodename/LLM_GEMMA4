"""
Gemma 4 E4B — Fase 5b: Inferência com SSN Low-Rank Router
=========================================================
Este script carrega o modelo original e acopla os Cérebros Secundários (Routers)
treinados no passo anterior.

Como funciona a aceleração de CPU/GPU:
Durante a geração (seq_len == 1), o Router prevê os neurônios ativos.
Nós FATIAMOS (slice) as matrizes de pesos (gate, up, down) dinamicamente 
antes de multiplicar. Assim, reduzimos o FLOP real de 10240 para 5120!

Uso no Kaggle:
    !python unified/phase5b_ssn_inference.py
"""

import os
import sys
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Importar a mesma arquitetura do Router do script anterior
from unified.phase5_ssn_calibration import LowRankRouter

class SSNDynamicMLP(nn.Module):
    """
    MLP que fatiará dinamicamente os pesos com base na previsão do Router.
    Geração rápida se seq_len == 1 (Autoregressive).
    """
    def __init__(self, original_mlp, router, keep_ratio=0.50):
        super().__init__()
        # Desempacotar pesos para acesso rápido
        self.gate_weight = original_mlp.gate_proj.weight
        self.up_weight = original_mlp.up_proj.weight
        self.down_weight = original_mlp.down_proj.weight
        self.act_fn = original_mlp.act_fn if hasattr(original_mlp, 'act_fn') else None
        
        self.router = router
        self.router.eval() # Router sempre em eval mode
        self.k = int(self.gate_weight.shape[0] * keep_ratio)
        
    def forward(self, x):
        bsz, seq_len, dim = x.shape
        
        # Otk: Otimização de Geração (1 token por vez)
        if seq_len == 1:
            with torch.no_grad():
                # 1. Router prevê quem deve ligar
                logits = self.router(x) # [1, 1, 10240]
                _, topk_indices = torch.topk(logits, self.k, dim=-1)
                idx = topk_indices.squeeze() # [5120]
                
                # 2. Slice dinâmico das matrizes (Zero FLOPs para neurônios desligados!)
                active_gate_w = self.gate_weight[idx, :] # [5120, 2560]
                active_up_w = self.up_weight[idx, :]     # [5120, 2560]
                active_down_w = self.down_weight[:, idx] # [2560, 5120]
                
                # 3. Matmuls Reduzidos
                gate_out = F.linear(x, active_gate_w)
                up_out = F.linear(x, active_up_w)
                
                if self.act_fn:
                    act_out = self.act_fn(gate_out)
                else:
                    act_out = F.gelu(gate_out, approximate="tanh")
                    
                return F.linear(act_out * up_out, active_down_w)
        else:
            # Para prompt processing (seq_len > 1), o slice dinâmico é complexo
            # Fallback para forward denso, mas aplicando a máscara do router
            with torch.no_grad():
                gate_out = F.linear(x, self.gate_weight)
                up_out = F.linear(x, self.up_weight)
                
                logits = self.router(x)
                top_vals, _ = torch.topk(logits, self.k, dim=-1)
                threshold = top_vals[:, :, -1].unsqueeze(-1)
                mask = (logits >= threshold).to(gate_out.dtype)
                
                if self.act_fn:
                    act_out = self.act_fn(gate_out)
                else:
                    act_out = F.gelu(gate_out, approximate="tanh")
                
                # Aplicamos a máscara aqui para manter fidelidade
                return F.linear(act_out * up_out * mask, self.down_weight)

def benchmark_generate(model, tokenizer, prompts, max_new_tokens=50):
    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen_ids = out[0][input_len:]
        n_tok = len(gen_ids)
        tps = n_tok / elapsed if elapsed > 0 else 0
        text_out = tokenizer.decode(gen_ids, skip_special_tokens=True)

        print(f"      {tps:.1f} tok/s | {n_tok} tok | {elapsed:.2f}s")
        print(f"      → {text_out[:100]}\n")
        results.append(tps)
        
    return sum(results) / len(results)

def main():
    router_path = "checkpoints/ssn_routers.pt"
    if not os.path.exists(router_path):
        logger.error(f"Arquivo não encontrado: {router_path}. Rode o script de calibração primeiro.")
        return 1
        
    print(f"\n{'═'*60}")
    print(f"  Fase 5b: INFERÊNCIA SSN (DYNAMIC SLICING)")
    print(f"{'═'*60}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "google/gemma-4-e4b-it"

    logger.info("Carregando modelo original (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers."
    ]
    
    print(f"\n{'─'*60}")
    print(f"  BASELINE: MODELO DENSO ORIGINAL")
    print(f"{'─'*60}")
    tps_denso = benchmark_generate(model, tokenizer, prompts)

    # ════════════════════════════════════════════════════════════
    # Injetar os Routers e testar
    # ════════════════════════════════════════════════════════════
    logger.info("Carregando pesos dos roteadores SSN...")
    router_states = torch.load(router_path, weights_only=True)
    
    named_mods = dict(model.named_modules())
    layer_idx = 0
    injected = 0
    
    for name, mod in list(named_mods.items()):
        if not hasattr(mod, 'act_fn') or not hasattr(mod, 'gate_proj'):
            continue
        if "vision" in name or "audio" in name:
            continue
            
        in_dim = mod.gate_proj.in_features
        out_dim = mod.gate_proj.out_features
        
        # Instanciar e carregar os pesos previstos
        router = LowRankRouter(in_features=in_dim, out_features=out_dim, rank=128)
        router.load_state_dict(router_states[f"layer_{layer_idx}"])
        router.to(mod.gate_proj.weight.device).to(torch.bfloat16)
        
        # Envolver na nova MLP de slicing dinâmico
        ssn_mlp = SSNDynamicMLP(mod, router, keep_ratio=0.50)
        
        parent_parts = name.rsplit(".", 1)
        if len(parent_parts) == 2:
            parent = named_mods.get(parent_parts[0])
            if parent:
                setattr(parent, parent_parts[1], ssn_mlp)
                injected += 1
                
        layer_idx += 1

    logger.info(f"Injetados {injected} SSN Dynamic MLPs.")

    print(f"\n{'─'*60}")
    print(f"  ESPARSO: SSN (DYNAMIC SLICING 50%)")
    print(f"{'─'*60}")
    tps_ssn = benchmark_generate(model, tokenizer, prompts)
    
    print(f"\n{'═'*60}")
    print(f"  RESUMO")
    print(f"{'═'*60}")
    print(f"  Denso: {tps_denso:.1f} tok/s")
    print(f"  SSN 50%: {tps_ssn:.1f} tok/s (Speedup real no PyTorch por slicing dinâmico)")
    print(f"{'═'*60}\n")

if __name__ == "__main__":
    main()
