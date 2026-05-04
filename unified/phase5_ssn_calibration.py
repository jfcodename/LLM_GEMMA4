"""
Gemma 4 E4B — Fase 5: SSN Low-Rank Router Calibration
=====================================================
Em vez de depender de heurísticas ou do `per_layer_input_gate` original 
(que falhou por ter granularidade na camada toda, não em neurônios individuais),
aqui nós treinamos ativamente um Roteador Dinâmico de baixo custo (Low-Rank)
para prever quais neurônios da MLP devem ser ativados a cada token.

Arquitetura do Router:
    Input: 2560 (hidden_states)
    Hidden: 128 (gargalo de compressão)
    Output: 10240 (logits para a máscara Top-K)
    Custo: ~1.6M parâmetros por camada (vs 26.2M do gate_proj original)

Uso no Kaggle:
    %cd /kaggle/working/LLM_GEMMA4
    !python unified/phase5_ssn_calibration.py
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


class LowRankRouter(nn.Module):
    """Preditor de esparsidade super leve para a MLP."""
    def __init__(self, in_features=2560, out_features=10240, rank=128):
        super().__init__()
        # Gargalo de baixo posto para economizar FLOPs
        self.down = nn.Linear(in_features, rank, bias=False)
        self.act = nn.ReLU()
        self.up = nn.Linear(rank, out_features, bias=False)
        
        # Inicialização: queremos que comece neutro
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class SSNCalibrationWrapper(nn.Module):
    """
    Hook que envolve a MLP original. No forward:
    1. Calcula a MLP normal (gate_proj).
    2. Identifica os 50% maiores valores (Target Mask).
    3. Roda o LowRankRouter.
    4. Salva a perda (BCE) entre o Router e a Máscara ideal para treinar o router.
    """
    def __init__(self, original_mlp, keep_ratio=0.50):
        super().__init__()
        self.original_mlp = original_mlp
        self.keep_ratio = keep_ratio
        
        in_dim = original_mlp.gate_proj.in_features
        out_dim = original_mlp.gate_proj.out_features
        
        self.router = LowRankRouter(in_features=in_dim, out_features=out_dim, rank=128)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.current_loss = torch.tensor(0.0, device=original_mlp.gate_proj.weight.device)
        self.samples = 0

    def forward(self, x):
        # 1. Forward da MLP Original (como Professor)
        with torch.no_grad():
            gate_out = self.original_mlp.gate_proj(x)
            
            # Encontrar o threshold Top-K ideal
            k = int(gate_out.shape[-1] * self.keep_ratio)
            # Flatten batch e seq para calcular topk por token
            flat_gate = gate_out.view(-1, gate_out.shape[-1])
            
            # Máscara ideal: 1.0 para os Top-K, 0.0 para o resto
            top_vals, _ = torch.topk(flat_gate.abs(), k, dim=-1)
            threshold = top_vals[:, -1].unsqueeze(-1)
            target_mask = (flat_gate.abs() >= threshold).float()
            
            # Se for fase de inferência sem professor, usaríamos só o router aqui.
        
        # 2. Forward do Router (Estudante)
        # O x de entrada tem gradiente habilitado para o router? Sim, queremos treinar o router.
        # Desanexamos o x para não propagar gradientes para o resto do modelo de base.
        router_logits = self.router(x.detach())
        flat_logits = router_logits.view(-1, router_logits.shape[-1])
        
        # 3. Calcular Perda
        loss = self.loss_fn(flat_logits, target_mask)
        self.current_loss += loss
        self.samples += 1
        
        # O forward continua normal para o resto da rede não quebrar
        # A MLP em si NÃO será esparsa durante o treinamento do router, 
        # para não acumular erros nas camadas de cima.
        up_out = self.original_mlp.up_proj(x)
        
        if hasattr(self.original_mlp, 'act_fn'):
            act_out = self.original_mlp.act_fn(gate_out)
        else:
            act_out = F.gelu(gate_out, approximate="tanh")
            
        return self.original_mlp.down_proj(act_out * up_out)

    def get_avg_loss_and_reset(self):
        if self.samples == 0:
            return 0.0
        avg = (self.current_loss / self.samples).item()
        self.current_loss.zero_()
        self.samples = 0
        return avg


def main():
    if not torch.cuda.is_available():
        logger.error("Requer GPU (Kaggle T4) para calibrar os roteadores.")
        return 1

    print(f"\n{'═'*60}")
    print(f"  Fase 5: CALIBRAÇÃO DO SSN LOW-RANK ROUTER")
    print(f"{'═'*60}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    model_id = "google/gemma-4-e4b-it"
    logger.info(f"Carregando {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Congelar o modelo inteiro (não vamos treinar a LLM, só os routers)
    for param in model.parameters():
        param.requires_grad = False

    # Injetar os Wrappers do Router nas MLPs
    wrappers = []
    named_mods = dict(model.named_modules())
    for name, mod in list(named_mods.items()):
        if not hasattr(mod, 'act_fn') or not hasattr(mod, 'gate_proj'):
            continue
        if "vision" in name or "audio" in name:
            continue
            
        wrapper = SSNCalibrationWrapper(mod, keep_ratio=0.50).to(mod.gate_proj.weight.device)
        wrapper.router.to(torch.bfloat16)  # Treinar em bf16 pra economizar vram
        
        parent_parts = name.rsplit(".", 1)
        if len(parent_parts) == 2:
            parent = named_mods.get(parent_parts[0])
            if parent:
                setattr(parent, parent_parts[1], wrapper)
                wrappers.append(wrapper)

    logger.info(f"Injetados {len(wrappers)} Low-Rank Routers nas MLPs.")

    # Preparar Otimizador apenas para os Routers
    router_params = []
    for w in wrappers:
        # Habilitar gradientes só nos routers
        for p in w.router.parameters():
            p.requires_grad = True
            router_params.append(p)
            
    optimizer = torch.optim.Adam(router_params, lr=1e-3)

    # Preparar Dataset de Calibração
    logger.info("Carregando WikiText-2 para calibração...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")
    
    texts = [t for t in dataset["text"] if len(t.strip()) > 50]
    
    def collate_fn(batch):
        return tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
    tokenizer.pad_token = tokenizer.eos_token
    dataloader = DataLoader(texts[:100], batch_size=2, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'─'*60}")
    print(f"  TREINAMENTO DOS ROUTERS (Destilação de Máscara)")
    print(f"{'─'*60}")

    model.train() # Routers no modo de treino
    epochs = 2
    
    for epoch in range(epochs):
        t0 = time.time()
        for step, batch in enumerate(dataloader):
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # O forward triggera a acumulação de loss dentro dos wrappers
            _ = model(**inputs)
            
            # Somar a loss de todos os layers resolvendo possível espalhamento em multi-GPU (device_map)
            total_loss = torch.tensor(0.0, device=model.device)
            for w in wrappers:
                # Move a loss local daquela camada para o device principal antes de somar
                total_loss += w.current_loss.to(model.device)
                
            total_loss.backward()
            optimizer.step()
            
            # Print de progresso e reseta losses
            if step % 10 == 0:
                avg_losses = [w.get_avg_loss_and_reset() for w in wrappers]
                global_avg = sum(avg_losses) / len(avg_losses)
                print(f"  Epoch {epoch+1}/{epochs} | Step {step:2d} | Avg BCE Loss: {global_avg:.4f}")
            else:
                for w in wrappers:
                    w.get_avg_loss_and_reset()
                    
        print(f"  > Epoch {epoch+1} concluída em {time.time() - t0:.1f}s")

    print(f"\n{'─'*60}")
    print(f"  CALIBRAÇÃO CONCLUÍDA")
    print(f"{'─'*60}")
    print("  Os Low-Rank Routers agora sabem quais neurônios ativar!")
    
    # Salvar os pesos dos routers (Opcional, implementaremos no próximo passo)
    os.makedirs("checkpoints", exist_ok=True)
    router_state_dicts = {f"layer_{i}": w.router.state_dict() for i, w in enumerate(wrappers)}
    torch.save(router_state_dicts, "checkpoints/ssn_routers.pt")
    logger.info("Pesos dos roteadores salvos em checkpoints/ssn_routers.pt")

if __name__ == "__main__":
    main()
