import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified.modules import Mamba2Router, DeepSliceMoE
from unified.mock_gemma4_e4b import MockGemma4E4B
from deepslice_moe_converter import DeepSliceConverter
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATASET DE TEXTO SIMPLES
# ═══════════════════════════════════════════════════════════════════════════

class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        # Ajuste para mock: garante que os indices estao no range do vocab reduzido
        if item['input_ids'].max() >= 1024:
            item['input_ids'] = item['input_ids'] % 1024
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def get_sample_data():
    return [
        "The capital of France is Paris, a city known for its art and culture.",
        "Quantum physics describes the behavior of matter and energy at atomic scales.",
        "Deep learning is a subset of machine learning based on artificial neural networks.",
        "Python is a versatile programming language used for web development and data science.",
        "The Earth revolves around the Sun in an elliptical orbit once every year.",
        "Artificial intelligence is transforming various industries and daily life activities.",
        "Sustainable energy sources like solar and wind are crucial for the future.",
        "The Great Barrier Reef is the world's largest coral reef system.",
        "Space exploration continues to push the boundaries of human knowledge.",
        "Music is a universal language that connects people across different cultures."
    ] * 10

# ═══════════════════════════════════════════════════════════════════════════
# 2. TREINADOR DO ROTEADOR
# ═══════════════════════════════════════════════════════════════════════════

class RouterTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        learning_rate: float = 1e-4, 
        weight_decay: float = 0.01,
        balance_loss_weight: float = 0.1
    ):
        self.model = model
        self.balance_loss_weight = balance_loss_weight
        
        # Congela TUDO exceto os roteadores
        num_frozen = 0
        num_trainable = 0
        trainable_params = []
        
        for name, param in model.named_parameters():
            if "router" in name:
                param.requires_grad = True
                trainable_params.append(param)
                num_trainable += 1
            else:
                param.requires_grad = False
                num_frozen += 1
        
        logger.info(f"Parametros congelados: {num_frozen} | Parametros treinaveis (Router): {num_trainable}")
        
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        
        # Hooks para coletar logits dos roteadores para Balance Loss
        self.collected_logits = []
        for module in model.modules():
            if isinstance(module, Mamba2Router):
                module.register_forward_hook(self._collect_logits_hook)

    def _collect_logits_hook(self, module, input, output):
        # output: router_logits (B, T, num_experts)
        self.collected_logits.append(output.view(-1, output.size(-1)))

    def compute_loss(self, outputs, labels):
        # LM Loss (Cross Entropy)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Load Balancing Loss (Auxiliary Loss)
        balance_loss = 0
        if self.collected_logits:
            all_logits = torch.cat(self.collected_logits, dim=0) # (TotalTokens * Layers, num_experts)
            probs = torch.softmax(all_logits, dim=-1)
            mean_probs = probs.mean(dim=0)
            balance_loss = torch.var(mean_probs) * all_logits.size(-1)
            self.collected_logits = [] # Limpa para o proximo passo
            
        return lm_loss + (self.balance_loss_weight * balance_loss)

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        self.collected_logits = [] # Garante limpeza
        
        device = next(self.model.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.compute_loss(outputs, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# ═══════════════════════════════════════════════════════════════════════════
# 3. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fase 8: Router Fine-Tuning")
    parser.add_argument("--model-id", default="google/gemma-4-e4b-it")
    parser.add_argument("--mock", action="store_true", help="Usa mock model para testes")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0, help="Limite de passos por epoca (0 = todos)")
    parser.add_argument("--patience", type=int, default=5, help="Epocas para esperar antes de Early Stopping")
    parser.add_argument("--target-loss", type=float, default=1.5, help="Para se atingir essa loss")
    parser.add_argument("--save-path", default="checkpoints/router_trained")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando device: {device}")

    # 1. Carregar Modelo e Tokenizer
    if args.mock:
        logger.info("Carregando MockGemma4E4B...")
        model = MockGemma4E4B(lite=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained("gpt2") # Tokenizer nao-gated para mock
        tokenizer.pad_token = tokenizer.eos_token
    else:
        logger.info(f"Carregando {args.model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # 2. Converter para MoE (se ainda nao for)
    # Verificamos se ja tem DeepSliceMoE
    is_moe = any(isinstance(m, DeepSliceMoE) for m in model.modules())
    if not is_moe:
        logger.info("Convertendo modelo para DeepSliceMoE antes do treino...")
        converter = DeepSliceConverter(
            model=model,
            shared_ratio=0.5,
            num_routed_experts=8,
            num_experts_per_tok=2
        )
        model = converter.convert()

    # 3. Preparar Dataset
    logger.info("Preparando dados...")
    texts = get_sample_data()
    dataset = SimpleTextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 4. Inicializar Treinador
    trainer = RouterTrainer(model, learning_rate=args.lr)

    # 5. Configurações de Memória para evitar OOM
    if not args.mock:
        logger.info("Ativando Gradient Checkpointing para economizar VRAM...")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False # Necessario para gradient checkpointing
        
    # 5. Loop de Treino com Early Stopping e Convergência Inteligente
    logger.info(f"Iniciando Fine-Tuning do Roteador (Ate {args.epochs} epocas)...")
    model.train()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            loss = trainer.train_step(batch)
            epoch_loss += loss
            pbar.set_postfix({"loss": f"{loss:.4f}", "best": f"{best_loss:.4f}"})
            
            if args.steps > 0 and step >= args.steps - 1:
                break
        
        avg_loss = epoch_loss / (step + 1)
        logger.info(f"Epoch {epoch+1} concluida. Loss media: {avg_loss:.4f}")

        # Lógica de Convergência Inteligente
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Salva o "Melhor Modelo"
            os.makedirs(args.save_path, exist_ok=True)
            router_state = {name: param for name, param in model.named_parameters() if "router" in name}
            torch.save(router_state, os.path.join(args.save_path, "router_best.pt"))
            logger.info(f"--- Melhor modelo salvo (loss: {best_loss:.4f}) ---")
        else:
            patience_counter += 1
            logger.info(f"Sem melhora por {patience_counter} epocas.")

        if avg_loss <= args.target_loss:
            logger.info(f"--- ALVO ATINGIDO: Loss {avg_loss:.4f} <= {args.target_loss} ---")
            break

        if patience_counter >= args.patience:
            logger.info(f"--- EARLY STOPPING: O modelo parou de aprender (convergência atingida) ---")
            break

    # 6. Salvar Pesos Finais
    router_state = {name: param for name, param in model.named_parameters() if "router" in name}
    torch.save(router_state, os.path.join(args.save_path, "router_final.pt"))
    logger.info(f"Pesos finais salvos em: {args.save_path}")

if __name__ == "__main__":
    main()
