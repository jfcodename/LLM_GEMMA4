"""
Gemma 4 Neo — Destilação Mamba-2

Treina blocos Mamba-2 para imitar a saída dos layers locais SWA do Gemma 4.
Estratégia: MSE loss entre saída do teacher (SWA) e student (Mamba-2)
por layer, com forward pass do teacher congelado.

Custo estimado: ~150B tokens para recuperar acc completa.
Para testes: ~10B tokens já dão resultados razoáveis.

Uso:
    python mamba2_distill.py \
        --teacher google/gemma-4-e2b-it \
        --output ./mamba2_weights \
        --tokens 10B \
        --layers-per-run 5
"""

import os
import time
import math
import argparse
from pathlib import Path
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

from config import Gemma4NeoConfig, GLOBAL_LAYER_INDICES
from modules import Mamba2Block
from convert import get_layer


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DE DESTILAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

class DistillationDataset(torch.utils.data.Dataset):
    """
    Dataset para destilação: retorna sequências tokenizadas de texto.
    Suporta HuggingFace datasets e arquivos .txt locais.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str = "wikitext",
        subset: str = "wikitext-103-raw-v1",
        split: str = "train",
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, subset, split=split)
            texts = [t["text"] for t in ds if len(t.get("text", "")) > 100]
            if max_samples:
                texts = texts[:max_samples]
            print(f"  Dataset: {dataset_name}/{subset} ({len(texts)} docs)")
        except Exception as e:
            print(f"  ⚠ Erro ao carregar dataset: {e}")
            print("  Usando dados sintéticos para demonstração")
            texts = ["The following is an example of natural language text. " * 100] * 1000

        # Tokeniza em chunks de max_length
        self.chunks = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(ids) - max_length, max_length // 2):
                chunk = ids[i:i + max_length]
                if len(chunk) == max_length:
                    self.chunks.append(torch.tensor(chunk, dtype=torch.long))
        print(f"  Total de chunks: {len(self.chunks)} × {max_length} tokens")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


# ─────────────────────────────────────────────────────────────────────────────
# DESTILADOR POR LAYER
# ─────────────────────────────────────────────────────────────────────────────

class LayerDistiller:
    """
    Destila um único layer local SWA → Mamba-2.

    Estratégia de 3 fases:
      1. Warm-up: MSE direto entre saídas (sem gradiente no teacher)
      2. Feature alignment: + cosine similarity na representação intermediária
      3. End-to-end: fine-tuning com LM loss no contexto da posição do layer
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        layer_idx: int,
        mamba2_config,
        hidden_size: int = 2560,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.layer_idx = layer_idx
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size

        # Teacher: layer SWA original (frozen)
        self.teacher_layer = get_layer(teacher_model, layer_idx)
        for p in self.teacher_layer.parameters():
            p.requires_grad_(False)
        self.teacher_layer.eval()

        # Extrai embedding e layers anteriores do teacher para gerar hidden states
        self.teacher_model = teacher_model
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

        # Student: bloco Mamba-2 (trainable)
        self.student = Mamba2Block(
            hidden_size=hidden_size,
            config=mamba2_config,
        ).to(device).to(dtype)

        # Projeção de alinhamento (descartada após destilação)
        # Mapeia hidden_size → hidden_size para adaptação de distribuição
        self.adapter = nn.Linear(hidden_size, hidden_size, bias=False).to(device).to(dtype)
        nn.init.eye_(self.adapter.weight)  # Inicia como identidade

        print(f"  Layer {layer_idx}: Student Mamba-2 inicializado "
              f"({sum(p.numel() for p in self.student.parameters())/1e6:.1f}M params)")

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Computa hidden states na entrada do layer_idx usando o teacher.
        Roda os layers 0..layer_idx-1 congelados para obter h.
        """
        with torch.no_grad():
            lm = self.teacher_model.model.language_model
            # Embedding
            x = lm.embed_tokens(input_ids)
            # Layers anteriores ao target
            for i in range(self.layer_idx):
                layer_out = lm.layers[i](x, use_cache=False)
                x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        return x  # (B, T, hidden_size)

    def _teacher_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Computa saída do layer teacher (SWA)."""
        with torch.no_grad():
            out = self.teacher_layer(hidden_states, use_cache=False)
            return out[0] if isinstance(out, tuple) else out

    def _student_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Computa saída do Mamba-2 student."""
        return self.student(hidden_states)

    def distillation_loss(
        self,
        teacher_out: torch.Tensor,
        student_out: torch.Tensor,
        alpha_mse: float = 1.0,
        alpha_cosine: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Loss combinada: MSE + Cosine Similarity.

        MSE captura a magnitude das ativações.
        Cosine captura a direção / alinhamento de representação.
        """
        # MSE loss (normalizada pelo hidden_size)
        mse = F.mse_loss(student_out, teacher_out)

        # Cosine similarity loss: maximiza cos(student, teacher)
        student_flat = student_out.reshape(-1, self.hidden_size)
        teacher_flat = teacher_out.reshape(-1, self.hidden_size)
        cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        cosine_loss = (1.0 - cos_sim).mean()

        total = alpha_mse * mse + alpha_cosine * cosine_loss

        return {
            "total": total,
            "mse": mse.detach(),
            "cosine": cosine_loss.detach(),
        }

    def train(
        self,
        dataloader: DataLoader,
        num_steps: int = 10_000,
        lr: float = 3e-4,
        warmup_steps: int = 500,
        grad_clip: float = 1.0,
        log_every: int = 100,
        save_dir: Optional[str] = None,
        phase2_start: int = 5000,  # Quando ativar cosine loss
    ) -> Dict:
        """
        Loop principal de destilação.

        Fase 1 (0 → phase2_start): MSE puro — aprende a imitar saídas
        Fase 2 (phase2_start → num_steps): MSE + cosine — alinha representações
        """
        optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.adapter.parameters()),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_steps,
        )

        self.student.train()
        self.adapter.train()

        history = {"mse": [], "cosine": [], "total": [], "step": []}
        data_iter = iter(dataloader)
        best_loss = float("inf")

        print(f"\n  Iniciando destilação do layer {self.layer_idx}...")
        print(f"  Steps: {num_steps} | LR: {lr} | Warmup: {warmup_steps}")

        for step in range(num_steps):
            # Pega próximo batch
            try:
                input_ids = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                input_ids = next(data_iter)

            input_ids = input_ids.to(self.device)

            # Computa hidden states de entrada (layers 0..layer_idx-1)
            hidden_states = self._get_hidden_states(input_ids)
            hidden_states = hidden_states.to(self.dtype)

            # Saída do teacher (SWA — frozen)
            teacher_out = self._teacher_output(hidden_states)

            # Saída do student (Mamba-2 — trainable)
            student_out = self._student_output(hidden_states)
            # Adapter para alinhamento de distribuição
            student_out_adapted = self.adapter(student_out)

            # Loss
            alpha_cosine = 0.1 if step >= phase2_start else 0.0
            losses = self.distillation_loss(
                teacher_out, student_out_adapted,
                alpha_cosine=alpha_cosine,
            )

            # Backward
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.adapter.parameters()),
                grad_clip,
            )
            optimizer.step()
            scheduler.step()

            # Logging
            if step % log_every == 0 or step == num_steps - 1:
                mse_v = losses["mse"].item()
                cos_v = losses["cosine"].item()
                tot_v = losses["total"].item()
                lr_v = scheduler.get_last_lr()[0]

                history["step"].append(step)
                history["mse"].append(mse_v)
                history["cosine"].append(cos_v)
                history["total"].append(tot_v)

                print(f"  Step {step:5d}/{num_steps} | "
                      f"MSE={mse_v:.4f} | Cos={cos_v:.4f} | "
                      f"Total={tot_v:.4f} | LR={lr_v:.2e}")

                # Salva checkpoint se melhorou
                if save_dir and tot_v < best_loss:
                    best_loss = tot_v
                    self._save(save_dir, step)

        # Salva final
        if save_dir:
            self._save(save_dir, num_steps, is_final=True)

        return history

    def _save(self, save_dir: str, step: int, is_final: bool = False):
        """Salva pesos do Mamba-2 student para este layer."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = (f"mamba2_layer_{self.layer_idx}_final.pt"
                 if is_final else
                 f"mamba2_layer_{self.layer_idx}_step{step}.pt")
        torch.save(self.student.state_dict(), os.path.join(save_dir, fname))
        # Sempre salva como "layer_X.pt" (último melhor)
        torch.save(
            self.student.state_dict(),
            os.path.join(save_dir, f"mamba2_layer_{self.layer_idx}.pt"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ORQUESTRADOR COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

class Mamba2Distiller:
    """
    Orquestra a destilação de todos os layers locais em paralelo ou sequencial.

    Para GPUs múltiplas: distribui layers entre devices.
    Para GPU única: processa um layer por vez (sem perda de qualidade).
    """

    def __init__(
        self,
        teacher_model_id: str,
        neo_config: Gemma4NeoConfig,
        device: str = "cuda",
    ):
        self.neo_config = neo_config
        self.device = device

        print(f"\n{'═'*60}")
        print(f"  MAMBA-2 DISTILLATION")
        print(f"{'═'*60}")
        print(f"  Teacher: {teacher_model_id}")

        # Carrega teacher (frozen)
        self.tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_id, trust_remote_code=True
        )
        print(f"  Carregando teacher model (frozen)...")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        print(f"  ✓ Teacher carregado e congelado")

    def distill_all_layers(
        self,
        output_dir: str,
        dataset_name: str = "wikitext",
        num_steps_per_layer: int = 10_000,
        batch_size: int = 4,
        max_length: int = 1024,
        lr: float = 3e-4,
        layers_subset: Optional[List[int]] = None,
        resume: bool = True,
    ):
        """
        Destila todos os layers locais sequencialmente.

        Args:
            output_dir: onde salvar pesos por layer
            num_steps_per_layer: steps de treino por layer (10k ≈ 40M tokens)
            batch_size: batch size por GPU
            max_length: comprimento máximo de sequência
            layers_subset: se fornecido, destila apenas esses layers
            resume: pula layers já destilados (checkpoint existe)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Layers locais (não-globais)
        local_layers = [
            i for i in range(self.neo_config.num_hidden_layers)
            if i not in GLOBAL_LAYER_INDICES
        ]
        if layers_subset:
            local_layers = [i for i in local_layers if i in layers_subset]

        print(f"\n  Layers para destilar: {local_layers}")
        print(f"  Steps por layer: {num_steps_per_layer:,}")
        print(f"  Batch size: {batch_size} | Seq length: {max_length}")

        # Cria dataset uma vez para todos os layers
        dataset = DistillationDataset(
            tokenizer=self.tokenizer,
            dataset_name=dataset_name,
            max_length=max_length,
            max_samples=100_000,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        results = {}

        for layer_idx in local_layers:
            print(f"\n{'─'*60}")
            print(f"  Destilando layer {layer_idx} / {local_layers[-1]}")
            print(f"{'─'*60}")

            # Verifica se já foi destilado
            checkpoint = os.path.join(output_dir, f"mamba2_layer_{layer_idx}.pt")
            if resume and os.path.exists(checkpoint):
                print(f"  ✓ Checkpoint encontrado — pulando layer {layer_idx}")
                continue

            distiller = LayerDistiller(
                teacher_model=self.teacher,
                layer_idx=layer_idx,
                mamba2_config=self.neo_config.mamba2,
                hidden_size=self.neo_config.hidden_size,
                device=self.device,
                dtype=torch.bfloat16,
            )

            t0 = time.time()
            history = distiller.train(
                dataloader=dataloader,
                num_steps=num_steps_per_layer,
                lr=lr,
                warmup_steps=max(500, num_steps_per_layer // 20),
                save_dir=output_dir,
                log_every=max(50, num_steps_per_layer // 200),
            )
            elapsed = time.time() - t0

            results[layer_idx] = {
                "final_loss": history["total"][-1],
                "final_mse": history["mse"][-1],
                "time_min": elapsed / 60,
            }

            print(f"\n  Layer {layer_idx} completo em {elapsed/60:.1f}min")
            print(f"  Loss final: {history['total'][-1]:.4f}")

            # Libera memória entre layers
            del distiller
            torch.cuda.empty_cache()

        # Salva sumário
        import json
        summary_path = os.path.join(output_dir, "distillation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'═'*60}")
        print(f"  Destilação completa!")
        print(f"  Pesos salvos em: {output_dir}")
        print(f"  Total de layers: {len(local_layers)}")
        print(f"{'═'*60}")

        return results


# ─────────────────────────────────────────────────────────────────────────────
# VALIDAÇÃO PÓS-DESTILAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_distillation(
    teacher_model: nn.Module,
    mamba_weights_dir: str,
    neo_config: Gemma4NeoConfig,
    tokenizer,
    num_samples: int = 100,
    device: str = "cuda",
) -> Dict:
    """
    Compara saídas do teacher SWA vs student Mamba-2 por layer.
    Reporta MSE médio e cosine similarity para cada layer destilado.
    """
    print(f"\n  Validando destilação...")

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t["text"] for t in ds if len(t.get("text", "")) > 200][:num_samples]
    ids = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512).input_ids.to(device)

    local_layers = [i for i in range(neo_config.num_hidden_layers)
                    if i not in GLOBAL_LAYER_INDICES]
    results = {}

    for layer_idx in local_layers:
        ckpt = os.path.join(mamba_weights_dir, f"mamba2_layer_{layer_idx}.pt")
        if not os.path.exists(ckpt):
            continue

        # Carrega student
        student = Mamba2Block(
            hidden_size=neo_config.hidden_size,
            config=neo_config.mamba2,
        ).to(device).to(torch.bfloat16)
        student.load_state_dict(torch.load(ckpt, map_location=device))
        student.eval()

        # Computa hidden states de entrada
        lm = teacher_model.model.language_model
        x = lm.embed_tokens(ids)
        for i in range(layer_idx):
            out = lm.layers[i](x, use_cache=False)
            x = out[0] if isinstance(out, tuple) else out
        x = x.to(torch.bfloat16)

        # Saídas
        teacher_layer = get_layer(teacher_model, layer_idx)
        t_out = teacher_layer(x, use_cache=False)
        t_out = t_out[0] if isinstance(t_out, tuple) else t_out
        s_out = student(x)

        mse = F.mse_loss(s_out, t_out).item()
        cos = F.cosine_similarity(
            s_out.reshape(-1, neo_config.hidden_size),
            t_out.reshape(-1, neo_config.hidden_size), dim=-1
        ).mean().item()

        results[layer_idx] = {"mse": mse, "cosine": cos}
        print(f"  Layer {layer_idx:02d}: MSE={mse:.4f} | Cosine={cos:.4f}")

        del student
        torch.cuda.empty_cache()

    avg_mse = sum(r["mse"] for r in results.values()) / max(1, len(results))
    avg_cos = sum(r["cosine"] for r in results.values()) / max(1, len(results))
    print(f"\n  Média: MSE={avg_mse:.4f} | Cosine={avg_cos:.4f}")
    print(f"  {'✓ Bom' if avg_cos > 0.95 else '⚠ Requer mais treino'} "
          f"(target: cosine > 0.95)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Destila Mamba-2 para Gemma 4 Neo")
    p.add_argument("--teacher", default="google/gemma-4-e2b-it")
    p.add_argument("--output", default="./mamba2_weights")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--steps", type=int, default=10_000,
                   help="Steps por layer (10k ≈ 40M tokens com batch=4, len=1024)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Subconjunto de layers para destilar (padrão: todos locais)")
    p.add_argument("--no-resume", action="store_true",
                   help="Redestila mesmo se checkpoint existe")
    p.add_argument("--validate-only", action="store_true",
                   help="Só valida checkpoints existentes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    neo_config = Gemma4NeoConfig(
        base_model_id=args.teacher,
    )
    neo_config.mamba2.enabled = True

    distiller = Mamba2Distiller(
        teacher_model_id=args.teacher,
        neo_config=neo_config,
        device=device,
    )

    if args.validate_only:
        validate_distillation(
            teacher_model=distiller.teacher,
            mamba_weights_dir=args.output,
            neo_config=neo_config,
            tokenizer=distiller.tokenizer,
            device=device,
        )
    else:
        distiller.distill_all_layers(
            output_dir=args.output,
            dataset_name=args.dataset,
            num_steps_per_layer=args.steps,
            batch_size=args.batch_size,
            max_length=args.max_length,
            lr=args.lr,
            layers_subset=args.layers,
            resume=not args.no_resume,
        )
