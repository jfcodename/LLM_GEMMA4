"""
Gemma 4 Neo — Inferência Otimizada

Carrega o modelo Neo convertido e executa inferência com todas as otimizações
ativas: sparsidade MLP, SnapKV, quantização, e opcionalmente speculative decoding.

Uso:
    python inference.py --model ./gemma4_neo_checkpoint --prompt "Explique..."
    python inference.py --model ./gemma4_neo_checkpoint --benchmark
    python inference.py --model ./gemma4_neo_checkpoint --speculative --draft e2b
"""

import os
import time
import json
import argparse
from typing import Optional, List, Generator

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from config import Gemma4NeoConfig, GLOBAL_LAYER_INDICES
from modules import SnapKVCache


# ─────────────────────────────────────────────────────────────────────────────
# CARREGADOR DO MODELO NEO
# ─────────────────────────────────────────────────────────────────────────────

class Gemma4NeoInference:
    """
    Interface de inferência para o Gemma 4 Neo.
    Gerencia o modelo, tokenizer e otimizações de runtime.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        compile_model: bool = False,
    ):
        self.model_dir = model_dir
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch_dtype

        # Carrega manifest de modificações
        manifest_path = os.path.join(model_dir, "neo_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            print(f"  Modelo Neo — Modificações: {self.manifest.get('applied_steps', [])}")
        else:
            self.manifest = {}
            print("  ⚠ Manifest não encontrado. Carregando como modelo padrão.")

        print(f"  Carregando de: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Compila com torch.compile para máximo throughput (requer PyTorch 2.x)
        if compile_model and hasattr(torch, "compile"):
            print("  Compilando modelo com torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("  ✓ Compilação concluída")

        print(f"  ✓ Modelo pronto no device: {self.device}")

        # Estado do SnapKV (reset entre gerações)
        self._snap_kv_caches: List[SnapKVCache] = []
        self._reset_kv_caches()

    def _reset_kv_caches(self):
        """Reseta SnapKV caches dos layers globais."""
        self._snap_kv_caches = []
        # Tenta acessar os caches existentes nos layers globais
        try:
            for layer_idx in sorted(GLOBAL_LAYER_INDICES):
                lm = self.model.model.language_model
                layer = lm.layers[layer_idx]
                if hasattr(layer, "_gated_attn") and layer._gated_attn.snap_kv is not None:
                    layer._gated_attn.snap_kv.reset()
                    self._snap_kv_caches.append(layer._gated_attn.snap_kv)
        except (AttributeError, IndexError):
            pass

    # ── Inferência básica ─────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = False,
    ) -> str:
        """
        Gera texto a partir de um prompt.

        Args:
            prompt: texto de entrada
            max_new_tokens: máximo de tokens a gerar
            temperature: temperatura de sampling (0 = greedy)
            stream: se True, faz streaming token a token
        Returns:
            texto gerado
        """
        self._reset_kv_caches()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if stream:
            streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)
        elapsed = time.perf_counter() - t0

        new_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
        tokens_per_sec = new_tokens / elapsed

        generated = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        self._last_stats = {
            "tokens_generated": new_tokens,
            "time_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
        }

        return generated

    def generate_with_stats(self, prompt: str, **kwargs) -> dict:
        """Gera texto e retorna estatísticas detalhadas de desempenho."""
        text = self.generate(prompt, **kwargs)
        stats = getattr(self, "_last_stats", {})

        # Coleta estatísticas de sparsidade MLP
        sparsities = self._collect_sparsity_stats()

        return {
            "text": text,
            "performance": stats,
            "sparsity": sparsities,
        }

    def _collect_sparsity_stats(self) -> dict:
        """Coleta sparsidade atual do SparsityPredictor por layer."""
        stats = {}
        try:
            for i in range(42):
                layer = self.model.model.language_model.layers[i]
                if hasattr(layer, "sparsity_predictor"):
                    stats[i] = layer.sparsity_predictor.actual_sparsity
        except (AttributeError, IndexError):
            pass
        if stats:
            stats["mean"] = sum(stats.values()) / len(stats)
        return stats

    # ── Speculative Decoding ──────────────────────────────────────────────────

    @torch.inference_mode()
    def generate_speculative(
        self,
        prompt: str,
        draft_model: "Gemma4NeoInference",
        max_new_tokens: int = 512,
        num_draft: int = 5,
        temperature: float = 0.7,
    ) -> tuple:
        """
        Speculative decoding: draft (E2B) propõe, verifier (E4B/27B) valida.

        Implementação simplificada do algoritmo original (Leviathan et al., 2023).
        Para produção, usar vllm com speculative_config.

        Returns:
            (texto_gerado, stats_dict)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]

        generated = input_ids.clone()
        total_accepted = 0
        total_proposed = 0
        t0 = time.perf_counter()

        while generated.shape[1] - prompt_len < max_new_tokens:
            # ── Draft: propõe num_draft tokens ───────────────────────────────
            draft_ids = generated.clone()
            draft_probs = []

            for _ in range(num_draft):
                with torch.no_grad():
                    draft_out = draft_model.model(draft_ids)
                    logits = draft_out.logits[:, -1, :]
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)
                    draft_probs.append(probs[0, next_tok[0, 0]].item())
                    draft_ids = torch.cat([draft_ids, next_tok], dim=1)

            proposed_tokens = draft_ids[:, generated.shape[1]:]  # (1, num_draft)

            # ── Verifier: valida todos em paralelo ────────────────────────────
            # Um único forward pass do verifier para num_draft+1 posições
            with torch.no_grad():
                verify_input = torch.cat([generated, proposed_tokens], dim=1)
                verify_out = self.model(verify_input)
                verify_logits = verify_out.logits  # (1, T, vocab)

            # Verifica cada token proposto
            n_accepted = 0
            for i, draft_tok in enumerate(proposed_tokens[0]):
                pos = generated.shape[1] + i - 1  # posição no verify_logits
                v_probs = torch.softmax(
                    verify_logits[:, pos, :] / temperature, dim=-1
                )
                d_prob = draft_probs[i]
                v_prob = v_probs[0, draft_tok].item()

                # Aceita com probabilidade min(1, p_verifier / p_draft)
                acceptance = min(1.0, v_prob / max(d_prob, 1e-10))
                if torch.rand(1).item() < acceptance:
                    generated = torch.cat(
                        [generated, draft_tok.view(1, 1)], dim=1
                    )
                    n_accepted += 1
                    total_accepted += 1
                else:
                    # Rejeita: amostria do resíduo e para
                    residual = torch.clamp(v_probs - d_prob, min=0)
                    residual = residual / residual.sum()
                    new_tok = torch.multinomial(residual, num_samples=1)
                    generated = torch.cat([generated, new_tok], dim=1)
                    break

            total_proposed += num_draft

            # Verifica EOS
            if generated[0, -1].item() == self.tokenizer.eos_token_id:
                break

        elapsed = time.perf_counter() - t0
        new_tokens = generated.shape[1] - prompt_len
        acceptance_rate = total_accepted / max(total_proposed, 1)

        text = self.tokenizer.decode(
            generated[0, prompt_len:], skip_special_tokens=True
        )
        stats = {
            "tokens_generated": new_tokens,
            "time_sec": elapsed,
            "tokens_per_sec": new_tokens / elapsed,
            "acceptance_rate": acceptance_rate,
            "speedup_vs_greedy": acceptance_rate * num_draft + 1,
        }

        return text, stats

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 200,
        n_warmup: int = 2,
        n_runs: int = 5,
    ) -> dict:
        """
        Benchmark de latência e throughput.
        Compara prefill vs decode e reporta tokens/s.
        """
        if prompts is None:
            prompts = [
                "Explain quantum entanglement in simple terms.",
                "Write a Python function that sorts a list of dictionaries by multiple keys.",
                "What are the main differences between supervised and unsupervised learning?",
            ]

        print(f"\n{'═'*60}")
        print(f"  BENCHMARK — Gemma 4 Neo")
        print(f"{'═'*60}")
        print(f"  Device: {self.device}")
        print(f"  Max tokens: {max_new_tokens}")
        print(f"  Runs: {n_runs} (+ {n_warmup} warmup)\n")

        # Warmup
        for _ in range(n_warmup):
            self.generate(prompts[0], max_new_tokens=50, stream=False)

        # Benchmark runs
        all_tps = []
        results = {}

        for prompt in prompts:
            run_tps = []
            for run in range(n_runs):
                self._reset_kv_caches()
                t0 = time.perf_counter()
                result = self.generate_with_stats(
                    prompt, max_new_tokens=max_new_tokens, temperature=0
                )
                elapsed = time.perf_counter() - t0
                tps = result["performance"].get("tokens_per_sec", 0)
                run_tps.append(tps)

            avg_tps = sum(run_tps) / len(run_tps)
            all_tps.extend(run_tps)
            short_prompt = prompt[:50] + "..."
            print(f"  '{short_prompt}'")
            print(f"    Avg: {avg_tps:.1f} tok/s | "
                  f"Min: {min(run_tps):.1f} | Max: {max(run_tps):.1f}")

            # Sparsidade média
            sp = result["sparsity"].get("mean", None)
            if sp:
                print(f"    MLP sparsidade: {sp:.1%}")

            results[prompt] = {"avg_tps": avg_tps, "runs": run_tps}

        overall_avg = sum(all_tps) / len(all_tps)
        print(f"\n  Throughput médio geral: {overall_avg:.1f} tokens/s")
        print(f"{'═'*60}")

        return results


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS DE SPARSIDADE EM TEMPO REAL
# ─────────────────────────────────────────────────────────────────────────────

class SparsityMonitor:
    """
    Hook que monitora a esparsidade real das ativações MLP durante inferência.
    Útil para calibrar e validar o SparsityPredictor.
    """

    def __init__(self, model: nn.Module, n_layers: int = 42):
        self.hooks = []
        self.activation_sparsities = {}
        self.n_layers = n_layers

        for i in range(n_layers):
            try:
                layer = model.model.language_model.layers[i]
                mlp = layer.mlp

                def make_hook(idx):
                    def hook(module, inp, out):
                        # Mede sparsidade da entrada do down_proj
                        if hasattr(module, 'gate_proj'):
                            with torch.no_grad():
                                g = module.gate_proj(inp[0])
                                sp = (g.abs() < 1e-6).float().mean().item()
                                self.activation_sparsities[idx] = sp
                    return hook

                h = mlp.register_forward_hook(make_hook(i))
                self.hooks.append(h)
            except (AttributeError, IndexError):
                pass

    def report(self):
        if not self.activation_sparsities:
            print("  Sem dados de sparsidade (execute um forward pass primeiro).")
            return
        vals = list(self.activation_sparsities.values())
        print(f"\n  Sparsidade MLP por layer:")
        for idx in sorted(self.activation_sparsities):
            bar = "█" * int(self.activation_sparsities[idx] * 20)
            print(f"    Layer {idx:02d}: {self.activation_sparsities[idx]:.1%} {bar}")
        print(f"    Média: {sum(vals)/len(vals):.1%}")

    def remove(self):
        for h in self.hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inferência Gemma 4 Neo")
    p.add_argument("--model", required=True, help="Caminho para o modelo Neo")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--monitor-sparsity", action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="Usa torch.compile para máximo throughput (requer PyTorch 2.x)")
    p.add_argument("--speculative", action="store_true",
                   help="Usa speculative decoding (requer --draft)")
    p.add_argument("--draft", type=str, default=None,
                   help="Caminho para o modelo draft (E2B) para speculative decoding")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'═'*60}")
    print(f"  GEMMA 4 NEO — INFERÊNCIA")
    print(f"{'═'*60}")

    neo = Gemma4NeoInference(
        model_dir=args.model,
        compile_model=args.compile,
    )

    # Monitor de sparsidade (opcional)
    monitor = None
    if args.monitor_sparsity:
        monitor = SparsityMonitor(neo.model)
        print("  ✓ Monitor de sparsidade ativado")

    if args.benchmark:
        neo.benchmark(max_new_tokens=200)

    elif args.prompt:
        print(f"\n  Prompt: {args.prompt[:80]}...\n")
        print("  " + "─" * 58)

        if args.speculative and args.draft:
            # Speculative decoding
            print(f"  Modo: Speculative Decoding (draft: {args.draft})")
            draft_neo = Gemma4NeoInference(model_dir=args.draft)
            text, stats = neo.generate_speculative(
                prompt=args.prompt,
                draft_model=draft_neo,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(f"\n{text}\n")
            print(f"  ─── Stats ───────────────────────────────────────────")
            print(f"  Tokens: {stats['tokens_generated']} | "
                  f"Speed: {stats['tokens_per_sec']:.1f} tok/s | "
                  f"Acceptance: {stats['acceptance_rate']:.1%} | "
                  f"Speedup: {stats['speedup_vs_greedy']:.1f}×")
        else:
            # Geração padrão
            result = neo.generate_with_stats(
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=args.stream,
            )
            if not args.stream:
                print(f"\n{result['text']}\n")
            perf = result["performance"]
            print(f"  ─── Stats ───────────────────────────────────────────")
            print(f"  Tokens: {perf.get('tokens_generated', '?')} | "
                  f"Speed: {perf.get('tokens_per_sec', 0):.1f} tok/s | "
                  f"Time: {perf.get('time_sec', 0):.2f}s")
            sp = result["sparsity"].get("mean")
            if sp:
                print(f"  MLP sparsidade média: {sp:.1%}")

        if monitor:
            monitor.report()
            monitor.remove()

    else:
        # Modo interativo
        print("\n  Modo interativo. Digite 'quit' para sair.\n")
        while True:
            try:
                prompt = input("  > ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue
                text = neo.generate(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stream=True,
                )
                print()
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
