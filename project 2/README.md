# sparse_gemma4

Adaptação do **Gemma 4** para esparsidade estruturada (2:4 e 6:8), com pipeline completo de transferência de pesos e monitoramento de performance.

---

## Estrutura do projeto

```
sparse_gemma4/
├── configs/
│   └── sparsity_policy.py    # Política de esparsidade por layer (padrões glob)
├── core/
│   └── sparsifier.py         # Engine de esparsificação + transferência de pesos
├── monitor/
│   ├── profiler.py           # Métricas: FLOPs, tokens/s, latência, ativações
│   ├── gpu_monitor.py        # Thread de monitoramento: watts, temp, SM util
│   └── dashboard.py          # Geração de relatório HTML interativo
├── benchmark/
│   └── run_benchmark.py      # Pipeline completo dense vs 6:8 vs 2:4
├── utils/
│   └── layer_analysis.py     # Análise dimensional e compatibilidade
├── tests/
│   └── test_sparsifier.py    # Testes unitários (pytest)
├── quickstart.py             # Demo sem GPU, com MockGemma4
└── requirements.txt
```

---

## Fundamentos

| Abordagem | Zeros | Speedup kernel | Impacto acurácia | Requer |
|---|---|---|---|---|
| **Dense (baseline)** | 0% | 1.00x | — | — |
| **6:8 SlideSparse** | 25% | ~1.33x | Mínimo (~1-5%) | PyTorch ≥ 2.4 |
| **2:4 NVIDIA Sparse TC** | 50% | ~2.00x | Alto risco (reasoning) | Ampere+ (sm_80+) |
| Unstructured L1 | >90% | variável* | Mínimo | Kernels CUDA custom |

> **Por que 6:8 e não 2:4 direto?**  
> Em benchmarks de reasoning (GSM8K, MATH), o padrão 2:4 pode colapsar performance significativamente (ex: Qwen3 caiu de 54% → 15.3%). O padrão 6:8 via SlideSparse preserva ~95-99% da acurácia com ~1.33x de speedup — o equilíbrio ideal para produção sem re-treino.

---

## Decisões por módulo do Gemma 4

| Módulo | Modo aplicado | Razão |
|---|---|---|
| `language_model.layers.*.mlp.*` | **6:8** | FFN principal — 92% dos FLOPs do texto |
| `language_model.layers.*.self_attn.*` | **6:8** | GQA, dims ok (2048/512/4096/1024) |
| `language_model.layers.*.per_layer_*` | **SKIP** | Bottleneck 256-dim — AltUp/LAuReL crítico |
| `vision_tower.encoder.layers.*` | **2:4** | ClippableLinear nativo, 768-dim, seguro |
| `audio_tower.layers.*.feed_forward*` | **2:4** | Conformer FFN, 1024↔4096 |
| `audio_tower.layers.*.lconv1d*` | **SKIP** | Depthwise Conv1d — não mapeia para Sparse TC |
| `audio_tower.subsample_conv_projection` | **SKIP** | Conv2d subsampling |
| `embed_tokens` / `lm_head` | **SKIP** | Lookup table — sem ganho computacional |

---

## Instalação

```bash
pip install -r requirements.txt
```

**Requisitos de hardware para speedup real:**
- GPU NVIDIA Ampere ou superior (A100, A10G, RTX 3090, H100…)
- CUDA ≥ 12.1
- PyTorch ≥ 2.4 (para `to_sparse_semi_structured` estável)
- BF16 ou FP16 (FP32 não suportado em Sparse Tensor Cores)

---

## Uso

### 1. Análise rápida (sem GPU, sem modelo real)

```bash
cd sparse_gemma4
python quickstart.py
```

Executa análise teórica completa com `MockGemma4` — estima parâmetros,
elegibilidade por layer e FLOPs sem precisar baixar o modelo.

### 2. Esparsificar um modelo já carregado

```python
from transformers import AutoModelForCausalLM
from sparse_gemma4 import Gemma4Sparsifier, CONSERVATIVE_POLICY

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-pt-2b",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Aplica 6:8 no text decoder, 2:4 na vision/audio tower
sparsifier = Gemma4Sparsifier(
    model,
    policy=CONSERVATIVE_POLICY,
    policy_name="conservative",
    use_native_sparse=True,   # Ativa SparseSemiStructuredTensor para layers 2:4
)
report = sparsifier.apply()
print(report.summary())

# Salva modelo esparso
sparsifier.save_sparse_model("gemma4_sparse_68.pt")
```

### 3. Medir performance (dense vs sparse)

```python
from sparse_gemma4 import Gemma4Profiler
import torch

profiler = Gemma4Profiler(model, tokenizer, device=0, track_activation_sparsity=True)

# Baseline denso
inputs = tokenizer("Explain neural sparsity.", return_tensors="pt").to("cuda")
profiler.begin_run("dense")
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=200)
metrics = profiler.end_run("dense", inputs, out)
profiler.print_summary("dense")

# Depois de esparsificar...
profiler_sp = Gemma4Profiler(sparse_model, tokenizer, device=0)
profiler_sp.begin_run("sparse_68")
with torch.inference_mode():
    out_sp = sparse_model.generate(**inputs, max_new_tokens=200)
metrics_sp = profiler_sp.end_run("sparse_68", inputs, out_sp)

print(profiler.compare("dense", "sparse_68"))
profiler.to_json("results/metrics.json")
```

### 4. Benchmark completo (linha de comando)

```bash
python benchmark/run_benchmark.py \
    --model_path google/gemma-4-pt-2b \
    --policy conservative \
    --max_new_tokens 200 \
    --num_runs 3 \
    --output_dir ./results
```

**Flags:**
- `--skip_24` — pula benchmark 2:4 (mais rápido)
- `--skip_dense` — pula baseline (se já tiver os números)
- `--dtype bf16` — usa bfloat16 (recomendado para H100)
- `--prompts prompts.json` — prompts customizados

### 5. Dashboard interativo

```bash
# Após rodar o benchmark
python monitor/dashboard.py \
    --results_dir ./results \
    --output benchmark_report.html
```

Abre `benchmark_report.html` no browser — gráficos interativos de:
- Tokens/s por configuração e por prompt
- Memória GPU peak
- FLOPs estimados
- Esparsidade de ativações

### 6. Monitoramento de GPU/energia

```python
from sparse_gemma4.monitor.gpu_monitor import GpuMonitor

with GpuMonitor(device=0, interval_s=0.05) as mon:
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=200)

stats = mon.get_stats()
print(stats.summary(num_tokens=200))
# → Potência média: 187.3W | Energia: 1.247J | SM util: 73% | Eficiência: 160.4 tok/J
```

### 7. Análise dimensional de layers

```python
from sparse_gemma4.utils.layer_analysis import analyze_all_layers, print_layer_table

layers = analyze_all_layers(model)
print_layer_table(layers, top_n=30)

# Verificar esparsidade real após aplicação
from sparse_gemma4.utils.layer_analysis import sparsity_histogram
sparsity_histogram(model)
```

---

## Testes

```bash
pip install pytest
pytest tests/ -v
```

---

## Métricas coletadas

| Métrica | Fonte | Granularidade |
|---|---|---|
| Tokens/s prefill | Wall clock + token count | Por run |
| Tokens/s decode | Wall clock + token count | Por run |
| Latência total | `time.perf_counter()` | Por run |
| GPU time (ms) | `torch.cuda.Event` | Por run |
| Memória alocada (MB) | `torch.cuda.memory_stats` | Por run |
| Memória peak (MB) | `torch.cuda.memory_stats` | Por run |
| FLOPs (GFLOPs) | Analítico por arquitetura | Por seq_len |
| Esparsidade ativações (%) | Forward hooks em Linear | Por layer |
| Potência GPU (W) | NVML / pynvml | 50ms samples |
| Energia total (J) | Integração trapezoidal | Por run |
| Eficiência (tok/J) | tokens ÷ joules | Por run |
| Temperatura GPU (°C) | NVML | 50ms samples |
| Utilização SM (%) | NVML | 50ms samples |

---

## Referências científicas

- **2:4 Sparsity** — NVIDIA Ampere Architecture Whitepaper (2020)  
- **SlideSparse (6:8)** — "Sliding Window Sparse Patterns for LLMs" (arxiv 2603.05232, Março 2026)  
- **Sakana AI + NVIDIA** — "Sparser, Faster, Lighter" — L1 regularization para >99% sparsity  
- **ASP (Automatic SParsity)** — `github.com/NVIDIA/apex` — magnitude pruning para 2:4  
- **torch.sparse** — `to_sparse_semi_structured` — PyTorch ≥ 2.1  
- **Gemma 4** — Google DeepMind (2025) — arquitetura multimodal com AltUp/LAuReL

---

## Roadmap

- [ ] Integrar SlideSparse runtime quando open-source disponível  
- [ ] Fine-tuning com máscara fixa (sparse-aware training) pós-pruning  
- [ ] Combinar com quantização FP8 (multiplicação de ganhos)  
- [ ] Suporte a `bitsandbytes` para GPU com memória limitada  
- [ ] Benchmarks de acurácia: MMLU, GSM8K, HumanEval  
- [ ] Suporte a 8:16 para hardware de próxima geração (Blackwell)
