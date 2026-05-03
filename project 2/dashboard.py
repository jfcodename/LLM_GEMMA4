"""
sparse_gemma4/monitor/dashboard.py
====================================
Dashboard de visualização dos resultados do benchmark.
Gera gráficos comparativos e HTML interativo.

Uso:
  python monitor/dashboard.py --results_dir ./results --output benchmark_report.html
"""

import json
import sys
from pathlib import Path
from typing import Optional


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gemma 4 — Sparse vs Dense Benchmark</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3e;
    --text: #e2e8f0; --muted: #94a3b8;
    --green: #10b981; --amber: #f59e0b; --red: #ef4444; --blue: #3b82f6;
    --purple: #8b5cf6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, sans-serif; padding: 24px; }}
  h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 32px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }}
  .card h3 {{ font-size: 13px; font-weight: 500; color: var(--muted); margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 14px;
  }}
  .metric-row:last-child {{ border-bottom: none; }}
  .metric-label {{ color: var(--muted); }}
  .metric-values {{ display: flex; gap: 16px; }}
  .val {{ min-width: 70px; text-align: right; font-variant-numeric: tabular-nums; }}
  .val.dense {{ color: var(--blue); }}
  .val.s68 {{ color: var(--green); }}
  .val.s24 {{ color: var(--amber); }}
  .badge {{
    display: inline-block; font-size: 11px; font-weight: 600;
    padding: 2px 8px; border-radius: 4px;
  }}
  .badge.speedup {{ background: rgba(16,185,129,0.15); color: var(--green); }}
  .badge.slower  {{ background: rgba(239,68,68,0.15); color: var(--red); }}
  .legend {{
    display: flex; gap: 20px; margin-bottom: 16px; font-size: 12px; color: var(--muted);
  }}
  .legend-dot {{
    display: inline-block; width: 10px; height: 10px;
    border-radius: 2px; margin-right: 4px;
  }}
  .chart-wrap {{ position: relative; height: 240px; }}
  .tab-bar {{
    display: flex; gap: 4px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 12px;
  }}
  .tab {{
    padding: 6px 16px; border-radius: 6px; cursor: pointer;
    font-size: 13px; color: var(--muted); border: 1px solid transparent;
  }}
  .tab.active {{ background: var(--card); border-color: var(--border); color: var(--text); }}
  .section {{ display: none; }}
  .section.active {{ display: block; }}
  .policy-note {{
    background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.3);
    border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #c4b5fd;
    margin-bottom: 20px; line-height: 1.6;
  }}
</style>
</head>
<body>
<h1>Gemma 4 — Sparse Benchmark</h1>
<p class="subtitle">Dense vs 6:8 SlideSparse vs 2:4 NVIDIA Sparse TC · Política: {policy_name}</p>

<div class="policy-note">
  <strong>Nota sobre 2:4:</strong> Em modelos de reasoning (transformers grandes), esparsidade 2:4 pode colapsar
  a performance (ex: Qwen3 caiu de 54% → 15.3% em benchmarks de raciocínio). 
  <strong>6:8 é recomendado</strong> para deployment em produção — preserva ~95-99% da acurácia com ~1.33x speedup.
</div>

<div class="tab-bar">
  <div class="tab active" onclick="switchTab('throughput')">Throughput</div>
  <div class="tab" onclick="switchTab('memory')">Memória</div>
  <div class="tab" onclick="switchTab('flops')">FLOPs</div>
  <div class="tab" onclick="switchTab('activation')">Ativação Esparsidade</div>
</div>

<div id="throughput" class="section active">
  <div class="legend">
    <span><span class="legend-dot" style="background:#3b82f6"></span>Dense</span>
    <span><span class="legend-dot" style="background:#10b981"></span>6:8 SlideSparse</span>
    <span><span class="legend-dot" style="background:#f59e0b"></span>2:4 NVIDIA</span>
  </div>
  <div class="card" style="margin-bottom:16px">
    <h3>Tokens por segundo (decode)</h3>
    <div class="chart-wrap"><canvas id="tpsChart" aria-label="Tokens per second comparison"></canvas></div>
  </div>
  <div class="grid" id="tps-details"></div>
</div>

<div id="memory" class="section">
  <div class="card">
    <h3>Memória GPU Peak (MB)</h3>
    <div class="chart-wrap"><canvas id="memChart" aria-label="Peak GPU memory comparison"></canvas></div>
  </div>
</div>

<div id="flops" class="section">
  <div class="card">
    <h3>FLOPs Estimados (GFLOPs)</h3>
    <div class="chart-wrap"><canvas id="flopsChart" aria-label="Estimated FLOPs"></canvas></div>
  </div>
</div>

<div id="activation" class="section">
  <div class="card">
    <h3>Esparsidade Média de Ativações</h3>
    <div class="chart-wrap"><canvas id="actChart" aria-label="Activation sparsity"></canvas></div>
  </div>
</div>

<script>
const DATA = {data_json};
const PROMPTS = Object.keys(DATA);

const COLORS = {{
  dense: '#3b82f6',
  sparse_68: '#10b981',
  sparse_24: '#f59e0b',
}};

function getVal(p, key, sub) {{
  const m = DATA[p]?.[key];
  return m ? (m[sub] ?? 0) : 0;
}}

function makeChart(id, labels, datasets, unit='') {{
  const ctx = document.getElementById(id);
  if (!ctx) return;
  new Chart(ctx, {{
    type: 'bar',
    data: {{ labels, datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: (c) => ` ${{c.dataset.label}}: ${{c.raw.toFixed(1)}}${{unit}}` }} }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#94a3b8', font: {{ size: 11 }} }}, grid: {{ color: '#2a2d3e' }} }},
        y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 11 }} }}, grid: {{ color: '#2a2d3e' }},
          beginAtZero: true }}
      }}
    }}
  }});
}}

// Throughput chart
makeChart('tpsChart', PROMPTS, [
  {{ label: 'Dense',          data: PROMPTS.map(p => getVal(p,'dense','decode_tps')),     backgroundColor: COLORS.dense }},
  {{ label: '6:8 SlideSparse',data: PROMPTS.map(p => getVal(p,'sparse_68','decode_tps')), backgroundColor: COLORS.sparse_68 }},
  {{ label: '2:4 NVIDIA',     data: PROMPTS.map(p => getVal(p,'sparse_24','decode_tps')), backgroundColor: COLORS.sparse_24 }},
], ' tok/s');

// Memory chart
makeChart('memChart', PROMPTS, [
  {{ label: 'Dense',          data: PROMPTS.map(p => getVal(p,'dense','peak_memory_mb')),     backgroundColor: COLORS.dense }},
  {{ label: '6:8 SlideSparse',data: PROMPTS.map(p => getVal(p,'sparse_68','peak_memory_mb')), backgroundColor: COLORS.sparse_68 }},
  {{ label: '2:4 NVIDIA',     data: PROMPTS.map(p => getVal(p,'sparse_24','peak_memory_mb')), backgroundColor: COLORS.sparse_24 }},
], ' MB');

// FLOPs chart
makeChart('flopsChart', PROMPTS, [
  {{ label: 'Dense',          data: PROMPTS.map(p => getVal(p,'dense','estimated_flops_gflops')),     backgroundColor: COLORS.dense }},
  {{ label: '6:8 SlideSparse',data: PROMPTS.map(p => getVal(p,'sparse_68','estimated_flops_gflops')), backgroundColor: COLORS.sparse_68 }},
  {{ label: '2:4 NVIDIA',     data: PROMPTS.map(p => getVal(p,'sparse_24','estimated_flops_gflops')), backgroundColor: COLORS.sparse_24 }},
], ' G');

// Activation sparsity chart
makeChart('actChart', PROMPTS, [
  {{ label: 'Dense',          data: PROMPTS.map(p => (getVal(p,'dense','avg_activation_sparsity')*100)),     backgroundColor: COLORS.dense }},
  {{ label: '6:8 SlideSparse',data: PROMPTS.map(p => (getVal(p,'sparse_68','avg_activation_sparsity')*100)), backgroundColor: COLORS.sparse_68 }},
  {{ label: '2:4 NVIDIA',     data: PROMPTS.map(p => (getVal(p,'sparse_24','avg_activation_sparsity')*100)), backgroundColor: COLORS.sparse_24 }},
], '%');

// Per-prompt details cards
const container = document.getElementById('tps-details');
PROMPTS.forEach(p => {{
  const dense = DATA[p]?.dense;
  const s68   = DATA[p]?.sparse_68;
  const s24   = DATA[p]?.sparse_24;
  const sp68  = DATA[p]?.speedup_68 ?? 0;
  const sp24  = DATA[p]?.speedup_24 ?? 0;
  
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <h3>${{p.replace(/_/g,' ')}}</h3>
    <div class="metric-row">
      <span class="metric-label">tok/s decode</span>
      <div class="metric-values">
        <span class="val dense">${{dense?.decode_tps?.toFixed(1) ?? '—'}}</span>
        <span class="val s68">${{s68?.decode_tps?.toFixed(1) ?? '—'}}</span>
        <span class="val s24">${{s24?.decode_tps?.toFixed(1) ?? '—'}}</span>
      </div>
    </div>
    <div class="metric-row">
      <span class="metric-label">Latência (s)</span>
      <div class="metric-values">
        <span class="val dense">${{dense?.total_time_s?.toFixed(3) ?? '—'}}</span>
        <span class="val s68">${{s68?.total_time_s?.toFixed(3) ?? '—'}}</span>
        <span class="val s24">${{s24?.total_time_s?.toFixed(3) ?? '—'}}</span>
      </div>
    </div>
    <div class="metric-row">
      <span class="metric-label">Mem. peak (MB)</span>
      <div class="metric-values">
        <span class="val dense">${{dense?.peak_memory_mb?.toFixed(0) ?? '—'}}</span>
        <span class="val s68">${{s68?.peak_memory_mb?.toFixed(0) ?? '—'}}</span>
        <span class="val s24">${{s24?.peak_memory_mb?.toFixed(0) ?? '—'}}</span>
      </div>
    </div>
    <div class="metric-row" style="border:none; padding-top:12px">
      <span class="metric-label">Speedup</span>
      <div class="metric-values" style="gap:8px">
        <span class="badge ${{sp68 >= 1 ? 'speedup' : 'slower'}}">${{sp68.toFixed(2)}}x 6:8</span>
        <span class="badge ${{sp24 >= 1 ? 'speedup' : 'slower'}}">${{sp24.toFixed(2)}}x 2:4</span>
      </div>
    </div>
  `;
  container.appendChild(card);
}});

function switchTab(name) {{
  document.querySelectorAll('.tab, .section').forEach(el => el.classList.remove('active'));
  document.querySelector(`.tab[onclick="switchTab('${{name}}')"]`).classList.add('active');
  document.getElementById(name).classList.add('active');
}}
</script>
</body>
</html>'''


def generate_html_report(
    results_json_path: str,
    output_path: str = "benchmark_report.html",
    policy_name: str = "conservative",
) -> None:
    """Gera relatório HTML interativo a partir do JSON de resultados."""
    with open(results_json_path) as f:
        data = json.load(f)

    html = HTML_TEMPLATE.format(
        data_json=json.dumps(data, indent=2),
        policy_name=policy_name,
    )

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Dashboard gerado: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output", default="benchmark_report.html")
    parser.add_argument("--policy", default="conservative")
    args = parser.parse_args()

    json_path = Path(args.results_dir) / "benchmark_final.json"
    if not json_path.exists():
        print(f"Arquivo não encontrado: {json_path}")
        sys.exit(1)

    generate_html_report(str(json_path), args.output, args.policy)
