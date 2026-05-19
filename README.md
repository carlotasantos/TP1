# TP1 — From Raw Detections to Real Intelligence
## LIACD 2025/2026

Pipeline que reconstrói trajectórias de clientes a partir de eventos anónimos de visão computacional e gera um briefing semanal automático para o gestor de loja.

---

## Requisitos

- Python 3.10+
- [Ollama](https://ollama.com) com o modelo `llama3.1:8b`

---

## Instalação

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
```

---

## Execução do pipeline completo

```bash
python src/stitcher.py --input data/events.csv --output output/journeys.csv --zones data/zones.json
python src/analytics.py --input output/journeys.csv --output output/metrics.json
python src/insights.py --input output/metrics.json --output output/insights.json --strategy both --model llama3.1:8b
python src/report.py --input output/insights.json --output output/weekly_report.md --metrics output/metrics.json --model llama3.1:8b
```

---

## Avaliação end-to-end

Comando base (sem ground truth — deteção de anomalias reportada como N/A):
```bash
python evaluate.py --data data/events_validation.csv --output output/evaluation_report.json
```

Com ground truth para medir deteção de anomalias conhecidas (opcional):
```bash
python evaluate.py --data data/events_validation.csv --output output/evaluation_report.json --ground-truth data/ground_truth.json
```

---

## Estrutura do projecto

```
tp1/
├── README.md
├── requirements.txt
├── evaluate.py
├── data/
│   ├── events.csv
│   ├── events_validation.csv
│   ├── ground_truth.json
│   └── zones.json
├── src/
│   ├── stitcher.py
│   ├── analytics.py
│   ├── insights.py
│   ├── report.py
│   └── inject_anomalies.py
├── prompts/
│   ├── insights_zero_shot_v1.txt
│   ├── insights_few_shot_v1.txt
│   └── report_v1.txt
└── output/
    ├── journeys.csv
    ├── metrics.json
    ├── insights.json
    └── weekly_report.md
```

---

## Módulos

| Módulo | Input | Output | Descrição |
|---|---|---|---|
| `stitcher.py` | `events.csv` | `journeys.csv` | Reconstrói trajectórias individuais a partir de eventos anónimos |
| `analytics.py` | `journeys.csv` | `metrics.json` | Calcula métricas de tráfego, zonas, funil, demografia e anomalias |
| `insights.py` | `metrics.json` | `insights.json` | Gera insights acionáveis via LLM (estratégias zero-shot e few-shot) |
| `report.py` | `insights.json` | `weekly_report.md` | Gera briefing semanal em Markdown para o gestor de loja |
| `evaluate.py` | `events_validation.csv` | `evaluation_report.json` | Avalia o pipeline end-to-end com métricas de qualidade |
| `inject_anomalies.py` | `events.csv` | `events_validation.csv` + `ground_truth.json` | Injeta anomalias conhecidas para teste |

---

## Modelo LLM

O pipeline usa `llama3.1:8b` via Ollama com `temperature=0` e `seed=42` para garantir reprodutibilidade.

Para usar outro modelo:

```bash
python src/insights.py --input output/metrics.json --output output/insights.json --model mistral:7b
python src/report.py --input output/insights.json --output output/weekly_report.md --metrics output/metrics.json --model mistral:7b
```

---

## Reprodutibilidade

Todos os componentes com aleatoriedade têm seed fixo (`seed=42`). O harness de avaliação usa `temperature=0`. Os resultados são reprodutíveis em execuções sucessivas.
