import argparse
import json
import os
import re

import requests


# configuracao do modulo
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"
TEMPERATURE = 0.0
MAX_TOKENS = 1200


# Leitura e preparacao dos dados
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_metrics_path(insights_path: str, explicit_path: str | None = None) -> str | None:
    if explicit_path:
        return explicit_path if os.path.exists(explicit_path) else None
    folder = os.path.dirname(os.path.abspath(insights_path))
    candidates = [
        os.path.join(folder, "metrics.json"),
        os.path.join(os.path.dirname(folder), "output", "metrics.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def sort_zones_by_performance(zones: dict) -> list:
    return sorted(
        zones.items(),
        key=lambda item: (item[1]["stop_rate_pct"], item[1]["avg_dwell_s"], item[1]["total_visits"]),
        reverse=True,
    )


def sort_zones_by_reach(funnel: dict) -> list:
    return sorted(funnel["zone_reach"].items(), key=lambda item: item[1]["reach_pct"])


def insight_by_category(insights: list, category: str) -> list:
    return [i for i in insights if i.get("categoria") == category]


# chamada ao LLM
def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "seed": 42},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["response"].strip()


def numbers_in(text: str) -> set:
    return {n.replace(",", ".") for n in re.findall(r"\b\d+(?:[.,]\d+)?\b", text or "")}


def clean_llm_section(text: str) -> str:
    text = re.sub(r"```.*?```", "", text or "", flags=re.DOTALL)
    text = re.sub(r"^#+\s.*\n?", "", text.strip())
    return text.strip()


def build_report_prompt(section_name: str, factual_draft: str) -> str:
    return f"""Es um consultor de retalho a escrever para um gestor de loja em portugues europeu.
Transforma o RASCUNHO FACTUAL abaixo numa seccao clara do briefing semanal.

REGRAS OBRIGATORIAS:
- Usa apenas os factos e numeros do rascunho.
- Nao cries numeros, percentagens, datas, horas, zonas ou causas novas.
- Mantem linguagem simples, operacional e sem jargao tecnico desnecessario.
- Evita termos tecnicos como "sigma" e "desvio padrao"; escreve "muito fora do habitual".
- Nao uses markdown, nem titulo da seccao.
- Se precisares de mencionar incerteza, escreve "hipotese operacional", nao uses "pode indicar".

SECCAO: {section_name}
RASCUNHO FACTUAL:
{factual_draft}

Escreve apenas o texto final da seccao:"""


def rewrite_sections_with_llm(sections: dict, model: str) -> tuple[dict, dict]:
    rewritten = {}
    prompts = {}
    names = {
        "resumo": "Resumo Executivo",
        "trafego": "Performance de Trafego",
        "zonas": "Analise de Zonas",
        "funil": "Funil de Clientes",
        "anomalias": "Anomalias da Semana",
        "recomendacoes": "Recomendacoes para a Proxima Semana",
    }
    for key, draft in sections.items():
        prompt = build_report_prompt(names[key], draft)
        prompts[key] = prompt
        raw = call_ollama(prompt, model)
        clean = clean_llm_section(raw)

        # Nota: se o LLM inventar numeros, fica o rascunho factual.
        if not numbers_in(clean).issubset(numbers_in(draft)):
            print(f"[report] AVISO: numeros novos na seccao {key}; a usar rascunho factual.")
            rewritten[key] = draft
        else:
            rewritten[key] = clean
    return rewritten, prompts


# Criação das seções do briefing
def build_sections_from_metrics(metrics: dict, insights: list) -> dict:
    traffic = metrics["traffic"]
    zones = metrics["zones"]["per_zone"]
    sequences = metrics["zones"]["top_sequences"]
    funnel = metrics["funnel"]
    demographics = metrics["demographics"]
    anomalies = metrics["anomalies"].get("anomalies", [])

    busiest = traffic["busiest_day"]
    quietest = traffic["quietest_day"]
    best_zones = sort_zones_by_performance(zones)[:3]
    low_reach = sort_zones_by_reach(funnel)[:3]
    top_zone_id, top_zone = max(zones.items(), key=lambda item: item[1]["total_visits"])
    worst_zone_id, worst_zone = min(zones.items(), key=lambda item: item[1]["total_visits"])
    no_checkout = funnel["no_checkout_profile"]
    top_no_checkout_age = max(no_checkout["age_dist"].items(), key=lambda item: item[1])
    top_sequence = sequences[0] if sequences else {"sequence": "sem sequencia dominante", "count": 0}

    resumo = (
        f"A loja recebeu {traffic['total_visitors_week']} visitantes na semana; "
        f"{busiest} foi o dia mais movimentado com {traffic['visitors_per_day'][busiest]} visitantes e "
        f"{quietest} o mais calmo com {traffic['visitors_per_day'][quietest]}. "
        f"Chegaram a caixa {funnel['reached_checkout']} visitantes de um total de {funnel['total_visitors']}, "
        f"uma conversao de {funnel['conversion_rate_pct']}%. "
        f"A anomalia mais forte ocorreu em {anomalies[0]['zone_id']} as {anomalies[0]['hour_of_day']}h, "
        f"com {anomalies[0]['day7_visitors']} visitantes contra media de {anomalies[0]['baseline_mean']}."
        if anomalies else
        f"A loja recebeu {traffic['total_visitors_week']} visitantes e converteu {funnel['reached_checkout']} em caixa."
    )

    trafego = (
        f"A afluencia semanal foi de {traffic['total_visitors_week']} visitantes. "
        f"O dia mais movimentado foi {busiest}, com {traffic['visitors_per_day'][busiest]} visitantes, "
        f"e o dia mais calmo foi {quietest}, com {traffic['visitors_per_day'][quietest]} visitantes.\n\n"
        f"A hora de pico da semana foi as {traffic['peak_hour']}h. "
        f"A duracao media de visita foi {traffic['avg_visit_duration_min']} minutos, com mediana de "
        f"{traffic['median_visit_duration_min']} minutos.\n\n"
        f"Para staffing, o foco operacional deve estar no dia {busiest} e no bloco em torno das "
        f"{traffic['peak_hour']}h, evitando colocar reposicao, pausas longas ou tarefas administrativas nesse periodo."
    )

    best_text = "; ".join(
        f"{zid} ({data['total_visits']} visitas, dwell {data['avg_dwell_s']}s, paragem {data['stop_rate_pct']}%)"
        for zid, data in best_zones
    )
    low_text = "; ".join(
        f"{zid} ({data['visitors']} visitantes, alcance {data['reach_pct']}%)"
        for zid, data in low_reach
    )
    zonas = (
        f"As tres zonas com melhor performance combinada de paragem e permanencia foram {best_text}. "
        f"Estas zonas merecem manter boa exposicao de produto porque ja conseguem prender a atencao dos clientes.\n\n"
        f"A zona com maior volume foi {top_zone_id}, com {top_zone['total_visits']} visitas. "
        f"A zona com menor volume foi {worst_zone_id}, com {worst_zone['total_visits']} visitas; deve ser analisada "
        f"fisicamente para perceber se o problema e localizacao, sinalizacao ou sortido.\n\n"
        f"As zonas com menor alcance foram {low_text}. A recomendacao e criar ligacoes visuais a partir das zonas "
        f"mais visitadas e testar uma promocao simples nestes pontos.\n\n"
        f"A sequencia mais frequente foi {top_sequence['sequence']}, com {top_sequence['count']} ocorrencias. "
        f"Este percurso deve ser usado para decidir onde colocar comunicacao e onde evitar obstaculos."
    )

    funil = (
        f"No funil, ha {funnel['total_visitors']} visitantes unicos, {funnel['total_entered']} com passagem registada "
        f"em zonas de entrada e {funnel['reached_checkout']} que chegaram a caixa. A taxa de conversao para caixa e "
        f"{funnel['conversion_rate_pct']}%.\n\n"
        f"Ficaram sem chegar a caixa {no_checkout['total']} visitantes. As zonas com menor alcance no percurso sao "
        f"{', '.join(z for z, _ in low_reach)}, pelo que a perda de trafego deve ser observada sobretudo nesses pontos.\n\n"
        f"No grupo sem checkout, a faixa etaria mais frequente e {top_no_checkout_age[0]}, com {top_no_checkout_age[1]} pessoas. "
        f"A distribuicao de genero nesse grupo e M={no_checkout['gender_dist'].get('M', 0)} e "
        f"F={no_checkout['gender_dist'].get('F', 0)}.\n\n"
        f"A acao recomendada e observar estes visitantes nas zonas de menor alcance e testar sinalizacao direta para caixa "
        f"e produtos de decisao rapida antes da hora de pico."
    )

    anomaly_paragraphs = []
    for anomaly in anomalies[:5]:
        direction = "abaixo" if anomaly["direction"] == "abaixo" else "acima"
        cause = "obstrucao, reposicao ou falha de sensor" if direction == "abaixo" else "promocao local, concentracao de fila ou evento pontual"
        anomaly_paragraphs.append(
            f"Em {anomaly['zone_id']} as {anomaly['hour_of_day']}h, o dia analisado registou "
            f"{anomaly['day7_visitors']} visitantes contra uma media de {anomaly['baseline_mean']}. "
            f"O valor ficou {direction} do padrao historico, com magnitude calculada de {anomaly['sigma_dev']}. "
            f"A causa mais provavel a verificar e {cause}; a acao e inspecionar a zona antes da abertura seguinte "
            f"e confirmar a leitura do sensor nesse horario."
        )
    anomalias = "\n\n".join(anomaly_paragraphs) if anomaly_paragraphs else "Nao foram detetadas anomalias acima do limiar definido."

    ordered = sorted(
        insights,
        key=lambda i: {"imediata": 0, "esta_semana": 1, "proximo_mes": 2}.get(i.get("urgencia"), 1),
    )[:5]
    recomendacoes = "\n".join(
        f"{idx}. [{ins.get('urgencia', 'esta_semana')}] {ins.get('recomendacao', '').strip()}"
        for idx, ins in enumerate(ordered, 1)
    )

    return {
        "resumo": resumo,
        "trafego": trafego,
        "zonas": zonas,
        "funil": funil,
        "anomalias": anomalias,
        "recomendacoes": recomendacoes,
    }


def build_sections_from_insights(insights: list, resumo_executivo: str) -> dict:
    def join(category: str) -> str:
        items = insight_by_category(insights, category)
        if not items:
            return "Sem insights suficientes para esta seccao."
        return "\n\n".join(
            f"{i['observacao']} {i['implicacao']} Recomendacao: {i['recomendacao']}"
            for i in items
        )

    ordered = sorted(
        insights,
        key=lambda i: {"imediata": 0, "esta_semana": 1, "proximo_mes": 2}.get(i.get("urgencia"), 1),
    )[:5]
    return {
        "resumo": resumo_executivo or "\n".join(i.get("titulo", "") for i in insights[:3]),
        "trafego": join("trafego"),
        "zonas": join("zona"),
        "funil": join("funil"),
        "anomalias": join("anomalia"),
        "recomendacoes": "\n".join(f"{idx}. [{i.get('urgencia')}] {i.get('recomendacao')}" for idx, i in enumerate(ordered, 1)),
    }


def assemble_report(sections: dict, semana: str) -> str:
    return f"""# Briefing Semanal - Loja de Retalho
## Semana de {semana}
*Gerado automaticamente pelo sistema de retail intelligence*

---

## 1. Resumo Executivo
{sections['resumo']}

---

## 2. Performance de Trafego
{sections['trafego']}

---

## 3. Analise de Zonas
{sections['zonas']}

---

## 4. Funil de Clientes
{sections['funil']}

---

## 5. Anomalias da Semana
{sections['anomalias']}

---

## 6. Recomendacoes para a Proxima Semana
{sections['recomendacoes']}

---
*Relatorio gerado por pipeline automatizado. Os factos numericos foram retirados de metrics.json ou de insights.json validado.*
"""


def save_report_prompt(prompts_dir: str, prompts: dict | None = None) -> None:
    os.makedirs(prompts_dir, exist_ok=True)
    content = "REPORT_V2\nO LLM redige o report a partir de rascunhos factuais calculados em Python.\n"
    content += "Guardrail: se a resposta introduzir numeros novos, a seccao volta ao rascunho factual.\n"
    if prompts:
        for name, prompt in prompts.items():
            content += f"\n{'=' * 60}\nSECCAO: {name.upper()}\n{'=' * 60}\n{prompt}\n"
    with open(os.path.join(prompts_dir, "report_v1.txt"), "w", encoding="utf-8") as f:
        f.write(content)


# Execução por linha de comandos
def main():
    parser = argparse.ArgumentParser(description="Gera weekly_report.md a partir de insights.json e metrics.json.")
    parser.add_argument("--input", required=True, help="Caminho para insights.json")
    parser.add_argument("--output", required=True, help="Caminho para weekly_report.md")
    parser.add_argument("--metrics", default=None, help="Caminho opcional para metrics.json")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--semana", default="semana em analise")
    args = parser.parse_args()

    print(f"[report] A carregar {args.input}...")
    data = load_json(args.input)
    insights = data.get("insights", [])
    resumo_exec = data.get("resumo_executivo", "")

    metrics_path = find_metrics_path(args.input, args.metrics)
    if metrics_path:
        print(f"[report] A usar metricas verificaveis: {metrics_path}")
        metrics = load_json(metrics_path)
        sections = build_sections_from_metrics(metrics, insights)
    else:
        print("[report] AVISO: metrics.json nao encontrado; fallback para insights.json validado.")
        sections = build_sections_from_insights(insights, resumo_exec)

    prompts_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(args.output)), "..", "prompts"))
    report_prompts = {}
    try:
        print(f"[report] A redigir seccoes com LLM local: {args.model}")
        sections, report_prompts = rewrite_sections_with_llm(sections, args.model)
    except requests.RequestException as exc:
        print(f"[report] AVISO: LLM indisponivel ({exc}); a usar rascunhos factuais.")

    save_report_prompt(prompts_dir, report_prompts)

    report_md = assemble_report(sections, args.semana)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_md)

    words = len(sections["resumo"].split())
    print(f"[report] weekly_report.md guardado em: {args.output}")
    print(f"[report] Resumo executivo: {words} palavras (limite: 150)")


if __name__ == "__main__":
    main()
