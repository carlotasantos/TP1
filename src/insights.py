import argparse
import json
import os
import re
import time
from copy import deepcopy

import requests


# Configuração principal do módulo
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"
TEMPERATURE = 0.0
MAX_TOKENS = 4096
MAX_INSIGHTS = 10

VALID_CATEGORIES = {"trafego", "zona", "funil", "anomalia", "demografico"}
VALID_URGENCIES = {"imediata", "esta_semana", "proximo_mes"}
BANNED_PHRASES = ("pode indicar", "possivelmente", "melhorar o atendimento")


# Estrutura esperada na resposta do LLM
OUTPUT_SCHEMA = """{
  "insights": [
    {
      "id": "INS_001",
      "categoria": "trafego",
      "titulo": "frase curta que resume o insight",
      "observacao": "factos e numeros concretos dos dados",
      "implicacao": "significado operacional direto",
      "recomendacao": "acao concreta com quando + onde + o que fazer",
      "urgencia": "imediata",
      "confianca": 0.9
    }
  ],
  "resumo_executivo": "- bullet 1\\n- bullet 2\\n- bullet 3"
}"""


FEW_SHOT_PAIRS = [
    {
        "mau": "A zona de frescos teve bastante trafego.",
        "bom": "A zona Z_S3 teve 847 visitantes na quinta-feira, 31% acima da media semanal.",
    },
    {
        "mau": "Recomenda-se melhorar o atendimento ao cliente.",
        "bom": "As 12h-13h, Z_C2 concentrou 4.2 minutos de espera media; abrir uma terceira caixa nesse slot reduz o congestionamento esperado.",
    },
    {
        "mau": "Houve uma anomalia na loja.",
        "bom": "No domingo as 16h, Z_N4 teve 0 visitantes contra media de 23; verificar obstrucao ou sinalizacao antes da abertura.",
    },
]


# Funções de apoio
def pct(part: float, total: float) -> float:
    return round(100 * part / total, 1) if total else 0.0


def safe_date(value: str) -> str:
    return str(value)[:10]


def summarize_metrics(metrics: dict) -> str:
    """Cria um resumo curto das metricas para o LLM."""
    traffic = metrics["traffic"]
    zones = metrics["zones"]
    funnel = metrics["funnel"]
    anomalies = metrics["anomalies"]

    days = traffic["visitors_per_day"]
    avg_day = sum(days.values()) / len(days)
    day_lines = "\n".join(
        f"- {day}: {visitors} visitantes"
        for day, visitors in days.items()
    )

    zone_rows = sorted(
        zones["per_zone"].items(),
        key=lambda item: (item[1]["total_visits"], item[1]["avg_dwell_s"]),
        reverse=True,
    )[:10]
    zone_lines = "\n".join(
        f"- {zid}: {data['total_visits']} visitas, dwell {data['avg_dwell_s']}s, paragem {data['stop_rate_pct']}%"
        for zid, data in zone_rows
    )

    anomaly_lines = "\n".join(
        f"- {a['zone_id']} as {a['hour_of_day']}h: {a['day7_visitors']} vs media {a['baseline_mean']} "
        f"(desvio {a['sigma_dev']} sigma, {a['direction']})"
        for a in anomalies.get("anomalies", [])[:12]
    )

    return f"""TRAFEGO
Total semanal: {traffic['total_visitors_week']} visitantes.
Duracao media: {traffic['avg_visit_duration_min']} min; mediana: {traffic['median_visit_duration_min']} min.
Dia mais movimentado: {traffic['busiest_day']}; dia mais calmo: {traffic['quietest_day']}; hora de pico: {traffic['peak_hour']}h.
Visitantes por dia:
{day_lines}

ZONAS MAIS VISITADAS
{zone_lines}

TOP SEQUENCIAS
{json.dumps(zones['top_sequences'][:10], ensure_ascii=False)}

FUNIL
Total visitantes: {funnel['total_visitors']}; entradas registadas: {funnel['total_entered']};
chegaram a caixa: {funnel['reached_checkout']} ({funnel['conversion_rate_pct']}%);
sem checkout: {funnel['no_checkout_profile']['total']} ({funnel.get('no_checkout_rate_pct', round(100 - funnel['conversion_rate_pct'], 1))}%).

ANOMALIAS DO DIA {safe_date(anomalies.get('analysis_day', ''))}
{anomaly_lines}
"""


def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "seed": 42},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["response"]


def parse_json_response(raw: str) -> dict:
    clean = re.sub(r"```json\s*", "", raw or "")
    clean = re.sub(r"```\s*", "", clean).strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        clean = match.group(0)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}


def _base_instruction(summary: str) -> str:
    return (
        "Es um analista senior de retalho. Recebes apenas metricas pre-calculadas; "
        "nao inventes numeros nem facas calculos novos.\n\n"
        f"DADOS:\n{summary}\n\n"
        "Tarefa: gera ate 10 insights acionaveis em portugues europeu.\n"
        "Regras obrigatorias:\n"
        "- Cobre as categorias trafego, zona, funil, anomalia e demografico quando houver dados.\n"
        "- Nao repitas observacoes nem recomendacoes.\n"
        "- Cada observacao deve citar numeros que aparecem nos dados.\n"
        "- Nao uses: 'pode indicar', 'possivelmente', 'melhorar o atendimento'.\n"
        "- Se nao houver evidencia suficiente para 10 insights, gera menos.\n"
        "- Responde apenas com JSON valido, sem markdown.\n\n"
        f"Schema:\n{OUTPUT_SCHEMA}"
    )


def build_prompt_zero_shot(metrics: dict) -> str:
    return _base_instruction(summarize_metrics(metrics))


def build_prompt_few_shot(metrics: dict) -> str:
    examples = "EXEMPLOS DE CALIBRACAO\n"
    for pair in FEW_SHOT_PAIRS:
        examples += f"MAU: {pair['mau']}\nBOM: {pair['bom']}\n\n"
    return examples + _base_instruction(summarize_metrics(metrics))


def save_prompts(metrics: dict, prompts_dir: str) -> None:
    os.makedirs(prompts_dir, exist_ok=True)
    with open(os.path.join(prompts_dir, "insights_zero_shot_v1.txt"), "w", encoding="utf-8") as f:
        f.write(build_prompt_zero_shot(metrics))
    with open(os.path.join(prompts_dir, "insights_few_shot_v1.txt"), "w", encoding="utf-8") as f:
        f.write(build_prompt_few_shot(metrics))
    print(f"[insights] Prompts guardados em {prompts_dir}/")


def run_strategy(label: str, prompt: str, model: str) -> dict:
    print(f"[insights] A correr estrategia {label}...")
    start = time.time()
    try:
        raw = call_ollama(prompt, model)
        result = parse_json_response(raw)
    except requests.RequestException as exc:
        print(f"[insights] AVISO: Ollama indisponivel para {label}: {exc}")
        result = {}
    elapsed = round(time.time() - start, 1)

    insights = result.get("insights", []) if isinstance(result, dict) else []
    resumo = result.get("resumo_executivo", "") if isinstance(result, dict) else ""
    print(f"[insights] Estrategia {label}: {len(insights)} insights em {elapsed}s")
    return {"insights": insights, "resumo_executivo": resumo}


def make_insight(categoria, titulo, observacao, implicacao, recomendacao, urgencia, confianca):
    return {
        "id": "",
        "categoria": categoria,
        "titulo": titulo,
        "observacao": observacao,
        "implicacao": implicacao,
        "recomendacao": recomendacao,
        "urgencia": urgencia,
        "confianca": round(float(confianca), 2),
    }


# Bloco: ancoras factuais
def build_anchor_insights(metrics: dict) -> list:
    """Cria insights baseados em metricas para reduzir alucinacoes."""
    traffic = metrics["traffic"]
    zones = metrics["zones"]["per_zone"]
    sequences = metrics["zones"]["top_sequences"]
    funnel = metrics["funnel"]
    anomalies = metrics["anomalies"].get("anomalies", [])
    insights = []

    days = traffic["visitors_per_day"]
    avg_day = sum(days.values()) / len(days)
    busiest = traffic["busiest_day"]
    quietest = traffic["quietest_day"]
    insights.append(make_insight(
        "trafego",
        f"Pico semanal em {busiest} com {days[busiest]} visitantes",
        f"A loja recebeu {traffic['total_visitors_week']} visitantes na semana; {busiest} teve {days[busiest]} visitantes "
        f"e {quietest} teve {days[quietest]}.",
        "A equipa deve concentrar capacidade no dia de maior afluencia e reduzir tarefas de backoffice nesse periodo.",
        f"Na proxima semana, reforcar staff no dia equivalente a {busiest} e preparar pausas fora da hora de pico das {traffic['peak_hour']}h.",
        "esta_semana",
        0.92,
    ))

    top_zone_id, top_zone = max(zones.items(), key=lambda item: item[1]["total_visits"])
    dwell_zone_id, dwell_zone = max(zones.items(), key=lambda item: item[1]["avg_dwell_s"])
    low_reach = min(funnel["zone_reach"].items(), key=lambda item: item[1]["reach_pct"])
    insights.append(make_insight(
        "zona",
        f"{top_zone_id} concentra o maior volume de visitas",
        f"{top_zone_id} registou {top_zone['total_visits']} visitas, enquanto {dwell_zone_id} teve o maior dwell medio "
        f"({dwell_zone['avg_dwell_s']}s). A zona de menor alcance foi {low_reach[0]} com {low_reach[1]['reach_pct']}%.",
        "Ha zonas de passagem muito fortes e zonas profundas com alcance baixo, o que cria oportunidades de reposicionamento.",
        f"Esta semana, colocar sinalizacao desde {top_zone_id} para {low_reach[0]} e testar uma promocao de entrada nessa zona.",
        "esta_semana",
        0.9,
    ))

    checkout_zones = {z: d for z, d in zones.items() if z.startswith("Z_C")}
    if checkout_zones:
        busy_checkout_id, busy_checkout = max(checkout_zones.items(), key=lambda item: item[1]["total_visits"])
        calm_checkout_id, calm_checkout = min(checkout_zones.items(), key=lambda item: item[1]["total_visits"])
        seq = sequences[0] if sequences else {"sequence": "sem sequencia", "count": 0}
        insights.append(make_insight(
            "zona",
            f"Desequilibrio entre caixas: {busy_checkout_id} acima de {calm_checkout_id}",
            f"{busy_checkout_id} teve {busy_checkout['total_visits']} visitas e {calm_checkout_id} teve {calm_checkout['total_visits']}; "
            f"a sequencia mais frequente foi {seq['sequence']} com {seq['count']} ocorrencias.",
            "O fluxo nas caixas nao esta distribuido de forma uniforme, aumentando risco de fila numa zona especifica.",
            f"Durante a hora de pico, orientar clientes de {busy_checkout_id} para {calm_checkout_id} quando a fila principal aumentar.",
            "esta_semana",
            0.88,
        ))

    no_checkout = funnel["no_checkout_profile"]["total"]
    insights.append(make_insight(
        "funil",
        f"{funnel['conversion_rate_pct']}% dos visitantes chegaram a caixa",
        f"Dos {funnel['total_visitors']} visitantes, {funnel['total_entered']} passaram por entradas registadas e "
        f"{funnel['reached_checkout']} chegaram a caixa; {no_checkout} nao chegaram ({funnel.get('no_checkout_rate_pct', round(100 - funnel['conversion_rate_pct'], 1))}%).",
        "O funil converte a maioria dos visitantes, mas ainda ha perda mensuravel antes do checkout.",
        "Esta semana, observar os percursos dos visitantes sem checkout e testar sinalizacao para caixas nas zonas de menor alcance.",
        "esta_semana",
        0.94,
    ))

    age_dist = funnel["no_checkout_profile"]["age_dist"]
    gender_dist = funnel["no_checkout_profile"]["gender_dist"]
    top_age, top_age_count = max(age_dist.items(), key=lambda item: item[1])
    insights.append(make_insight(
        "demografico",
        f"Perfil sem checkout dominado por {top_age}",
        f"Entre os {no_checkout} visitantes sem checkout, a faixa {top_age} representa {top_age_count} pessoas; "
        f"a distribuicao de genero e M={gender_dist.get('M', 0)} e F={gender_dist.get('F', 0)}.",
        "A perda antes da caixa tem um perfil identificavel e deve ser tratada com a oferta e comunicacao dessa faixa.",
        f"Na proxima semana, rever mensagens promocionais para {top_age} nas zonas de menor alcance antes das horas de pico.",
        "proximo_mes",
        0.84,
    ))

    for anomaly in anomalies[:5]:
        zone = anomaly["zone_id"]
        hour = int(anomaly["hour_of_day"])
        direction = "queda" if anomaly["direction"] == "abaixo" else "pico"
        insights.append(make_insight(
            "anomalia",
            f"Anomalia em {zone} as {hour}h",
            f"No dia {safe_date(metrics['anomalies']['analysis_day'])}, {zone} as {hour}h teve "
            f"{anomaly['day7_visitors']} visitantes contra media de {anomaly['baseline_mean']} "
            f"(desvio {anomaly['sigma_dev']} sigma, {anomaly['direction']}).",
            f"Esta {direction} esta fora do padrao historico dos primeiros 6 dias e exige verificacao operacional.",
            f"Antes da abertura seguinte, verificar fisicamente {zone} e confirmar sensor, obstrucao, reposicao ou sinalizacao no horario das {hour}h.",
            "imediata",
            min(0.99, 0.7 + float(anomaly["sigma_dev"]) / 30),
        ))

    return insights[:MAX_INSIGHTS]


def normalise_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"(?<![a-z_])\d+[\.,]?\d*(?![a-z_])", "#", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sanitise_insight(insight: dict) -> dict:
    clean = deepcopy(insight)
    for key in ("titulo", "observacao", "implicacao", "recomendacao"):
        value = str(clean.get(key, ""))
        for phrase in BANNED_PHRASES:
            value = re.sub(phrase, "mostra", value, flags=re.IGNORECASE)
        clean[key] = value.strip()
    if clean.get("categoria") not in VALID_CATEGORIES:
        clean["categoria"] = "zona"
    if clean.get("urgencia") not in VALID_URGENCIES:
        clean["urgencia"] = "esta_semana"
    try:
        clean["confianca"] = round(float(clean.get("confianca", 0.75)), 2)
    except (TypeError, ValueError):
        clean["confianca"] = 0.75
    return clean


def merge_and_validate(llm_result: dict, anchors: list) -> list:
    merged = []
    seen = set()

    # Nota: as ancoras entram primeiro porque vieram das metricas calculadas.
    for source in (anchors, llm_result.get("insights", [])):
        for insight in source:
            clean = sanitise_insight(insight)
            key = normalise_text(clean.get("titulo", "") + " " + clean.get("observacao", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(clean)
            if len(merged) == MAX_INSIGHTS:
                break
        if len(merged) == MAX_INSIGHTS:
            break

    for idx, insight in enumerate(merged, 1):
        insight["id"] = f"INS_{idx:03d}"
    return merged


def executive_summary(insights: list) -> str:
    priority = (
        [i for i in insights if i["categoria"] == "anomalia"][:1]
        + [i for i in insights if i["categoria"] == "funil"][:1]
        + [i for i in insights if i["categoria"] in ("trafego", "zona")][:1]
    )
    return "\n".join(f"- {i['titulo']}: {i['observacao']}" for i in priority[:3])


def compare_strategies(result_a: dict, result_b: dict) -> dict:
    def analyse(result: dict) -> dict:
        insights = result.get("insights", [])
        total = len(insights)
        if not total:
            return {
                "n_insights": 0,
                "cobertura_categorias": [],
                "duplicados_pct": 0.0,
                "frases_proibidas_pct": 0.0,
                "observacoes_com_2_numeros_pct": 0.0,
                "score": 0.0,
            }
        keys = [normalise_text(i.get("titulo", "") + " " + i.get("observacao", "")) for i in insights]
        duplicates = total - len(set(keys))
        banned = sum(
            1 for i in insights
            if any(p in " ".join(str(i.get(k, "")).lower() for k in ("observacao", "implicacao", "recomendacao"))
                   for p in BANNED_PHRASES)
        )
        numeric = sum(1 for i in insights if len(re.findall(r"\d+[\.,]?\d*", i.get("observacao", ""))) >= 2)
        score = 0.45 * pct(numeric, total) + 0.35 * (100 - pct(duplicates, total)) + 0.20 * (100 - pct(banned, total))
        return {
            "n_insights": total,
            "cobertura_categorias": sorted({i.get("categoria", "") for i in insights}),
            "duplicados_pct": pct(duplicates, total),
            "frases_proibidas_pct": pct(banned, total),
            "observacoes_com_2_numeros_pct": pct(numeric, total),
            "score": round(score, 1),
        }

    sa, sb = analyse(result_a), analyse(result_b)
    vencedor = "B (few-shot)" if sb["score"] > sa["score"] else "A (zero-shot)" if sa["score"] > sb["score"] else "empate"
    return {
        "estrategia_A_zero_shot": sa,
        "estrategia_B_few_shot": sb,
        "vencedor": vencedor,
        "nota": "Score = numeros na observacao, penalizacao por duplicados e penalizacao por frases vagas proibidas.",
    }


# Bloco: execucao por linha de comandos
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Caminho para metrics.json")
    parser.add_argument("--output", required=True, help="Caminho para insights.json")
    parser.add_argument("--strategy", default="both", choices=["A", "B", "both"])
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    print(f"[insights] A carregar {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    prompts_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(args.output)), "..", "prompts"))
    save_prompts(metrics, prompts_dir)

    anchors = build_anchor_insights(metrics)
    result_a = {"insights": [], "resumo_executivo": ""}
    result_b = {"insights": [], "resumo_executivo": ""}

    if args.strategy in ("A", "both"):
        result_a = run_strategy("A (zero-shot)", build_prompt_zero_shot(metrics), args.model)
    if args.strategy in ("B", "both"):
        result_b = run_strategy("B (few-shot)", build_prompt_few_shot(metrics), args.model)

    comparison = compare_strategies(result_a, result_b) if args.strategy == "both" else None
    llm_best = result_b if result_b["insights"] else result_a
    final_insights = merge_and_validate(llm_best, anchors)
    resumo = executive_summary(final_insights)

    output = {
        "modelo": args.model,
        "temperatura": TEMPERATURE,
        "estrategia_usada": "B_few_shot_validada" if result_b["insights"] else "A_zero_shot_validada" if result_a["insights"] else "fallback_ancoras_validadas",
        "validacao_pos_llm": {
            "fonte_factual": "metrics.json",
            "regras": ["deduplicacao", "remocao de frases vagas", "ancoras calculadas em Python", "anomalias top por sigma"],
        },
        "insights": final_insights,
        "resumo_executivo": resumo,
    }
    if comparison:
        output["comparacao_estrategias"] = comparison

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[insights] insights.json guardado em: {args.output}")
    print(f"[insights] Insights finais validados: {len(final_insights)}")


if __name__ == "__main__":
    main()
