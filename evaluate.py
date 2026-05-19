import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd


# Bloco: execucao dos modulos do pipeline
def run_module(script_path: str, extra_args: list) -> None:
    cmd = [sys.executable, script_path] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end='')
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(
            f"O módulo {os.path.basename(script_path)} terminou com erro (código {result.returncode})."
        )


def compute_phase2_metrics(metrics_path: str, insights_path: str, report_path: str, ground_truth_path: str = None) -> dict:
    """Calcula as metricas automaticas da fase dos insights e do report."""
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    with open(insights_path, 'r', encoding='utf-8') as f:
        insights_data = json.load(f)
    with open(report_path, 'r', encoding='utf-8') as f:
        report_text = f.read()

    def extract_numbers(obj, out=None):
        if out is None:
            out = set()
        if isinstance(obj, dict):
            for v in obj.values():
                extract_numbers(v, out)
        elif isinstance(obj, list):
            for item in obj:
                extract_numbers(item, out)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            out.add(str(round(float(obj), 1)))
            if obj == int(obj):
                out.add(str(int(obj)))
        return out

    known_numbers = extract_numbers(metrics)

    def count_verified(text: str):
        nums = re.findall(r'\b\d+(?:[.,]\d+)?\b', text)
        verified = sum(
            1 for n in nums
            if n.replace(',', '.') in known_numbers
            or (n.replace(',', '.').split('.')[0]) in known_numbers
        )
        return len(nums), verified

    insight_text = ' '.join(
        ' '.join([
            ins.get('observacao', ''),
            ins.get('implicacao', ''),
            ins.get('recomendacao', ''),
        ])
        for ins in insights_data.get('insights', [])
    )
    total_ins, verified_ins = count_verified(insight_text)
    precision_pct = round(100 * verified_ins / total_ins, 1) if total_ins else None

    total_rep, verified_rep = count_verified(report_text)
    hallucination_pct = round(100 * verified_rep / total_rep, 1) if total_rep else None

    return {
        'numerical_precision_pct': precision_pct,
        'total_insight_numbers': total_ins,
        'verified_insight_numbers': verified_ins,
        'hallucination_absence_pct': hallucination_pct,
        'total_report_numbers': total_rep,
        'verified_report_numbers': verified_rep,
        'anomaly_detection_pct': _compute_anomaly_detection(insights_path, ground_truth_path),
    }


def _compute_anomaly_detection(insights_path: str, ground_truth_path: str) -> object:
    """Confirma se os insights mencionam a zona e a hora das anomalias conhecidas."""
    if not ground_truth_path or not os.path.exists(ground_truth_path):
        return 'N/A — ground_truth.json nao fornecido (usar --ground-truth)'

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    with open(insights_path, 'r', encoding='utf-8') as f:
        insights_data = json.load(f)

    anomaly_insights = [
        i for i in insights_data.get('insights', [])
        if i.get('categoria') == 'anomalia'
    ]
    insight_text = ' '.join(
        ' '.join([i.get('titulo',''), i.get('observacao',''), i.get('implicacao',''), i.get('recomendacao','')])
        for i in anomaly_insights
    ).lower()

    total = 0
    detected = 0
    detection_details = []

    for ano in gt.get('anomalias', []):
        zona = ano.get('zona', '')
        # Nota: algumas anomalias podem ocupar mais do que uma hora.
        horas = ano.get('horas', [ano.get('hora')] if ano.get('hora') is not None else [])

        for hora in horas:
            total += 1
            # Nota: a avaliacao textual exige zona e hora no insight.
            zona_found = zona.lower() in insight_text
            hora_found = str(hora) in insight_text
            found = zona_found and hora_found
            if found:
                detected += 1
            detection_details.append({
                'anomaly_id': ano['id'],
                'zona': zona,
                'hora': hora,
                'detectada': found,
                'zona_mencionada': zona_found,
                'hora_mencionada': hora_found,
            })

    pct = round(100 * detected / total, 1) if total > 0 else 0.0
    return {
        'pct': pct,
        'detectadas': detected,
        'total': total,
        'detalhes': detection_details,
    }


# Bloco: execucao por linha de comandos
def main():
    parser = argparse.ArgumentParser(
        description='Harness de avaliação end-to-end — LIACD TP1.'
    )
    parser.add_argument('--data',    required=True, help='Caminho para events_validation.csv')
    parser.add_argument('--output',  required=True, help='Caminho para evaluation_report.json')
    parser.add_argument('--workdir',       default=None,  help='Diretório de trabalho (default: temporário)')
    parser.add_argument('--ground-truth', default=None,  help='Caminho para ground_truth.json (opcional)')
    args = parser.parse_args()

    src_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    work_dir = args.workdir or tempfile.mkdtemp(prefix='evaluate_')
    os.makedirs(work_dir, exist_ok=True)

    journeys_path = os.path.join(work_dir, 'journeys.csv')
    metrics_path  = os.path.join(work_dir, 'metrics.json')
    insights_path = os.path.join(work_dir, 'insights.json')
    report_path   = os.path.join(work_dir, 'weekly_report.md')

    print(f"\n{'='*60}")
    print("HARNESS DE AVALIAÇÃO — LIACD TP1")
    print(f"Dataset : {args.data}")
    print(f"Workdir : {work_dir}")
    print(f"{'='*60}\n")

    # Nota: o stitcher e importado para obter as metricas internas exactas.
    print("[1/4] stitcher.py — Reconstrução de trajectórias")
    sys.path.insert(0, src_dir)
    from stitcher import build_journeys, compute_quality_metrics, load_zones, precompute_walk_table

    df = pd.read_csv(args.data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  {len(df):,} eventos carregados.")

    zones_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'zones.json')
    walk_table = {}
    if os.path.exists(zones_path):
        walk_times = load_zones(zones_path)
        walk_table = precompute_walk_table(walk_times)

    journeys, attributed_events = build_journeys(df, walk_table)
    quality = compute_quality_metrics(journeys, df, attributed_events)
    journeys.to_csv(journeys_path, index=False)
    print(f"  journeys.csv guardado ({len(journeys):,} linhas).\n")

    # Nota: os restantes modulos correm como scripts independentes.
    for step_num, (module, in_path, out_path, extra) in enumerate([
        ('analytics', journeys_path, metrics_path,  []),
        ('insights',  metrics_path,  insights_path, ['--strategy', 'B']),
        ('report',    insights_path, report_path,   []),
    ], start=2):
        print(f"[{step_num}/4] {module}.py")
        run_module(os.path.join(src_dir, f'{module}.py'), ['--input', in_path, '--output', out_path] + extra)
        print()

    # Nota: calcula a qualidade dos textos produzidos na fase 2.
    gt_path = args.ground_truth if hasattr(args, 'ground_truth') else None
    phase2 = compute_phase2_metrics(metrics_path, insights_path, report_path, gt_path)

    # Nota: junta tudo num ficheiro JSON de avaliacao.
    evaluation = {
        'dataset': args.data,
        'pipeline_status': 'OK',
        'phase1': {
            'consistency_pct':           quality['consistency_pct'],
            'trajectories_with_overlap': quality['trajectories_with_overlap'],
            'coverage_pct':              quality['coverage_pct'],
            'attributed_events':         quality['attributed_events'],
            'total_events':              quality['total_events'],
            'completeness_pct':          quality['completeness_pct'],
            'complete_trajectories':     quality['complete_trajectories'],
            'total_trajectories':        quality['total_trajectories'],
            'gap_stats':                 quality.get('gap_stats', {}),
        },
        'phase2': phase2,
        'output_files': {
            'journeys': journeys_path,
            'metrics':  metrics_path,
            'insights': insights_path,
            'report':   report_path,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("RESULTADOS")
    print(f"{'='*60}")
    p1 = evaluation['phase1']
    print(f"  Consistência : {p1['consistency_pct']}%  ({p1['trajectories_with_overlap']} trajectórias com sobreposição)")
    print(f"  Cobertura    : {p1['coverage_pct']}%  ({p1['attributed_events']:,} / {p1['total_events']:,} eventos)")
    print(f"  Completude   : {p1['completeness_pct']}%  ({p1['complete_trajectories']} / {p1['total_trajectories']} trajectórias)")
    if p1['gap_stats']:
        g = p1['gap_stats']
        print(f"  Gap médio    : {g['mean_s']}s  |  mediana: {g['median_s']}s  |  p95: {g['p95_s']}s")
    p2 = evaluation['phase2']
    print(f"\n  Precisão numérica (insights) : {p2['numerical_precision_pct']}%")
    print(f"  Ausência de alucinação       : {p2['hallucination_absence_pct']}%")
    ano_result = p2['anomaly_detection_pct']
    if isinstance(ano_result, dict):
        print(f"  Deteção de anomalias         : {ano_result['pct']}%  ({ano_result['detectadas']}/{ano_result['total']} anomalias detectadas)")
        for d in ano_result['detalhes']:
            status = 'OK' if d['detectada'] else 'NAO DETECTADA'
            print(f"    {status}: {d['anomaly_id']} — {d['zona']} às {d['hora']}h")
    else:
        print(f"  Deteção de anomalias         : {ano_result}")
    print(f"\n  Relatório guardado em: {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
