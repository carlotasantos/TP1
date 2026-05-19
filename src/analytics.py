import argparse
import json
import os
from collections import Counter
import pandas as pd


# Leitura dos dados de trajetórias
def load_journeys(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time']  = pd.to_datetime(df['exit_time'])
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    return df


# Cálculo das métricas gerais de tráfego
def compute_traffic_metrics(df: pd.DataFrame) -> dict:
    # Conta quantos visitantes únicos houve em cada dia.
    visitors_per_day = (
        df.groupby('visit_date')['person_id']
        .nunique()
        .reset_index()
        .rename(columns={'person_id': 'visitors'})
    )
    visitors_per_day['visit_date'] = visitors_per_day['visit_date'].dt.strftime('%Y-%m-%d')
    visitors_per_day_dict = dict(zip(
        visitors_per_day['visit_date'],
        visitors_per_day['visitors'].tolist()
    ))

    # Conta quantos visitantes únicos passaram pela loja em cada hora da semana.
    visitors_per_hour = (
        df.groupby('hour_of_day')['person_id']
        .nunique()
        .reset_index()
        .rename(columns={'person_id': 'visitors'})
    )
    visitors_per_hour_dict = {
        int(row['hour_of_day']): int(row['visitors'])
        for _, row in visitors_per_hour.iterrows()
    }

    # Calcula quanto tempo cada visitante ficou, da primeira entrada até à última saída.
    visit_duration = df.groupby('person_id').agg(
        first_entry=('entry_time', 'min'),
        last_exit=('exit_time', 'max')
    )
    visit_duration['duration_min'] = (
        (visit_duration['last_exit'] - visit_duration['first_entry'])
        .dt.total_seconds() / 60
    )
    avg_visit_min = round(float(visit_duration['duration_min'].mean()), 1)
    median_visit_min = round(float(visit_duration['duration_min'].median()), 1)

    # Total de visitantes sintéticos identificados ao longo da semana.
    total_visitors = int(df['person_id'].nunique())

    # Identifica o dia com mais movimento e o dia com menos movimento.
    busiest_day   = max(visitors_per_day_dict, key=visitors_per_day_dict.get)
    quietest_day  = min(visitors_per_day_dict, key=visitors_per_day_dict.get)

    # Encontra a hora em que houve maior afluência.
    peak_hour = int(max(visitors_per_hour_dict, key=visitors_per_hour_dict.get))

    return {
        'total_visitors_week':    total_visitors,
        'visitors_per_day':       visitors_per_day_dict,
        'visitors_per_hour':      visitors_per_hour_dict,
        'avg_visit_duration_min': avg_visit_min,
        'median_visit_duration_min': median_visit_min,
        'busiest_day':            busiest_day,
        'quietest_day':           quietest_day,
        'peak_hour':              peak_hour,
    }


# Análise das métricas por zona
def compute_zone_metrics(df: pd.DataFrame) -> dict:
    """Resume as visitas, as paragens e os percursos mais comuns em cada zona."""

    zones = {}
    for zone, group in df.groupby('zone_id'):
        total_visits  = len(group)
        linger_visits = group[group['dwell_s'] > 0]
        stop_rate     = round(100 * len(linger_visits) / max(total_visits, 1), 1)

        avg_dwell    = round(float(linger_visits['dwell_s'].mean()), 1) if len(linger_visits) > 0 else 0.0
        median_dwell = round(float(linger_visits['dwell_s'].median()), 1) if len(linger_visits) > 0 else 0.0

        zones[zone] = {
            'total_visits':     int(total_visits),
            'avg_dwell_s':      avg_dwell,
            'median_dwell_s':   median_dwell,
            'stop_rate_pct':    stop_rate,
        }

    # Cria sequências simples com base nas zonas visitadas por cada pessoa.
    transitions = []
    for _, person_df in df.groupby('person_id'):
        person_df = person_df.sort_values('entry_time')
        zone_seq = person_df['zone_id'].tolist()
        for i in range(len(zone_seq) - 1):
            transitions.append(f"{zone_seq[i]} → {zone_seq[i+1]}")

    top_sequences = [
        {'sequence': seq, 'count': count}
        for seq, count in Counter(transitions).most_common(10)
    ]

    return {
        'per_zone':       zones,
        'top_sequences':  top_sequences,
    }


# Análise do funil de clientes
def compute_funnel_metrics(df: pd.DataFrame) -> dict:
    """Calcula quantos visitantes entram na loja e quantos chegam às caixas."""

    entrance_zones = {'Z_E1', 'Z_E2'}
    checkout_zones = {'Z_C1', 'Z_C2', 'Z_C3', 'Z_CK'}

    # Usa todos os visitantes únicos como ponto de partida do funil.
    all_persons      = set(df['person_id'].unique())
    total_visitors   = len(all_persons)

    # Identifica os visitantes que passaram por uma zona de entrada.
    entrance_persons = set(df[df['zone_id'].isin(entrance_zones)]['person_id'])
    total_entered    = len(entrance_persons)

    # Identifica os visitantes que chegaram a pelo menos uma zona de caixa.
    checkout_persons = set(df[df['zone_id'].isin(checkout_zones)]['person_id'])
    conversion_rate  = round(100 * len(checkout_persons) / max(total_visitors, 1), 1)

    # Mede o alcance de cada zona dentro da loja.
    zone_reach = {}
    for zone in sorted(df['zone_id'].unique()):
        visitors_in_zone = set(df[df['zone_id'] == zone]['person_id'])
        zone_reach[zone] = {
            'visitors':  int(len(visitors_in_zone)),
            'reach_pct': round(100 * len(visitors_in_zone) / max(total_visitors, 1), 1),
        }

    # Resume o perfil dos visitantes que não chegaram à caixa.
    no_checkout    = all_persons - checkout_persons
    no_checkout_df = df[df['person_id'].isin(no_checkout)].drop_duplicates('person_id')

    gender_dist = no_checkout_df['gender'].value_counts().to_dict()
    age_dist    = no_checkout_df['age_range'].value_counts().to_dict()

    return {
        'total_visitors':        int(total_visitors),
        'total_entered':         int(total_entered),
        'reached_checkout':      int(len(checkout_persons)),
        'conversion_rate_pct':   conversion_rate,
        'no_checkout_rate_pct':  round(100 - conversion_rate, 1),
        'zone_reach':            zone_reach,
        'no_checkout_profile': {
            'total':       int(len(no_checkout)),
            'gender_dist': {k: int(v) for k, v in gender_dist.items()},
            'age_dist':    {k: int(v) for k, v in age_dist.items()},
        },
        'note': (
            'O stitcher cria nova track a cada entrada em Z_E, pelo que '
            'conversion_rate é calculada sobre o total de visitantes únicos.'
        ),
    }


# Análise da segmentação demográfica
def compute_demographic_metrics(df: pd.DataFrame) -> dict:
    """Resume género, idade e tempo médio de permanência por segmento."""

    # Evita contar a mesma pessoa mais do que uma vez na mesma hora.
    person_hour = df.drop_duplicates(subset=['person_id', 'hour_of_day'])

    gender_by_hour = {}
    for hour, group in person_hour.groupby('hour_of_day'):
        counts = group['gender'].value_counts().to_dict()
        gender_by_hour[int(hour)] = {k: int(v) for k, v in counts.items()}

    age_by_hour = {}
    for hour, group in person_hour.groupby('hour_of_day'):
        counts = group['age_range'].value_counts().to_dict()
        age_by_hour[int(hour)] = {k: int(v) for k, v in counts.items()}

    # Calcula o tempo médio de permanência por género e zona.
    linger_df = df[df['dwell_s'] > 0]
    dwell_by_gender_zone = {}
    for (gender, zone), group in linger_df.groupby(['gender', 'zone_id']):
        key = f"{gender}_{zone}"
        dwell_by_gender_zone[key] = round(float(group['dwell_s'].mean()), 1)

    # Calcula o tempo médio de permanência por faixa etária e zona.
    dwell_by_age_zone = {}
    for (age, zone), group in linger_df.groupby(['age_range', 'zone_id']):
        key = f"{age}_{zone}"
        dwell_by_age_zone[key] = round(float(group['dwell_s'].mean()), 1)

    return {
        'gender_by_hour':       gender_by_hour,
        'age_by_hour':          age_by_hour,
        'dwell_by_gender_zone': dwell_by_gender_zone,
        'dwell_by_age_zone':    dwell_by_age_zone,
    }


# Deteção de anomalias
def compute_anomaly_metrics(df: pd.DataFrame) -> dict:
    """Compara o último dia com os seis dias anteriores para identificar desvios relevantes."""

    dates = sorted(df['visit_date'].unique())
    if len(dates) < 7:
        return {'error': 'Menos de 7 dias no dataset — anomalias não calculadas.'}

    baseline_dates = dates[:6]
    day7           = dates[6]

    # Prepara o tráfego por zona, hora e dia.
    traffic = (
        df.groupby(['zone_id', 'hour_of_day', 'visit_date'])['person_id']
        .nunique()
        .reset_index()
        .rename(columns={'person_id': 'visitors'})
    )

    # Usa os primeiros seis dias como referência do comportamento normal.
    baseline = traffic[traffic['visit_date'].isin(baseline_dates)]
    baseline_stats = (
        baseline.groupby(['zone_id', 'hour_of_day'])['visitors']
        .agg(['mean', 'std'])
        .reset_index()
    )
    baseline_stats['std'] = baseline_stats['std'].fillna(0)

    # Também inclui horas sem registos no dia analisado.
    day7_traffic = traffic[traffic['visit_date'] == day7]
    merged = pd.merge(
        baseline_stats[['zone_id', 'hour_of_day']],
        day7_traffic[['zone_id', 'hour_of_day', 'visitors']],
        on=['zone_id', 'hour_of_day'],
        how='left')
    merged['visitors'] = merged['visitors'].fillna(0).astype(int)
    merged = pd.merge(merged, baseline_stats, on=['zone_id', 'hour_of_day'], how='left')
    merged['std'] = merged['std'].fillna(0)
    merged['mean'] = merged['mean'].fillna(0)

    # Mede o afastamento em relação ao comportamento habitual.
    merged['sigma_dev'] = merged.apply(
        lambda r: abs(r['visitors'] - r['mean']) / r['std']
        if r['std'] > 0 else 0.0,
        axis=1
    )

    anomalies = merged[merged['sigma_dev'] > 2].copy()
    anomalies = anomalies.sort_values('sigma_dev', ascending=False)

    anomaly_list = []
    for _, row in anomalies.iterrows():
        direction = 'acima' if row['visitors'] > row['mean'] else 'abaixo'
        anomaly_list.append({
            'zone_id':       row['zone_id'],
            'hour_of_day':   int(row['hour_of_day']),
            'day7_visitors': int(row['visitors']),
            'baseline_mean': round(float(row['mean']), 1),
            'baseline_std':  round(float(row['std']), 1),
            'sigma_dev':     round(float(row['sigma_dev']), 2),
            'direction':     direction,
        })

    return {
        'baseline_days':   [pd.Timestamp(d).strftime('%Y-%m-%d') for d in baseline_dates],
        'analysis_day':    pd.Timestamp(day7).strftime('%Y-%m-%d'),
        'total_anomalies': len(anomaly_list),
        'anomalies':       anomaly_list,
    }


# Execução através da linha de comandos
def main():
    parser = argparse.ArgumentParser(
        description='Analytics: calcula métricas a partir do journeys.csv.'
    )
    parser.add_argument('--input',  required=True, help='Caminho para journeys.csv')
    parser.add_argument('--output', required=True, help='Caminho para metrics.json')
    args = parser.parse_args()

    print(f'[analytics] A carregar {args.input}...')
    df = load_journeys(args.input)
    print(f'[analytics] {len(df):,} linhas carregadas ({df["person_id"].nunique():,} pessoas).')

    print('[analytics] A calcular métricas de tráfego...')
    traffic = compute_traffic_metrics(df)

    print('[analytics] A calcular métricas por zona...')
    zones = compute_zone_metrics(df)

    print('[analytics] A calcular funil de clientes...')
    funnel = compute_funnel_metrics(df)

    print('[analytics] A calcular segmentação demográfica...')
    demographics = compute_demographic_metrics(df)

    print('[analytics] A calcular anomalias do dia 7...')
    anomalies = compute_anomaly_metrics(df)

    metrics = {
        'traffic':      traffic,
        'zones':        zones,
        'funnel':       funnel,
        'demographics': demographics,
        'anomalies':    anomalies,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f'[analytics] metrics.json guardado em: {args.output}')
    print(f'[analytics] Anomalias detetadas no dia 7: {anomalies.get("total_anomalies", "N/A")}')
    print(f'[analytics] Taxa de conversão para caixa: {funnel["conversion_rate_pct"]}%')
    print(f'[analytics] Visitantes únicos na semana: {traffic["total_visitors_week"]:,}')


if __name__ == '__main__':
    main()
