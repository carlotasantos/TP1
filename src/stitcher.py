import argparse
import heapq
import pandas as pd
import os
import json
from typing import Optional

# Bloco: configuracao do modulo
# Estes valores definem as regras principais do stitching.
MAX_GAP_S = 300
WALK_TOLERANCE_S = 20
EXIT_ZONES = {'Z_E1', 'Z_E2', 'Z_CK'}
ENTRANCE_ZONES = {'Z_E1', 'Z_E2'}


# Bloco: leitura do mapa de zonas
def load_zones(zones_path: str) -> dict:
    with open(zones_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    walk_times = {}
    for zone_id, zone_info in data['zones'].items():
        walk_times[zone_id] = zone_info.get('walk_seconds', {})
    return walk_times


def _dijkstra(start: str, walk_times: dict) -> dict:
    heap = [(0.0, start)]
    dist = {}
    while heap:
        cost, zone = heapq.heappop(heap)
        if zone in dist:
            continue
        dist[zone] = cost
        for neighbor, edge_cost in walk_times.get(zone, {}).items():
            if neighbor not in dist:
                heapq.heappush(heap, (cost + edge_cost, neighbor))
    return dist


def precompute_walk_table(walk_times: dict) -> dict:
    """Guarda antecipadamente os tempos entre zonas para acelerar o matching."""
    table = {}
    for zone in walk_times:
        dist = _dijkstra(zone, walk_times)
        for target, d in dist.items():
            table[(zone, target)] = d
    return table


# Estado de uma trajectoria
class Track:
    """Representa uma pessoa enquanto a trajectoria ainda esta a ser reconstruida."""
    __slots__ = [
        'person_id', 'visits',
        'current_zone', 'current_entry_time', 'current_dwell_s',
        'last_exit_time', 'last_exit_s', 'last_zone',
        'gender_votes', 'age_votes',
    ]

    def __init__(self, person_id: str, gender: str, age_range: str):
        self.person_id = person_id
        self.visits = []
        self.current_zone = None
        self.current_entry_time = None
        self.current_dwell_s = 0
        self.last_exit_time = None
        self.last_exit_s = None    # timestamp em segundos Unix (float) para comparações rápidas
        self.last_zone = None
        self.gender_votes = {gender: 1}
        self.age_votes = {age_range: 1}

    def dominant_gender(self) -> str:
        return max(self.gender_votes, key=self.gender_votes.get)

    def dominant_age(self) -> str:
        return max(self.age_votes, key=self.age_votes.get)

    def _vote(self, gender: str, age_range: str):
        self.gender_votes[gender] = self.gender_votes.get(gender, 0) + 1
        self.age_votes[age_range] = self.age_votes.get(age_range, 0) + 1

    def open_entry(self, zone: str, ts, gender: str, age_range: str):
        self.current_zone = zone
        self.current_entry_time = ts
        self.current_dwell_s = 0
        self._vote(gender, age_range)

    def update_linger(self, dwell_s: int, gender: str, age_range: str):
        self.current_dwell_s = dwell_s
        self._vote(gender, age_range)

    def close_exit(self, ts, ts_s: float, gender: str, age_range: str):
        self.visits.append({
            'zone_id':    self.current_zone,
            'entry_time': self.current_entry_time,
            'exit_time':  ts,
            'dwell_s':    self.current_dwell_s,
        })
        self.last_exit_time = ts
        self.last_exit_s = ts_s
        self.last_zone = self.current_zone
        self.current_zone = None
        self.current_entry_time = None
        self.current_dwell_s = 0
        self._vote(gender, age_range)

    def force_close(self, ts, ts_s: float):
        if self.current_zone is not None:
            self.visits.append({
                'zone_id':    self.current_zone,
                'entry_time': self.current_entry_time,
                'exit_time':  ts,
                'dwell_s':    self.current_dwell_s,
            })
            self.last_exit_time = ts
            self.last_exit_s = ts_s
            self.last_zone = self.current_zone
            self.current_zone = None


# Bloco: regras para ligar eventos a trajectorias
def _match_score(track: Track, gender: str, age_range: str,
                 gap_s: float, to_zone: str, walk_table: dict) -> Optional[float]:
    """Calcula se uma entrada pode pertencer a uma trajectoria em espera."""
    if gap_s < 0 or gap_s > MAX_GAP_S:
        return None

    if track.last_zone and walk_table:
        min_walk = walk_table.get((track.last_zone, to_zone), 0)
        if gap_s < min_walk - WALK_TOLERANCE_S:
            return None

    dom_gender = track.dominant_gender()
    dom_age    = track.dominant_age()
    gender_match = dom_gender == gender
    age_match    = dom_age    == age_range

    total_gender_votes = sum(track.gender_votes.values())
    if not gender_match and total_gender_votes >= 3:
        return None

    gender_score = 10.0 if gender_match else 2.0
    age_score    = 5.0  if age_match    else 0.0
    time_score   = 1.0 - (gap_s / MAX_GAP_S)
    return gender_score + age_score + time_score


def _best_active(tracks: list, gender: str, age_range: str) -> Optional[Track]:
    """Escolhe a trajectoria activa mais provavel dentro da mesma zona."""
    best, best_score = None, -1
    for t in tracks:
        gender_match = t.dominant_gender() == gender
        age_match = t.dominant_age() == age_range
        attr_score = (2 if gender_match else 0) + (1 if age_match else 0)
        if attr_score > best_score:
            best_score, best = attr_score, t
        elif attr_score == best_score and best is not None:
            if (t.current_entry_time is not None and best.current_entry_time is not None
                    and t.current_entry_time < best.current_entry_time):
                best = t
    return best


# reconstrucao das trajectorias
def build_journeys(df: pd.DataFrame, walk_table: dict) -> tuple:
    """Percorre os eventos por tempo e decide a que pessoa sintetica pertencem."""

    # ordenar por timestamp evita depender da ordem dos event_id.
    df = df.sort_values('timestamp').reset_index(drop=True)
    ts_seconds = (df['timestamp'].values.astype('int64') / 1e9)

    #entradas iguais no mesmo segundo sao tratadas como pessoas diferentes.
    entry_df = df[df['event_type'] == 'entry']
    _counts = entry_df.groupby(['timestamp', 'zone_id', 'gender', 'age_range']).size()
    simultaneous_keys: set = set(_counts[_counts > 1].index)

    active_in_zone: dict = {}
    idle_tracks: list = []
    all_tracks: list = []

    person_counter = 0
    attributed = 0
    orphans = 0
    forced_closes = 0
    gender_mismatches = 0
    age_mismatches = 0

    records = df.to_dict('records')

    for idx, row in enumerate(records):
        ts = row['timestamp']
        ts_s = ts_seconds[idx]
        zone = row['zone_id']
        etype = row['event_type']
        gender = row['gender']
        age_range = row['age_range']
        dwell_s = row['duration_s']

        # trajectorias paradas ha demasiado tempo deixam de ser candidatas.
        if etype == 'entry':
            still_idle = []
            for t in idle_tracks:
                if ts_s - t.last_exit_s > MAX_GAP_S:
                    all_tracks.append(t)
                else:
                    still_idle.append(t)
            idle_tracks = still_idle

            # entradas simultaneas nao reutilizam trajectorias antigas.
            is_simultaneous = (ts, zone, gender, age_range) in simultaneous_keys

            # uma entrada pela porta cria sempre uma nova pessoa sintetica.
            is_entrance = zone in ENTRANCE_ZONES

            # fora das portas, tenta-se continuar a trajectoria mais plausivel.
            best_idx, best_score = -1, -1
            if not is_simultaneous and not is_entrance:
                for i, t in enumerate(idle_tracks):
                    gap_s = ts_s - t.last_exit_s
                    s = _match_score(t, gender, age_range, gap_s, zone, walk_table)
                    if s is not None and s > best_score:
                        best_score, best_idx = s, i

            if best_idx >= 0:
                track = idle_tracks.pop(best_idx)
            else:
                person_counter += 1
                track = Track(f"P_{person_counter:05d}", gender, age_range)

            track.open_entry(zone, ts, gender, age_range)
            active_in_zone.setdefault(zone, []).append(track)
            attributed += 1

        elif etype == 'linger':
            zone_tracks = active_in_zone.get(zone, [])
            t = _best_active(zone_tracks, gender, age_range)
            if t:
                if t.dominant_gender() != gender:
                    gender_mismatches += 1
                if t.dominant_age() != age_range:
                    age_mismatches += 1
                t.update_linger(dwell_s, gender, age_range)
                attributed += 1
            else:
                orphans += 1

        elif etype == 'exit':
            zone_tracks = active_in_zone.get(zone, [])
            t = _best_active(zone_tracks, gender, age_range)
            if t:
                if t.dominant_gender() != gender:
                    gender_mismatches += 1
                if t.dominant_age() != age_range:
                    age_mismatches += 1
                active_in_zone[zone].remove(t)
                t.close_exit(ts, ts_s, gender, age_range)
                if zone == 'Z_CK':
                    # Nota: sair pelo checkout fecha a visita da pessoa.
                    all_tracks.append(t)
                elif zone in ENTRANCE_ZONES:
                    # Nota: a porta pode ser entrada inicial ou saida final.
                    has_interior = any(
                        v['zone_id'] not in ENTRANCE_ZONES for v in t.visits
                    )
                    if has_interior:
                        all_tracks.append(t)
                    else:
                        idle_tracks.append(t)
                else:
                    idle_tracks.append(t)
                attributed += 1
            else:
                orphans += 1

    # Visitas abertas no fim do ficheiro sao fechadas no ultimo timestamp.
    last_ts = df['timestamp'].max()
    last_ts_s = ts_seconds[-1]
    for tracks in active_in_zone.values():
        for t in tracks:
            t.force_close(last_ts, last_ts_s)
            forced_closes += 1
            all_tracks.append(t)
    all_tracks.extend(idle_tracks)

    total_attributed = attributed
    print(f"  [stitcher] Trajectórias reconstruídas: {person_counter}")
    print(f"  [stitcher] Eventos atribuídos: {attributed} / {len(df)}")
    print(f"  [stitcher] Eventos órfãos (sem track compatível): {orphans}")
    print(f"  [stitcher] Visitas fechadas forçadamente (exit em falta): {forced_closes}")
    print(f"  [stitcher] Mismatches de género absorvidos: {gender_mismatches} ({100*gender_mismatches/max(total_attributed,1):.1f}%)")
    print(f"  [stitcher] Mismatches de idade absorvidos:  {age_mismatches} ({100*age_mismatches/max(total_attributed,1):.1f}%)")

    # Nota: cada linha final representa uma pessoa numa zona.
    rows = []
    for track in all_tracks:
        if not track.visits:
            continue
        dom_g = track.dominant_gender()
        dom_a = track.dominant_age()
        for v in track.visits:
            entry_t = v['entry_time']
            exit_t = v['exit_time'] or entry_t
            rows.append({
                'person_id':   track.person_id,
                'zone_id':     v['zone_id'],
                'entry_time':  entry_t,
                'exit_time':   exit_t,
                'dwell_s':     v['dwell_s'],
                'gender':      dom_g,
                'age_range':   dom_a,
                'visit_date':  str(entry_t.date()),
                'hour_of_day': entry_t.hour,
            })

    return pd.DataFrame(rows), attributed


# Metricas de qualidade da reconstrucao
def compute_quality_metrics(journeys: pd.DataFrame, original_df: pd.DataFrame, attributed_events: int) -> dict:
    """Calcula as metricas pedidas para avaliar se o stitching e coerente."""
    metrics = {}

    total_trajectories = journeys['person_id'].nunique()

    # Uma pessoa nao pode estar em duas zonas ao mesmo tempo.
    trajectories_with_violation = 0
    for _, group in journeys.groupby('person_id'):
        group = group.sort_values('entry_time')
        for i in range(len(group) - 1):
            if group.iloc[i]['exit_time'] > group.iloc[i + 1]['entry_time']:
                trajectories_with_violation += 1
                break

    consistency_pct = 100 * (1 - trajectories_with_violation / max(total_trajectories, 1))
    metrics['consistency_pct'] = round(consistency_pct, 2)
    metrics['trajectories_with_overlap'] = trajectories_with_violation

    # Mede quantos eventos foram aproveitados pelo algoritmo.
    coverage_pct = 100 * attributed_events / max(len(original_df), 1)
    metrics['coverage_pct'] = round(coverage_pct, 2)
    metrics['attributed_events'] = attributed_events
    metrics['total_events'] = len(original_df)

    # Mede quantas trajectorias parecem visitas completas.
    entrance_zones = {'Z_E1', 'Z_E2'}
    exit_zones = {'Z_E1', 'Z_E2', 'Z_CK'}
    complete = 0
    for _, group in journeys.groupby('person_id'):
        group = group.sort_values('entry_time')
        if group.iloc[0]['zone_id'] in entrance_zones and group.iloc[-1]['zone_id'] in exit_zones:
            complete += 1

    completeness_pct = 100 * complete / max(total_trajectories, 1)
    metrics['completeness_pct'] = round(completeness_pct, 2)
    metrics['complete_trajectories'] = complete
    metrics['total_trajectories'] = total_trajectories

    # Nota: mede se os intervalos entre zonas parecem razoaveis.
    gaps = []
    for _, group in journeys.groupby('person_id'):
        group = group.sort_values('entry_time')
        for i in range(len(group) - 1):
            gap = (group.iloc[i + 1]['entry_time'] - group.iloc[i]['exit_time']).total_seconds()
            gaps.append(gap)

    if gaps:
        gaps_s = pd.Series(gaps)
        metrics['gap_stats'] = {
            'mean_s':        round(float(gaps_s.mean()), 1),
            'median_s':      round(float(gaps_s.median()), 1),
            'p95_s':         round(float(gaps_s.quantile(0.95)), 1),
            'max_s':         round(float(gaps_s.max()), 1),
            'pct_over_300s': round(100 * (gaps_s > 300).mean(), 2),
            'pct_negative':  round(100 * (gaps_s < 0).mean(), 2),
        }

    return metrics


# Execução por linha de comandos
def main():
    parser = argparse.ArgumentParser(
        description='Stitcher: reconstrói trajectórias a partir de eventos anónimos.'
    )
    parser.add_argument('--input',  required=True, help='Caminho para events.csv')
    parser.add_argument('--output', required=True, help='Caminho para journeys.csv')
    parser.add_argument('--zones',  default=None,  help='Caminho para zones.json (opcional)')
    args = parser.parse_args()

    if args.zones is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, '..', 'data', 'zones.json'),
            os.path.join(script_dir, '..', 'zones.json'),
            'zones.json',
        ]
        for c in candidates:
            if os.path.exists(c):
                args.zones = c
                break

    print(f"[stitcher] A carregar {args.input}...")
    df = pd.read_csv(args.input)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"[stitcher] {len(df):,} eventos carregados.")

    walk_table = {}
    if args.zones and os.path.exists(args.zones):
        print(f"[stitcher] A carregar mapa de zonas: {args.zones}")
        walk_times = load_zones(args.zones)
        walk_table = precompute_walk_table(walk_times)
        print(f"[stitcher] {len(walk_table)} pares de zonas pré-computados.")
    else:
        print("[stitcher] AVISO: zones.json não encontrado. Validação física de transições desativada.")

    print("[stitcher] A reconstruir trajectórias...")
    journeys, attributed_events = build_journeys(df, walk_table)

    print("[stitcher] A calcular métricas de qualidade...")
    metrics = compute_quality_metrics(journeys, df, attributed_events)

    print("\n=== MÉTRICAS DE QUALIDADE ===")
    print(f"  Consistência (sem sobreposição):       {metrics['consistency_pct']}%  ({metrics['trajectories_with_overlap']} trajectórias com sobreposição)")
    print(f"  Cobertura (eventos atribuídos):        {metrics['coverage_pct']}%  ({metrics['attributed_events']:,} / {metrics['total_events']:,})")
    print(f"  Completude (início Z_E, fim Z_E/Z_CK): {metrics['completeness_pct']}%  ({metrics['complete_trajectories']} / {metrics['total_trajectories']})")
    if 'gap_stats' in metrics:
        g = metrics['gap_stats']
        print(f"  Gap médio entre zonas:  {g['mean_s']}s")
        print(f"  Gap mediano:            {g['median_s']}s")
        print(f"  Gap p95:                {g['p95_s']}s")
        print(f"  Gaps > 5min:            {g['pct_over_300s']}%")
        print(f"  Gaps negativos:         {g['pct_negative']}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    journeys.to_csv(args.output, index=False)
    print(f"\n[stitcher] journeys.csv guardado em: {args.output}")
    print(f"[stitcher] {len(journeys):,} linhas ({journeys['person_id'].nunique():,} pessoas × zonas visitadas)")


if __name__ == '__main__':
    main()
