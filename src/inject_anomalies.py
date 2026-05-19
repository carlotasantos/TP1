import argparse
import json
import os
import random
import pandas as pd


RANDOM_SEED = 42

# Configuração das anomalias usadas nos testes
ANOMALIES = [
    {
        "id": "ANO_001",
        "tipo": "entrada_bloqueada",
        "descricao": "Z_E1 às 10h do domingo — entrada principal quase vazia",
        "zona": "Z_E1",
        "data": "2025-03-16",
        "hora": 10,
        "operacao": "remover_percentagem",
        "percentagem_remover": 0.90,
        "impacto_esperado": "queda de ~90% nos eventos de entrada em hora movimentada",
    },
    {
        "id": "ANO_002",
        "tipo": "pico_secao_produto",
        "descricao": "Z_S3 às 13h do domingo — secção de produto com tráfego quadruplicado",
        "zona": "Z_S3",
        "data": "2025-03-16",
        "hora": 13,
        "operacao": "duplicar_eventos",
        "fator_multiplicacao": 4,
        "impacto_esperado": "tráfego 4x acima do normal numa secção ao almoço",
    },
    {
        "id": "ANO_003",
        "tipo": "caixa_offline",
        "descricao": "Z_C2 às 17h do domingo — caixa principal fora de serviço na hora de pico",
        "zona": "Z_C2",
        "data": "2025-03-16",
        "hora": 17,
        "operacao": "remover_percentagem",
        "percentagem_remover": 0.85,
        "impacto_esperado": "queda de ~85% na caixa mais movimentada durante o pico da tarde",
    },
    {
        "id": "ANO_004",
        "tipo": "corredor_bloqueado",
        "descricao": "Z_N8 às 11h, 12h e 13h do domingo — corredor bloqueado durante 3 horas",
        "zona": "Z_N8",
        "data": "2025-03-16",
        "horas": [11, 12, 13],
        "operacao": "remover_percentagem",
        "percentagem_remover": 0.95,
        "impacto_esperado": "queda de ~95% em 3 horas consecutivas — possível manutenção ou obstrução",
    },
    {
        "id": "ANO_005",
        "tipo": "pico_checkout",
        "descricao": "Z_CK às 19h do domingo — zona de checkout com pico anormal ao final da tarde",
        "zona": "Z_CK",
        "data": "2025-03-16",
        "hora": 19,
        "operacao": "duplicar_eventos",
        "fator_multiplicacao": 3,
        "impacto_esperado": "tráfego triplicado no checkout às 19h — possível evento ou promoção",
    },
]


# Alteração controlada dos eventos

def inject_remover(df: pd.DataFrame, zona: str, data: str, hora: int, pct: float) -> tuple:
    random.seed(RANDOM_SEED)
    mask = (
        (df["zone_id"] == zona) &
        (df["timestamp"].str[:10] == data) &
        (df["timestamp"].str[11:13].astype(int) == hora)
    )
    target = df[mask].copy()
    n_orig = len(target)
    n_remove = int(n_orig * pct)
    indices = random.sample(list(target.index), n_remove)
    return df.drop(index=indices).reset_index(drop=True), n_orig, n_remove


def inject_duplicar(df: pd.DataFrame, zona: str, data: str, hora: int, fator: int) -> tuple:
    random.seed(RANDOM_SEED)
    mask = (
        (df["zone_id"] == zona) &
        (df["timestamp"].str[:10] == data) &
        (df["timestamp"].str[11:13].astype(int) == hora)
    )
    target = df[mask].copy()
    n_orig = len(target)
    copies = []
    for k in range(1, fator):
        cp = target.copy()
        cp["event_id"] = cp["event_id"].apply(lambda x: f"{x}_inj{k}")
        cp["timestamp"] = (
            pd.to_datetime(cp["timestamp"]) + pd.Timedelta(seconds=k * 2)
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        copies.append(cp)
    df_out = pd.concat([df] + copies, ignore_index=True)
    df_out = df_out.sort_values("timestamp").reset_index(drop=True)
    n_final = len(df_out[
        (df_out["zone_id"] == zona) &
        (df_out["timestamp"].str[:10] == data) &
        (df_out["timestamp"].str[11:13].astype(int) == hora)
    ])
    return df_out, n_orig, n_final


# Verificação das anomalias que foram criadas

def count_events(df: pd.DataFrame, zona: str, data: str, hora: int) -> int:
    return len(df[
        (df["zone_id"] == zona) &
        (df["timestamp"].str[:10] == data) &
        (df["timestamp"].str[11:13].astype(int) == hora)
    ])


def verify_injection(df_orig: pd.DataFrame, df_mod: pd.DataFrame) -> dict:
    verification = {}
    for ano in ANOMALIES:
        zona = ano["zona"]
        data = ano["data"]
        horas = ano.get("horas", [ano["hora"]])
        for hora in horas:
            key = f"{ano['id']}_h{hora}" if len(horas) > 1 else ano["id"]
            orig  = count_events(df_orig, zona, data, hora)
            modif = count_events(df_mod,  zona, data, hora)
            if ano["operacao"] == "remover_percentagem":
                ok = modif < orig * (1 - ano["percentagem_remover"] * 0.5)
                verification[key] = {
                    "original": orig, "modificado": modif,
                    "reducao_pct": round(100 * (1 - modif / max(orig, 1)), 1),
                    "injectado": ok,
                }
            else:
                ok = modif > orig * 1.5
                verification[key] = {
                    "original": orig, "modificado": modif,
                    "aumento_pct": round(100 * (modif / max(orig, 1) - 1), 1),
                    "injectado": ok,
                }
    return verification


# Execução através da linha de comandos

def main():
    parser = argparse.ArgumentParser(
        description="Adiciona anomalias variadas ao events.csv para testar o pipeline."
    )
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--ground-truth", required=True)
    args = parser.parse_args()

    print(f"[inject] A carregar {args.input}...")
    df = pd.read_csv(args.input)
    n_original = len(df)
    print(f"[inject] {n_original:,} eventos carregados.")

    df_work = df.copy()
    injection_results = []

    for ano in ANOMALIES:
        zona = ano["zona"]
        data = ano["data"]
        print(f"\n[inject] {ano['id']}: {ano['descricao']}")

        if ano["operacao"] == "remover_percentagem":
            horas = ano.get("horas", [ano["hora"]])
            detalhe = []
            for hora in horas:
                df_work, n_orig, n_rem = inject_remover(
                    df_work, zona, data, hora, ano["percentagem_remover"]
                )
                detalhe.append({"hora": hora, "originais": n_orig,
                                 "removidos": n_rem, "restantes": n_orig - n_rem})
                print(f"  {hora}h: {n_orig} → {n_orig - n_rem} eventos ({n_rem} removidos)")
            injection_results.append({"anomaly_id": ano["id"], "detalhe": detalhe})

        elif ano["operacao"] == "duplicar_eventos":
            hora = ano["hora"]
            df_work, n_orig, n_final = inject_duplicar(
                df_work, zona, data, hora, ano["fator_multiplicacao"]
            )
            print(f"  {hora}h: {n_orig} → {n_final} eventos (+{n_final - n_orig} adicionados)")
            injection_results.append({
                "anomaly_id": ano["id"],
                "originais": n_orig, "totais": n_final,
                "adicionados": n_final - n_orig,
            })

    print(f"\n[inject] A verificar injeções...")
    verification = verify_injection(df, df_work)
    all_ok = all(v.get("injectado", False) for v in verification.values())
    for k, v in verification.items():
        status = "OK" if v.get("injectado", False) else "FALHOU"
        print(f"  [{status}] {k}: {v}")
    if not all_ok:
        print("[inject] AVISO: Nem todas as injeções foram verificadas.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_work.to_csv(args.output, index=False)
    print(f"\n[inject] Guardado: {args.output} ({len(df_work):,} eventos, era {n_original:,})")

    ground_truth = {
        "seed": RANDOM_SEED,
        "dataset_original": args.input,
        "dataset_validacao": args.output,
        "total_anomalias": len(ANOMALIES),
        "anomalias": [
            {
                **{k: v for k, v in a.items()},
                "resultado_injecao": injection_results[i],
                "verificacao": {k2: v2 for k2, v2 in verification.items()
                                if k2.startswith(a["id"])},
            }
            for i, a in enumerate(ANOMALIES)
        ],
        "instrucoes_avaliacao": (
            "Para cada anomalia, verificar se insights.json contém pelo menos 1 insight "
            "de categoria 'anomalia' que mencione a zona e hora correctas. "
            "ANO_001: Z_E1 + 10h. "
            "ANO_002: Z_S3 + 13h. "
            "ANO_003: Z_C2 + 17h. "
            "ANO_004: Z_N8 + 11h ou 12h ou 13h. "
            "ANO_005: Z_CK + 19h."
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.ground_truth)), exist_ok=True)
    with open(args.ground_truth, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    print(f"[inject] ground_truth.json guardado: {args.ground_truth}")

    print(f"\n[inject] Para testar o pipeline:")
    print(f"  python src/stitcher.py --input {args.output} --output output/journeys_val.csv")
    print(f"  python src/analytics.py --input output/journeys_val.csv --output output/metrics_val.json")
    print(f"  python src/insights.py --input output/metrics_val.json --output output/insights_val.json")
    print(f"  python evaluate.py --data {args.output} --output output/evaluation_report.json --ground-truth {args.ground_truth}")


if __name__ == "__main__":
    main()
