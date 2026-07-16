#!/usr/bin/env python3
"""
05_train.py — treina os braços do experimento (REQUER GPU).

Treina em disco LOCAL (Drive via FUSE quebra as escritas do Ultralytics —
Errno 95) e SINCRONIZA cada run para o Drive ao terminar. Resultados
acumulados em results.csv + summary.json (no Drive e local).

  python scripts/05_train.py --config configs/datasets.yaml --arms B2 A_joint_ABO --seeds 123 2024
  python scripts/05_train.py --config configs/datasets.yaml --arms all
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crossdomain import train  # noqa: E402

ALL_ARMS = ["B2", "C-pre", "C-joint", "A_joint_ABO", "A_joint_InaTech"]


def _sync_to_drive(local_run, drive_run):
    """Copia o run do disco local para o Drive (cópia simples de arquivos —
    funciona no FUSE, ao contrário das escritas atômicas do Ultralytics)."""
    try:
        os.makedirs(os.path.dirname(drive_run), exist_ok=True)
        shutil.copytree(local_run, drive_run, dirs_exist_ok=True)
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] falha ao sincronizar {local_run} -> Drive: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--arms", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(train.SEEDS))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    arms = ALL_ARMS if "all" in args.arms else args.arms
    drive_out = cfg["train"]["project"]
    local_out = cfg["train"].get("project_local", "/content/runs_local")
    os.makedirs(drive_out, exist_ok=True)
    os.makedirs(local_out, exist_ok=True)

    # acumula com resultados já existentes no Drive (seeds anteriores)
    prev = os.path.join(drive_out, "results.csv")
    rows = pd.read_csv(prev).to_dict("records") if os.path.exists(prev) else []

    for arm in arms:
        for seed in args.seeds:
            print(f"\n{'='*60}\n>>> {arm} | seed {seed}\n{'='*60}")
            row = train.train_arm(arm, cfg, seed)
            rows = [r for r in rows                      # evita duplicar arm/seed
                    if not (r["arm"] == arm and r["seed"] == seed)]
            rows.append(row)
            # sincroniza pesos do run para o Drive + salva CSV nos dois lugares
            _sync_to_drive(os.path.join(local_out, f"{arm}_seed{seed}"),
                           os.path.join(drive_out, f"{arm}_seed{seed}"))
            df = pd.DataFrame(rows)
            df.to_csv(f"{local_out}/results.csv", index=False)
            df.to_csv(f"{drive_out}/results.csv", index=False)

    summary = train.summarize(rows)
    for d in (local_out, drive_out):
        with open(f"{d}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n=== RESUMO (mean ± std across seeds) ===")
    for arm, m in summary.items():
        print(f"{arm:14} mAP50={m['mAP50'][0]:.4f}±{m['mAP50'][1]:.4f} | "
              f"mAP50-95={m['mAP50_95'][0]:.4f}±{m['mAP50_95'][1]:.4f} | "
              f"R={m['recall'][0]:.3f}±{m['recall'][1]:.3f}")
    print(f"\nresultados: {drive_out}/results.csv")


if __name__ == "__main__":
    main()
