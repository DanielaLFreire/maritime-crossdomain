#!/usr/bin/env python3
"""
05_train.py — treina os braços do experimento (REQUER GPU).

Braços disponíveis: B2, C-pre, C-joint, A_joint_ABO.
3 seeds (42, 123, 2024). Resultados em CSV + resumo mean±std.

  python scripts/05_train.py --config configs/datasets.yaml --arms B2 C-joint A_joint_ABO
  python scripts/05_train.py --config configs/datasets.yaml --arms all --seeds 42
"""
import argparse
import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crossdomain import train  # noqa: E402

ALL_ARMS = ["B2", "C-pre", "C-joint", "A_joint_ABO"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--arms", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(train.SEEDS))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    arms = ALL_ARMS if "all" in args.arms else args.arms
    out = cfg["train"]["project"]
    os.makedirs(out, exist_ok=True)

    rows = []
    for arm in arms:
        for seed in args.seeds:
            print(f"\n{'='*60}\n>>> {arm} | seed {seed}\n{'='*60}")
            rows.append(train.train_arm(arm, cfg, seed))
            pd.DataFrame(rows).to_csv(f"{out}/results.csv", index=False)  # salva incremental

    summary = train.summarize(rows)
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESUMO (mean ± std across seeds) ===")
    for arm, m in summary.items():
        print(f"{arm:14} mAP50={m['mAP50'][0]:.4f}±{m['mAP50'][1]:.4f} | "
              f"mAP50-95={m['mAP50_95'][0]:.4f}±{m['mAP50_95'][1]:.4f} | "
              f"R={m['recall'][0]:.3f}")
    print(f"\nresultados: {out}/results.csv | resumo: {out}/summary.json")


if __name__ == "__main__":
    main()
