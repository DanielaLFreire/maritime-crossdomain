#!/usr/bin/env python3
"""
06_eval_heldout.py — avaliação zero-shot dos modelos treinados num dataset
held-out (SeaShips), SEM fine-tune. Replica o protocolo do artigo (Tabela IV,
zero-shot no SMD) num held-out mais distante. Consolida mean±std por braço.

Pré-requisitos:
  - held-out preparado em YOLO (02_prepare_data com SeaShips.dir apontando p/
    a pasta VOC com XMLs) -> gera seaships_heldout/ + seaships_heldout.yaml
  - best.pt de cada braço/seed no Drive (runs/<arm>_seed<seed>/weights/best.pt)

  python scripts/06_eval_heldout.py --config configs/datasets.yaml \
      --arms B2 A_joint_ABO C-pre C-joint --seeds 42 123 2024
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--arms", nargs="+",
                    default=["B2", "A_joint_ABO", "C-pre", "C-joint"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--heldout", default="seaships_heldout.yaml",
                    help="nome do data.yaml do held-out (na pasta de yamls)")
    args = ap.parse_args()

    from ultralytics import YOLO
    cfg = yaml.safe_load(open(args.config))
    tr = cfg["train"]
    runs = tr["project"]            # Drive: onde estão os best.pt
    yamls = tr["yamls"]
    heldout_yaml = os.path.join(yamls, args.heldout)

    if not os.path.exists(heldout_yaml):
        print(f"[erro] {heldout_yaml} não existe. Rode 02_prepare_data com "
              "SeaShips.dir apontando para a pasta VOC (com XMLs).")
        return

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            best = os.path.join(runs, f"{arm}_seed{seed}", "weights", "best.pt")
            if not os.path.exists(best):
                print(f"[skip] {arm}_seed{seed}: best.pt ausente ({best})")
                continue
            # split='val' avalia sobre o conjunto do yaml (held-out inteiro)
            m = YOLO(best).val(data=heldout_yaml, split="val", verbose=False,
                               project=runs, name=f"{arm}_seed{seed}_heldout",
                               exist_ok=True)
            rows.append(dict(arm=arm, seed=seed,
                             mAP50=round(float(m.box.map50), 4),
                             mAP50_95=round(float(m.box.map), 4),
                             precision=round(float(m.box.mp), 4),
                             recall=round(float(m.box.mr), 4)))
            print(f"{arm}_seed{seed} (held-out): mAP50={m.box.map50:.4f} "
                  f"recall={m.box.mr:.4f}")

    if not rows:
        print("Nenhum modelo avaliado. Verifique os best.pt no Drive.")
        return

    df = pd.DataFrame(rows).sort_values(["arm", "seed"])
    out_csv = os.path.join(runs, "results_heldout_seaships.csv")
    df.to_csv(out_csv, index=False)

    # resumo mean±std por braço + delta vs B2
    print("\n=== ZERO-SHOT SeaShips (mean ± std) ===")
    summ = {}
    base = None
    for arm in args.arms:
        sub = df[df.arm == arm]
        if sub.empty:
            continue
        summ[arm] = {m: (float(sub[m].mean()), float(sub[m].std()))
                     for m in ("mAP50", "mAP50_95", "precision", "recall")}
        if arm == "B2":
            base = summ[arm]["mAP50"][0]
    for arm, s in summ.items():
        d = f" (Δ={s['mAP50'][0]-base:+.4f})" if base is not None else ""
        print(f"{arm:14} mAP50={s['mAP50'][0]:.4f}±{s['mAP50'][1]:.4f}{d} | "
              f"recall={s['recall'][0]:.4f}")
    json.dump(summ, open(os.path.join(runs, "summary_heldout.json"), "w"),
              indent=2)
    print(f"\nresultados: {out_csv}")


if __name__ == "__main__":
    main()
