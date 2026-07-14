#!/usr/bin/env python3
"""
09_profile_table.py — tabela de perfil estrutural de TODOS os datasets para o
artigo, em resolução FIXA (640x640, a de treino), garantindo comparabilidade.

Reporta, por dataset:
  - área mediana da caixa (% da imagem) — independe da resolução (métrica robusta)
  - distribuição COCO small/medium/large — medida a 640x640 (convenção; comparável)
  - densidade (objetos por imagem)

Lê os datasets do config (mesmos caminhos do 01_profile). Salva CSV + imprime
tabela pronta para o artigo.

  python scripts/09_profile_table.py --config configs/datasets.yaml \
      --res 640 --out docs/perfil_datasets_640.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crossdomain import loaders  # noqa: E402

COCO = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, 1e18)}


def perfil_dataset(bb, res):
    """bb: dict {img: (W, H, boxes_px)} do loaders.auto_load, onde boxes_px são
    (w_px, h_px) na resolução NATIVA. Recalcula:
      - área mediana em % da imagem (w_px/W * h_px/H): independe da resolução
      - COCO a `res`x`res`: reescala a caixa para a resolução fixa e classifica
      - densidade (obj/img)."""
    areas_pct, coco = [], {"small": 0, "medium": 0, "large": 0}
    n_obj = 0
    n_img = len(bb)
    for img, (W, H, boxes) in bb.items():
        for (w_px, h_px) in boxes:
            wn, hn = w_px / W, h_px / H              # fração da imagem
            areas_pct.append(wn * hn * 100)          # % da imagem (robusto)
            a_px = (wn * res) * (hn * res)           # área em px² a `res` (fixa)
            for nome, (lo, hi) in COCO.items():
                if lo <= a_px < hi:
                    coco[nome] += 1
                    break
            n_obj += 1
    tot = sum(coco.values()) or 1
    return {
        "area_med_pct": float(np.median(areas_pct)) if areas_pct else 0.0,
        "small": 100 * coco["small"] / tot,
        "medium": 100 * coco["medium"] / tot,
        "large": 100 * coco["large"] / tot,
        "dens": n_obj / n_img if n_img else 0.0,
        "n_img": n_img, "n_obj": n_obj,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--res", type=int, default=640,
                    help="resolução fixa para o cálculo COCO (default 640, a de treino)")
    ap.add_argument("--out", default="docs/perfil_datasets_640.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dsets = cfg["datasets"]

    rows = []
    for nome, d in dsets.items():
        if not os.path.isdir(d["dir"]):
            print(f"[pulado] {nome}: pasta inexistente ({d['dir']})")
            continue
        print(f"[perfilando] {nome} a {args.res}x{args.res} ...")
        keep = d.get("keep"); drop = d.get("drop")
        bb = loaders.auto_load(d["dir"], fmt=d.get("fmt", "auto"),
                               keep=keep, drop=drop)
        p = perfil_dataset(bb, args.res)
        p["dataset"] = nome
        rows.append(p)
        print(f"   área med={p['area_med_pct']:.3f}% | "
              f"small={p['small']:.1f}% med={p['medium']:.1f}% large={p['large']:.1f}% | "
              f"dens={p['dens']:.2f}")

    if not rows:
        print("Nenhum dataset perfilado.")
        return

    df = pd.DataFrame(rows)[["dataset", "area_med_pct", "small", "medium",
                             "large", "dens", "n_img", "n_obj"]]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n=== TABELA DE PERFIL (COCO medido a {args.res}x{args.res}) ===")
    print(df.to_string(index=False))
    print(f"\nsalvo: {args.out}")


if __name__ == "__main__":
    main()
