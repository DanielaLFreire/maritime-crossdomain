#!/usr/bin/env python3
"""
07_profile_heldout_sizes.py — perfil de tamanhos de objeto do held-out vs CITRA.

Testa o PRESSUPOSTO da hipótese de tamanho: o SeaShips tem objetos maiores que o
CITRA? Se sim, explica por que o C-joint (ABOShips real, embarcações grandes)
generaliza melhor lá que o A_joint_ABO (síntese na escala minúscula do CITRA).

Compara na mesma métrica do artigo (Tabela I / Fig. 1):
  - área da caixa em % da área da imagem (mediana)
  - convenção COCO: small <32²px, medium 32²–96², large >96² (em 640×640)

  python scripts/07_profile_heldout_sizes.py --config configs/datasets.yaml
"""
import argparse
import glob
import os

import numpy as np
import yaml

COCO = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, 1e18)}


def profile(labels_dir, img_w, img_h, label):
    """Lê labels YOLO (cls cx cy w h normalizados) e resume tamanhos."""
    pct_area, coco = [], {"small": 0, "medium": 0, "large": 0}
    n_boxes = n_imgs = 0
    files = glob.glob(os.path.join(labels_dir, "*.txt"))
    for f in files:
        has_box = False
        for line in open(f):
            p = line.split()
            if len(p) < 5:
                continue
            w_n, h_n = float(p[3]), float(p[4])          # normalizados (0-1)
            pct_area.append(w_n * h_n * 100)             # % da área da imagem
            a_px = (w_n * img_w) * (h_n * img_h)         # área em px²
            for name, (lo, hi) in COCO.items():
                if lo <= a_px < hi:
                    coco[name] += 1
                    break
            n_boxes += 1
            has_box = True
        if has_box:
            n_imgs += 1
    tot = sum(coco.values()) or 1
    pct_coco = {k: 100 * v / tot for k, v in coco.items()}
    print(f"=== {label} ===")
    print(f"  imagens c/ objeto: {n_imgs} | caixas: {n_boxes}")
    if pct_area:
        print(f"  área da caixa (% da imagem): mediana={np.median(pct_area):.3f}% "
              f"| média={np.mean(pct_area):.3f}% | p90={np.percentile(pct_area,90):.3f}%")
    print(f"  COCO: small={pct_coco['small']:.1f}% | "
          f"medium={pct_coco['medium']:.1f}% | large={pct_coco['large']:.1f}%")
    print()
    return {"median_pct": float(np.median(pct_area)) if pct_area else 0,
            "coco": pct_coco, "n_boxes": n_boxes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--heldout-labels",
                    default="/content/cross_domain/seaships_heldout/labels")
    ap.add_argument("--heldout-wh", nargs=2, type=int, default=[640, 640],
                    help="dimensões das imagens do held-out (Roboflow=640 640)")
    ap.add_argument("--citra-labels",
                    default="/content/cross_domain/citra_sc/train/labels")
    ap.add_argument("--citra-wh", nargs=2, type=int, default=[1920, 1080])
    args = ap.parse_args()

    print("\nPERFIL DE TAMANHOS — held-out (SeaShips) vs domínio operacional (CITRA)\n")
    res = {}
    if os.path.isdir(args.citra_labels):
        res["CITRA"] = profile(args.citra_labels, *args.citra_wh,
                               "CITRA-3D (operacional)")
    else:
        print(f"[aviso] CITRA labels não encontrados em {args.citra_labels}\n")
    if os.path.isdir(args.heldout_labels):
        res["SeaShips"] = profile(args.heldout_labels, *args.heldout_wh,
                                 "SeaShips (held-out)")
    else:
        print(f"[aviso] held-out labels não encontrados em {args.heldout_labels}\n")

    # veredito da hipótese
    if "CITRA" in res and "SeaShips" in res:
        c, s = res["CITRA"], res["SeaShips"]
        print("=== VEREDITO ===")
        print(f"  CITRA:    {c['coco']['small']:.0f}% small, mediana {c['median_pct']:.3f}%")
        print(f"  SeaShips: {s['coco']['small']:.0f}% small, mediana {s['median_pct']:.3f}%")
        maior = s["median_pct"] > c["median_pct"]
        print(f"\n  SeaShips tem objetos MAIORES que o CITRA? "
              f"{'SIM' if maior else 'NÃO'} "
              f"(mediana {s['median_pct']:.3f}% vs {c['median_pct']:.3f}%)")
        if maior:
            print("  → suporta a hipótese: C-joint (ABO real, objetos grandes) "
                  "generaliza melhor no SeaShips por afinidade de escala.")
        else:
            print("  → NÃO suporta a hipótese do tamanho; buscar outra explicação "
                  "(ex.: estilo de câmera, densidade, aparência).")


if __name__ == "__main__":
    main()
