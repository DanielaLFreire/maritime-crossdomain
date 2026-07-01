#!/usr/bin/env python3
"""
03_extract_crops.py — estágio 1 da composição sintética: extrai crops RGBA do
ABOShips (split train) com SAM ViT-B. REQUER GPU.

Pré-requisitos:
  pip install segment-anything torch torchvision scipy
  # checkpoint SAM ViT-B (~375 MB):
  wget -O sam_vit_b_01ec64.pth \\
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Uso:
  python scripts/03_extract_crops.py --config configs/datasets.yaml \\
      --sam sam_vit_b_01ec64.pth
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crossdomain import synth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--sam", required=True, help="checkpoint sam_vit_b_01ec64.pth")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sc = cfg["synth"]
    out = sc["crops_dir"]
    # crops vêm do split TRAIN do ABOShips (disjunto), preparado na Fase 1
    pairs = [(f"{sc['aboships_split']}/train/images",
              f"{sc['aboships_split']}/train/labels")]
    print(f"Extraindo crops de {pairs[0][0]} -> {out}")
    synth.extract_crops(pairs, out, args.sam, device=args.device)


if __name__ == "__main__":
    main()
