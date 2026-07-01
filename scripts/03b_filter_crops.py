#!/usr/bin/env python3
"""
03b_filter_crops.py — limpeza pós-segmentação dos crops SAM (CPU).
Remove crops com pouca opacidade (muito fundo transparente) ou minúsculos,
que são a principal fonte de ruído com embarcações pequenas do ABOShips.
Roda sobre os PNGs já gerados; NÃO re-executa o SAM.

  python scripts/03b_filter_crops.py --config configs/datasets.yaml
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
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sc = cfg["synth"]; f = sc.get("filters", {})
    synth.filter_crops(
        sc["crops_dir"],
        min_opacity=f.get("min_opacity", 0.20),
        min_side=f.get("min_side", 24))


if __name__ == "__main__":
    main()
