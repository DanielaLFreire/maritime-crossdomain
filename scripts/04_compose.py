#!/usr/bin/env python3
"""
04_compose.py — estágio 2 da composição sintética: gera as imagens sintéticas
substituindo cada embarcação real do CITRA (train+val) por um crop do ABOShips
redimensionado à bbox. CPU only. Labels herdados.

Uso:
  python scripts/04_compose.py --config configs/datasets.yaml
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
    sc = cfg["synth"]; citra = cfg["prepare"]["citra_dir"]
    crops = sc["crops_dir"]; out = sc["synth_dir"]; nvar = sc.get("n_variations", 13)

    total = 0
    for split in ("train", "val"):
        n = synth.compose_inplace(
            f"{citra}/{split}/images", f"{citra}/{split}/labels", crops,
            f"{out}/{split}/images", f"{out}/{split}/labels",
            n_variations=nvar, seed=cfg["prepare"]["seed"])
        total += n
    print(f"\n=== RESUMO ===\nsintéticas geradas: {total} ({nvar} variações/img)")
    print(f"saída: {out}/{{train,val}}/{{images,labels}}")


if __name__ == "__main__":
    main()
