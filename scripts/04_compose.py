#!/usr/bin/env python3
"""
04_compose.py — estágio 2 da composição sintética: gera as imagens sintéticas
substituindo cada embarcação real do CITRA (train+val) por um crop do ABOShips
redimensionado à bbox. CPU only. Labels herdados.

Gera no disco LOCAL (rápido) e, ao final, empacota num único .zip no Drive.
Escrita arquivo-a-arquivo no Drive é lenta e vulnerável a desconexão; um único
zip é rápido e robusto.

Uso:
  python scripts/04_compose.py --config configs/datasets.yaml
"""
import argparse
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crossdomain import synth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--local", default="/content/synth_abo",
                    help="pasta local (rápida) onde gerar antes de zipar")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sc = cfg["synth"]; citra = cfg["prepare"].get("citra_local", "/content/cross_domain/citra_sc")
    crops = sc["crops_dir"]; nvar = sc.get("n_variations", 13)
    local = args.local                 # geração local (rápida)
    drive_dir = sc["synth_dir"]        # destino final no Drive (será zipado)

    total = 0
    for split in ("train", "val"):
        total += synth.compose_inplace(
            f"{citra}/{split}/images", f"{citra}/{split}/labels", crops,
            f"{local}/{split}/images", f"{local}/{split}/labels",
            n_variations=nvar, seed=cfg["prepare"]["seed"])

    # empacota num único zip no Drive (um arquivo -> rápido e à prova de queda)
    os.makedirs(os.path.dirname(drive_dir), exist_ok=True)
    print(f"\nzipando {local} -> {drive_dir}.zip (no Drive)...")
    shutil.make_archive(drive_dir, "zip", root_dir=local)

    print(f"\n=== RESUMO ===\nsintéticas geradas: {total} ({nvar} variações/img)")
    print(f"local: {local}/{{train,val}}/{{images,labels}}")
    print(f"zip no Drive: {drive_dir}.zip")


if __name__ == "__main__":
    main()
