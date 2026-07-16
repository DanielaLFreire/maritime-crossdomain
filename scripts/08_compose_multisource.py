#!/usr/bin/env python3
"""
08_compose_multisource.py — composição sintética com VOLUME CONTROLADO por fonte.

Para o experimento multi-fonte (Fase 1: "a fonte importa?"). Cria um pool de
crops de UMA ou MAIS fontes, todos SUBAMOSTRADOS ao mesmo volume V, e compõe
in-place no CITRA. Garante comparação justa: mesma quantidade de crops, mesmo
background, mesmo protocolo — só a fonte muda.

Exemplos:
  # Fase 1 — braço InaTech (16.016 crops, igual ao ABO):
  python scripts/08_compose_multisource.py --config configs/datasets.yaml \
      --sources InaTech --volume 16016 --tag inatech

  # Fase 2 — combinar as duas (16.016 de cada = 32.032):
  python scripts/08_compose_multisource.py --config configs/datasets.yaml \
      --sources ABO InaTech --volume 16016 --tag both

  # Fase 2 controle — ABO dobrado (32.032 de uma fonte só):
  python scripts/08_compose_multisource.py --config configs/datasets.yaml \
      --sources ABO --volume 32032 --tag abo_2x
"""
import argparse
import glob
import os
import random
import shutil
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crossdomain import synth  # noqa: E402

# caminhos dos crops por fonte (RGBA, extraídos via SAM)
CROP_DIRS = {
    "ABO": "/content/drive/MyDrive/PROJETO_MARINHA/Experimento_CrossDomain/crops_abo",
    "InaTech": "/content/drive/MyDrive/PROJETO_MARINHA/Datasets/InaTechShips/crops_sam",
    # "SMD": "<extrair via SAM antes de usar>",
}


def build_pool(sources, volume_por_fonte, pool_dir, seed=42):
    """Copia `volume_por_fonte` crops de cada fonte para um pool único.
    Subamostragem ALEATÓRIA (seed fixo) para reprodutibilidade. Se uma fonte
    tem menos crops que o volume pedido, usa todos e avisa."""
    os.makedirs(pool_dir, exist_ok=True)
    rng = random.Random(seed)
    total = 0
    for src in sources:
        d = CROP_DIRS[src]
        crops = glob.glob(os.path.join(d, "**", "*.png"), recursive=True)
        if len(crops) < volume_por_fonte:
            print(f"[aviso] {src}: só {len(crops)} crops (< {volume_por_fonte}); usando todos.")
            escolhidos = crops
        else:
            escolhidos = rng.sample(crops, volume_por_fonte)
        for i, c in enumerate(escolhidos):
            # prefixo com a fonte evita colisão de nomes entre datasets
            shutil.copy(c, os.path.join(pool_dir, f"{src}_{i:06d}.png"))
        total += len(escolhidos)
        print(f"[pool] {src}: {len(escolhidos)} crops copiados")
    print(f"[pool] total no pool: {total} crops em {pool_dir}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--sources", nargs="+", required=True,
                    choices=list(CROP_DIRS.keys()),
                    help="fontes dos crops (ABO, InaTech, ...)")
    ap.add_argument("--volume", type=int, required=True,
                    help="crops por fonte (subamostra ao volume; ABO tem 16016)")
    ap.add_argument("--tag", required=True,
                    help="identificador do braço (ex.: inatech, both, abo_2x)")
    ap.add_argument("--local", default=None,
                    help="pasta local de geração (default: /content/synth_<tag>)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sc = cfg["synth"]
    citra = cfg["prepare"].get("citra_local", "/content/cross_domain/citra_sc")
    nvar = sc.get("n_variations", 13)
    seed = cfg["prepare"]["seed"]
    local = args.local or f"/content/synth_{args.tag}"
    pool = f"/content/crops_pool_{args.tag}"
    drive_zip = os.path.join(os.path.dirname(sc["synth_dir"]), f"synth_{args.tag}")

    # 1) monta o pool de crops no volume controlado
    print(f"=== construindo pool: fontes={args.sources} volume/fonte={args.volume} ===")
    build_pool(args.sources, args.volume, pool, seed=seed)

    # 2) compõe in-place no CITRA usando o pool
    total = 0
    for split in ("train", "val"):
        total += synth.compose_inplace(
            f"{citra}/{split}/images", f"{citra}/{split}/labels", pool,
            f"{local}/{split}/images", f"{local}/{split}/labels",
            n_variations=nvar, seed=seed)

    # 3) zipa no Drive (um arquivo — rápido e à prova de queda)
    os.makedirs(os.path.dirname(drive_zip), exist_ok=True)
    print(f"\nzipando {local} -> {drive_zip}.zip ...")
    shutil.make_archive(drive_zip, "zip", root_dir=local)

    print(f"\n=== RESUMO ({args.tag}) ===")
    print(f"fontes: {args.sources} | crops/fonte: {args.volume}")
    print(f"sintéticas geradas: {total} ({nvar} variações/img)")
    print(f"zip: {drive_zip}.zip")


if __name__ == "__main__":
    main()
