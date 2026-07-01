#!/usr/bin/env python3
"""
02_prepare_data.py — Fase 1: converte fontes externas para YOLO classe única e
faz splits DISJUNTOS por origem (ABOShips por sequência, SMD por vídeo);
SeaShips inteiro = held-out. CPU only.

  python scripts/02_prepare_data.py --config configs/datasets.yaml
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crossdomain import prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ds = cfg["datasets"]; pc = cfg["prepare"]
    out, seed, vf = pc["out"], pc["seed"], pc["val_frac"]

    abo_root = ds["ABOShips"]["dir"]
    smd_root = ds["SMD on-shore"]["dir"]
    sea_root = ds["SeaShips"]["dir"]

    # ---- ABOShips: split por sequência ----
    print("=" * 60, "\nABOShips")
    seqs = prepare.aboships_sequences(abo_root, drop=set(ds["ABOShips"].get("drop", [])))
    abo_train, abo_val = prepare.split_ids(seqs, vf, seed)
    print(f"  sequências: {len(seqs)} -> train {len(abo_train)} / val {len(abo_val)}")
    abo_counts = prepare.prepare_aboships(
        abo_root, out, abo_train, abo_val,
        img_w=ds["ABOShips"].get("img_w", 1280), img_h=ds["ABOShips"].get("img_h", 720),
        drop=set(ds["ABOShips"].get("drop", [])))
    print(f"  imagens: train {abo_counts['train']} / val {abo_counts['val']}")

    # ---- SMD: re-split por vídeo ----
    print("=" * 60, "\nSMD on-shore")
    by_vid = prepare.smd_videos(smd_root)
    smd_train, smd_val = prepare.split_ids(list(by_vid.keys()), vf, seed)
    print(f"  vídeos: {len(by_vid)} -> train {len(smd_train)} / val {len(smd_val)}")
    smd_counts = prepare.prepare_smd(by_vid, out, smd_train, smd_val)
    print(f"  imagens: train {smd_counts['train']} / val {smd_counts['val']}")

    # ---- SeaShips: held-out ----
    print("=" * 60, "\nSeaShips (held-out)")
    sea_n = prepare.prepare_seaships_heldout(sea_root, out)
    print(f"  imagens (held-out): {sea_n}")

    # ---- checagem + manifesto ----
    print("=" * 60, "\nCHECAGEM ANTI-VAZAMENTO")
    assert abo_train.isdisjoint(abo_val), "VAZAMENTO ABOShips!"
    assert smd_train.isdisjoint(smd_val), "VAZAMENTO SMD!"
    print("  OK: nenhuma sequência/vídeo compartilhada entre train e val.")

    manifest = {
        "seed": seed, "val_frac": vf,
        "aboships": {"split_by": "sequence", "train_seqs": sorted(abo_train),
                     "val_seqs": sorted(abo_val), "counts": abo_counts},
        "smd": {"split_by": "video", "train_vids": sorted(smd_train),
                "val_vids": sorted(smd_val), "counts": smd_counts},
        "seaships_heldout": {"n_images": sea_n},
    }
    path = prepare.write_manifest(pc["drive_out"], manifest)
    print(f"  manifesto: {path}")

    yamls = prepare.write_yamls(out, pc["citra_dir"])
    print(f"  data.yaml: {yamls} (confira o caminho do CITRA)")

    print("\n=== RESUMO ===")
    print(f"ABOShips (fonte 1): train {abo_counts['train']} | val {abo_counts['val']}")
    print(f"SMD      (fonte 2): train {smd_counts['train']} | val {smd_counts['val']}")
    print(f"SeaShips (held-out): {sea_n}")


if __name__ == "__main__":
    main()
