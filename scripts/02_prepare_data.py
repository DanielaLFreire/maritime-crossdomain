#!/usr/bin/env python3
"""
02_prepare_data.py — Fase 1: converte fontes externas para YOLO classe única e
faz splits DISJUNTOS por origem. Monta o CITRA classe única (labels_single_class).
Tolerante a fontes ausentes: se ABOShips/SMD/SeaShips não estiverem extraídos,
avisa e pula, mas sempre monta o citra_sc (essencial para treino). CPU only.

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

    # ---- CITRA classe única (SEMPRE — é o essencial para treino) ----
    print("=" * 60, "\nCITRA (classe única)")
    citra_sc = prepare.build_citra_singleclass(
        pc["citra_dir"], out, labels_src=pc.get("citra_labels", "labels_single_class"))
    yamls = prepare.write_yamls(out, citra_sc, synth_images=cfg["train"]["synth_images"])
    print(f"  data.yaml: {yamls}")

    # ---- ABOShips: split por sequência (pula se fonte ausente) ----
    print("=" * 60, "\nABOShips")
    abo_train, abo_val, abo_counts = set(), set(), {"train": 0, "val": 0}
    try:
        seqs = prepare.aboships_sequences(abo_root, drop=set(ds["ABOShips"].get("drop", [])))
        abo_train, abo_val = prepare.split_ids(seqs, vf, seed)
        print(f"  sequências: {len(seqs)} -> train {len(abo_train)} / val {len(abo_val)}")
        abo_counts = prepare.prepare_aboships(
            abo_root, out, abo_train, abo_val,
            img_w=ds["ABOShips"].get("img_w", 1280), img_h=ds["ABOShips"].get("img_h", 720),
            drop=set(ds["ABOShips"].get("drop", [])))
        print(f"  imagens: train {abo_counts['train']} / val {abo_counts['val']}")
    except (IndexError, FileNotFoundError):
        print(f"  [pulado] ABOShips não extraído em {abo_root} "
              "(ok para treinar B2 / A_joint_ABO; necessário só p/ C-pre/C-joint)")

    # ---- SMD: re-split por vídeo (pula se ausente) ----
    print("=" * 60, "\nSMD on-shore")
    smd_train, smd_val, smd_counts = set(), set(), {"train": 0, "val": 0}
    by_vid = prepare.smd_videos(smd_root)
    if by_vid:
        smd_train, smd_val = prepare.split_ids(list(by_vid.keys()), vf, seed)
        print(f"  vídeos: {len(by_vid)} -> train {len(smd_train)} / val {len(smd_val)}")
        smd_counts = prepare.prepare_smd(by_vid, out, smd_train, smd_val)
        print(f"  imagens: train {smd_counts['train']} / val {smd_counts['val']}")
    else:
        print(f"  [pulado] SMD não encontrado em {smd_root}")

    # ---- SeaShips: held-out (pula se ausente) ----
    print("=" * 60, "\nSeaShips (held-out)")
    try:
        sea_n = prepare.prepare_seaships_heldout(sea_root, out)
    except (IndexError, FileNotFoundError):
        sea_n = 0
    print(f"  imagens (held-out): {sea_n}"
          + ("" if sea_n else f"  [aviso] sem XMLs em {sea_root}"))

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

    print("\n=== RESUMO ===")
    print(f"CITRA classe única: {citra_sc}")
    print(f"ABOShips (fonte 1): train {abo_counts['train']} | val {abo_counts['val']}")
    print(f"SMD      (fonte 2): train {smd_counts['train']} | val {smd_counts['val']}")
    print(f"SeaShips (held-out): {sea_n}")


if __name__ == "__main__":
    main()