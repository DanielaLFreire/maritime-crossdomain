"""
Preparação de dados: converte as fontes externas para YOLO classe única
`vessel` e faz splits DISJUNTOS por origem (anti-vazamento):

  ABOShips -> split por SEQUÊNCIA de captura (prefixo do timestamp)
  SMD      -> split por VÍDEO (MVI_id), ignorando splits pré-fabricados
  SeaShips -> tudo = held-out (nunca treinado)

Persiste um MANIFESTO JSON (listas de sequências/vídeos por split) para o split
ser reproduzível mesmo com o disco local efêmero do Colab.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _fresh(d: str):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


def _write_label(lbl_dir: str, stem: str, boxes_norm: List[Tuple[float, float, float, float]]):
    with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
        for cx, cy, w, h in boxes_norm:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def _link(src: str, dst: str):
    if not os.path.exists(dst):
        os.symlink(src, dst)


def prepare_aboships(root: str, out: str, train_seqs: set, val_seqs: set,
                     img_w: int = 1280, img_h: int = 720,
                     drop=frozenset({"seamark", "miscellaneous"})) -> Dict[str, int]:
    import pandas as pd

    csv = glob.glob(os.path.join(root, "**", "Vesibussi_Labels.csv"), recursive=True)[0]
    df = pd.read_csv(csv)
    df["cls"] = df["class"].astype(str).str.strip().str.lower()
    df = df[~df["cls"].isin({d.lower() for d in drop})].copy()
    df["seq"] = df["filename"].astype(str).str.split("_").str[0]

    imgs = {}
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.lower().endswith(IMG_EXT):
            imgs[os.path.splitext(os.path.basename(p))[0]] = p

    counts = {"train": 0, "val": 0}
    for split, seqset in (("train", train_seqs), ("val", val_seqs)):
        _fresh(f"{out}/aboships/{split}/images"); _fresh(f"{out}/aboships/{split}/labels")
        for fn, g in df[df["seq"].isin(seqset)].groupby("filename"):
            stem = str(fn)
            ip = imgs.get(stem) or imgs.get(os.path.splitext(stem)[0])
            if ip is None:
                continue
            boxes = []
            for _, r in g.iterrows():
                cx = ((r.xmin + r.xmax) / 2) / img_w; cy = ((r.ymin + r.ymax) / 2) / img_h
                w = abs(r.xmax - r.xmin) / img_w; h = abs(r.ymax - r.ymin) / img_h
                if w <= 0 or h <= 0:
                    continue
                boxes.append((min(max(cx, 0), 1), min(max(cy, 0), 1), min(w, 1), min(h, 1)))
            if not boxes:
                continue
            _link(ip, f"{out}/aboships/{split}/images/{stem}.jpg")
            _write_label(f"{out}/aboships/{split}/labels", stem, boxes)
            counts[split] += 1
    return counts


def aboships_sequences(root: str, drop=frozenset({"seamark", "miscellaneous"})) -> List[str]:
    import pandas as pd
    csv = glob.glob(os.path.join(root, "**", "Vesibussi_Labels.csv"), recursive=True)[0]
    df = pd.read_csv(csv)
    df["cls"] = df["class"].astype(str).str.strip().str.lower()
    df = df[~df["cls"].isin({d.lower() for d in drop})]
    return sorted(df["filename"].astype(str).str.split("_").str[0].unique())


def _smd_video(path: str) -> str:
    m = re.search(r"(MVI_\d+)", os.path.basename(path))
    return m.group(1) if m else "UNK"


def smd_videos(root: str) -> Dict[str, List[str]]:
    by = defaultdict(list)
    for t in glob.glob(os.path.join(root, "**", "labels", "*.txt"), recursive=True):
        by[_smd_video(t)].append(t)
    return by


def prepare_smd(by_vid: Dict[str, List[str]], out: str,
                train_vids: set, val_vids: set) -> Dict[str, int]:
    counts = {"train": 0, "val": 0}
    for split, vset in (("train", train_vids), ("val", val_vids)):
        _fresh(f"{out}/smd/{split}/images"); _fresh(f"{out}/smd/{split}/labels")
        for v in vset:
            for t in by_vid[v]:
                stem = os.path.splitext(os.path.basename(t))[0]
                img = t.replace(f"{os.sep}labels{os.sep}", f"{os.sep}images{os.sep}")
                img = next((os.path.splitext(img)[0] + e for e in IMG_EXT
                            if os.path.exists(os.path.splitext(img)[0] + e)), None)
                if img is None:
                    continue
                _link(img, f"{out}/smd/{split}/images/{stem}.jpg")
                shutil.copy(t, f"{out}/smd/{split}/labels/{stem}.txt")
                counts[split] += 1
    return counts


def prepare_seaships_heldout(root: str, out: str, img_w=1920, img_h=1080) -> int:
    _fresh(f"{out}/seaships_heldout/images"); _fresh(f"{out}/seaships_heldout/labels")
    n = 0
    for xml in glob.glob(os.path.join(root, "**", "*.xml"), recursive=True):
        try:
            t = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        size = t.find("size")
        W = int(float(size.findtext("width"))) if size is not None else img_w
        H = int(float(size.findtext("height"))) if size is not None else img_h
        fn = t.findtext("filename")
        ip = os.path.join(os.path.dirname(xml), fn) if fn else None
        if not (ip and os.path.exists(ip)):
            cand = glob.glob(os.path.join(root, "**", fn), recursive=True) if fn else []
            ip = cand[0] if cand else None
        if ip is None:
            continue
        boxes = []
        for o in t.findall("object"):
            bb = o.find("bndbox")
            if bb is None:
                continue
            x0 = float(bb.findtext("xmin")); y0 = float(bb.findtext("ymin"))
            x1 = float(bb.findtext("xmax")); y1 = float(bb.findtext("ymax"))
            boxes.append((((x0 + x1) / 2) / W, ((y0 + y1) / 2) / H,
                          abs(x1 - x0) / W, abs(y1 - y0) / H))
        if not boxes:
            continue
        stem = os.path.splitext(os.path.basename(ip))[0]
        _link(ip, f"{out}/seaships_heldout/images/{stem}.jpg")
        _write_label(f"{out}/seaships_heldout/labels", stem, boxes)
        n += 1
    return n


def split_ids(ids: List[str], val_frac: float, seed: int) -> Tuple[set, set]:
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return set(ids[n_val:]), set(ids[:n_val])   # (train, val)


def write_yamls(out: str, citra_dir: str):
    yamls = {
        "citra_aboships_joint.yaml":
            f"train:\n  - {citra_dir}/train/images\n  - {out}/aboships/train/images\n"
            f"val: {citra_dir}/val/images\nnc: 1\nnames: [vessel]\n",
        "aboships_pretrain.yaml":
            f"train: {out}/aboships/train/images\nval: {out}/aboships/val/images\n"
            f"nc: 1\nnames: [vessel]\n",
        "seaships_heldout.yaml":
            f"train: {out}/seaships_heldout/images\nval: {out}/seaships_heldout/images\n"
            f"nc: 1\nnames: [vessel]\n",
    }
    os.makedirs(f"{out}/yamls", exist_ok=True)
    for name, txt in yamls.items():
        open(f"{out}/yamls/{name}", "w").write(txt)
    return list(yamls)


def write_manifest(drive_out: str, manifest: dict):
    os.makedirs(drive_out, exist_ok=True)
    path = os.path.join(drive_out, "split_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
