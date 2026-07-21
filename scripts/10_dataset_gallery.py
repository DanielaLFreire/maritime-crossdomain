#!/usr/bin/env python3
"""
10_dataset_gallery.py — gera a "galeria" dos datasets para a apresentação:
para cada fonte em configs/datasets.yaml, extrai N imagens de amostra com as
caixas desenhadas, um contact sheet (grade) e um manifesto CSV com contagens
(imagens/objetos), perfil estrutural resumido e o papel do dataset no
experimento.

Rode no MESMO ambiente onde os datasets estão montados (Colab/local), a
partir da raiz do repo `maritime-crossdomain` — reaproveita o
`configs/datasets.yaml` já usado pelos scripts 01–09.

Uso:
    python scripts/10_dataset_gallery.py --config configs/datasets.yaml \
        --out docs/gallery --n-samples 3 --seed 42

Saídas (em --out):
    <dataset_slug>/sample_1.jpg, sample_2.jpg, sample_3.jpg   (com bboxes)
    <dataset_slug>/contact_sheet.jpg                          (grade 1x3)
    gallery_manifest.csv                                      (resumo p/ o slide)

Formatos suportados para desenhar bbox na posição correta: YOLO (.txt),
Pascal VOC (.xml) e o CSV do ABOShips (Vesibussi_Labels.csv). Outros formatos
CSV genéricos caem no modo "sem caixas" (mostra a imagem, avisa no console).
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# Papel de cada dataset no experimento — ajuste os nomes-chave se o config usar
# rótulos diferentes dos do datasets.yaml atual.
ROLES = {
    "CITRA-3D-Real": "Alvo operacional — treino (todos os braços) + teste in-domain",
    "ABOShips": "Fonte primária de aumento — síntese SAM in-place e co-treino real "
                "(braços C-pre, C-joint, A_joint_ABO)",
    "SMD on-shore": "Fonte secundária de aumento — coerência com o probe já publicado "
                     "(candidata a réplica de H1/H2)",
    "SeaShips": "Held-out independente — avaliação zero-shot de generalização "
                "(nunca entra em treino/val)",
    "InaTechShips": "Âncora \"longe\" do artigo original (transferência −4,15 pp) — "
                     "fonte da Fase 1 multi-fonte (braço A_joint_InaTech, pendente)",
}

COCO_BINS = {"small": (0, 32 * 32), "medium": (32 * 32, 96 * 96), "large": (96 * 96, float("inf"))}
EVAL_SIZE = 640


# --------------------------------------------------------------------------- #
# Loaders "com posição" (mantêm xmin/ymin/xmax/ymax para desenhar as caixas)  #
# --------------------------------------------------------------------------- #

@dataclass
class ImgAnnot:
    path: str
    W: int
    H: int
    boxes_xyxy: List[Tuple[float, float, float, float]] = field(default_factory=list)


def _img_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def load_yolo_full(root: str) -> Dict[str, ImgAnnot]:
    out: Dict[str, ImgAnnot] = {}
    for txt in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        if os.path.basename(txt).lower() in ("classes.txt", "readme.txt"):
            continue
        stem = os.path.splitext(txt)[0]
        img = next((stem + e for e in IMG_EXT if os.path.exists(stem + e)), None)
        if img is None:
            cand = os.path.splitext(
                txt.replace(f"{os.sep}labels{os.sep}", f"{os.sep}images{os.sep}")
            )[0]
            img = next((cand + e for e in IMG_EXT if os.path.exists(cand + e)), None)
        if img is None:
            continue
        W, H = _img_size(img)
        boxes = []
        with open(txt) as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    cx, cy, w, h = (float(p[1]) * W, float(p[2]) * H,
                                     float(p[3]) * W, float(p[4]) * H)
                    boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        out[img] = ImgAnnot(img, W, H, boxes)
    return out


def load_voc_full(root: str) -> Dict[str, ImgAnnot]:
    out: Dict[str, ImgAnnot] = {}
    for xml in glob.glob(os.path.join(root, "**", "*.xml"), recursive=True):
        try:
            t = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        size = t.find("size")
        W = int(float(size.findtext("width"))) if size is not None else 0
        H = int(float(size.findtext("height"))) if size is not None else 0
        fn = t.findtext("filename")
        ip = os.path.join(os.path.dirname(xml), fn) if fn else None
        if not (W and H) and ip and os.path.exists(ip):
            W, H = _img_size(ip)
        if ip is None or not os.path.exists(ip):
            cand = glob.glob(os.path.join(root, "**", fn or "*NOPE*"), recursive=True)
            ip = cand[0] if cand else None
        if ip is None:
            continue
        boxes = []
        for obj in t.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append((float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                          float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
        out[ip] = ImgAnnot(ip, W, H, boxes)
    return out


def load_aboships_full(root: str, img_w=1280, img_h=720,
                        drop=frozenset({"seamark", "miscellaneous"})) -> Dict[str, ImgAnnot]:
    csv = glob.glob(os.path.join(root, "**", "Vesibussi_Labels.csv"), recursive=True)[0]
    df = pd.read_csv(csv)
    df = df[~df["class"].astype(str).str.strip().str.lower().isin(drop)]
    imgs = {}
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.lower().endswith(IMG_EXT):
            imgs[os.path.splitext(os.path.basename(p))[0]] = p
    out: Dict[str, ImgAnnot] = {}
    for fn, g in df.groupby("filename"):
        stem = os.path.splitext(str(fn))[0]
        ip = imgs.get(stem)
        if ip is None:
            continue
        boxes = [(float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax))
                 for r in g.itertuples()]
        out[ip] = out.get(ip, ImgAnnot(ip, img_w, img_h, []))
        out[ip].boxes_xyxy.extend(boxes)
    return out


def auto_load_full(root: str, fmt: str) -> Dict[str, ImgAnnot]:
    if fmt == "yolo":
        return load_yolo_full(root)
    if fmt == "voc":
        return load_voc_full(root)
    if fmt == "aboships":
        return load_aboships_full(root)
    # fallback: tenta detectar
    if glob.glob(os.path.join(root, "**", "*.xml"), recursive=True):
        return load_voc_full(root)
    if glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        return load_yolo_full(root)
    print(f"   [aviso] formato '{fmt}' sem desenho de caixa suportado — "
          f"amostras serão exportadas SEM bbox.")
    imgs = {}
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.lower().endswith(IMG_EXT):
            imgs[p] = ImgAnnot(p, *(_img_size(p)), [])
    return imgs


# --------------------------------------------------------------------------- #
# Desenho + contact sheet                                                     #
# --------------------------------------------------------------------------- #

def draw_boxes(annot: ImgAnnot, out_path: str, max_side: int = 1000, color=(232, 147, 91)):
    im = Image.open(annot.path).convert("RGB")
    scale = min(1.0, max_side / max(im.size))
    if scale < 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    else:
        scale = 1.0
    draw = ImageDraw.Draw(im)
    for (x0, y0, x1, y1) in annot.boxes_xyxy:
        draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], outline=color, width=3)
    im.save(out_path, quality=92)
    return out_path


def contact_sheet(sample_paths: List[str], out_path: str, thumb=(420, 315)):
    ims = [Image.open(p).convert("RGB").resize(thumb) for p in sample_paths]
    if not ims:
        return None
    sheet = Image.new("RGB", (thumb[0] * len(ims), thumb[1]), (255, 255, 255))
    for i, im in enumerate(ims):
        sheet.paste(im, (i * thumb[0], 0))
    sheet.save(out_path, quality=92)
    return out_path


# --------------------------------------------------------------------------- #
# Perfil (mesma lógica do 09_profile_table.py, resumida aqui p/ o manifesto)  #
# --------------------------------------------------------------------------- #

def perfil_resumo(annots: Dict[str, ImgAnnot], res: int = EVAL_SIZE) -> dict:
    import numpy as np

    areas_pct, coco = [], {"small": 0, "medium": 0, "large": 0}
    n_obj = 0
    for a in annots.values():
        for (x0, y0, x1, y1) in a.boxes_xyxy:
            w_px, h_px = abs(x1 - x0), abs(y1 - y0)
            if a.W <= 0 or a.H <= 0:
                continue
            wn, hn = w_px / a.W, h_px / a.H
            areas_pct.append(wn * hn * 100)
            a_px = (wn * res) * (hn * res)
            for nome, (lo, hi) in COCO_BINS.items():
                if lo <= a_px < hi:
                    coco[nome] += 1
                    break
            n_obj += 1
    tot = sum(coco.values()) or 1
    n_img = len(annots)
    return {
        "n_img": n_img,
        "n_obj": n_obj,
        "area_med_pct": float(np.median(areas_pct)) if areas_pct else 0.0,
        "small_pct": 100 * coco["small"] / tot,
        "medium_pct": 100 * coco["medium"] / tot,
        "large_pct": 100 * coco["large"] / tot,
        "dens": n_obj / n_img if n_img else 0.0,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--out", default="docs/gallery")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    dsets = cfg["datasets"]

    os.makedirs(args.out, exist_ok=True)
    rows = []

    for nome, d in dsets.items():
        root = d["dir"]
        if not os.path.isdir(root):
            print(f"[pulado] {nome}: pasta inexistente ({root})")
            continue
        fmt = d.get("fmt", "auto")
        print(f"[processando] {nome} (fmt={fmt}) ...")
        annots = auto_load_full(root, fmt)
        if not annots:
            print(f"   [aviso] nenhuma imagem/anotação encontrada em {root}")
            continue

        prof = perfil_resumo(annots)
        slug = nome.lower().replace(" ", "_").replace("-", "_")
        dset_dir = os.path.join(args.out, slug)
        os.makedirs(dset_dir, exist_ok=True)

        # amostra estratificada: tenta pegar imagens COM caixa antes de vazias
        keys = list(annots.keys())
        with_boxes = [k for k in keys if annots[k].boxes_xyxy]
        pool = with_boxes if len(with_boxes) >= args.n_samples else keys
        sample_keys = random.sample(pool, min(args.n_samples, len(pool)))

        sample_paths = []
        for i, k in enumerate(sample_keys, 1):
            out_path = os.path.join(dset_dir, f"sample_{i}.jpg")
            draw_boxes(annots[k], out_path)
            sample_paths.append(out_path)
            print(f"   amostra {i}: {os.path.basename(k)} "
                  f"({len(annots[k].boxes_xyxy)} caixas) -> {out_path}")

        sheet_path = contact_sheet(sample_paths, os.path.join(dset_dir, "contact_sheet.jpg"))

        rows.append({
            "dataset": nome,
            "papel": ROLES.get(nome, "(definir papel em ROLES no script)"),
            "n_imagens": prof["n_img"],
            "n_objetos": prof["n_obj"],
            "obj_por_imagem": round(prof["dens"], 2),
            "area_mediana_pct": round(prof["area_med_pct"], 3),
            "pct_small": round(prof["small_pct"], 1),
            "pct_medium": round(prof["medium_pct"], 1),
            "pct_large": round(prof["large_pct"], 1),
            "amostras": "; ".join(os.path.relpath(p, args.out) for p in sample_paths),
            "contact_sheet": os.path.relpath(sheet_path, args.out) if sheet_path else "",
        })

    manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(args.out, "gallery_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\n=== MANIFESTO ({len(manifest)} datasets) ===")
    print(manifest.drop(columns=["amostras", "contact_sheet"]).to_string(index=False))
    print(f"\nSalvo em: {args.out}/")
    print(f"  - {manifest_path}")
    print(f"  - <dataset>/sample_1.jpg, sample_2.jpg, sample_3.jpg (com bbox)")
    print(f"  - <dataset>/contact_sheet.jpg")
    print("\nPróximo passo: baixe a pasta 'docs/gallery/' e me envie (ou as amostras que quiser "
          "usar) para eu montar o slide da galeria com as imagens reais.")


if __name__ == "__main__":
    main()
