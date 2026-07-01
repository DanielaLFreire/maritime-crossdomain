"""
Loaders de anotação para múltiplos formatos, unificando para uma representação
comum: dict {chave_imagem: (W, H, [(bw_px, bh_px), ...])}, com W/H em pixels da
imagem original e bw/bh o tamanho da bounding box em pixels.

Formatos suportados: YOLO (.txt), Pascal VOC (.xml), COCO (.json), CSV genérico
(Roboflow/TF) e o CSV específico do ABOShips (Vesibussi_Labels.csv).

Todos colapsam para classe única `vessel`; use `keep`/`drop` para filtrar classes
por nome (só faz efeito em formatos que carregam nomes: VOC/COCO/CSV).
"""
from __future__ import annotations

import glob
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
BoxesByImage = Dict[str, Tuple[int, int, List[Tuple[float, float]]]]


def _img_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)


def _class_ok(name: str, keep: Optional[Iterable[str]], drop: Iterable[str]) -> bool:
    n = name.strip().lower()
    if n in {d.lower() for d in drop}:
        return False
    if keep is None:
        return True
    return n in {k.lower() for k in keep}


def load_yolo(root: str, keep=None, drop=frozenset()) -> BoxesByImage:
    """YOLO txt ('cls cx cy w h' normalizado). Sem nomes de classe: keep/drop
    não filtram (assume classe única já colapsada)."""
    out: BoxesByImage = {}
    for txt in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        if os.path.basename(txt).lower() in ("classes.txt", "readme.txt"):
            continue
        stem = os.path.splitext(txt)[0]
        img = next((stem + e for e in IMG_EXT if os.path.exists(stem + e)), None)
        if img is None:  # fallback labels/ -> images/
            cand = os.path.splitext(
                txt.replace(f"{os.sep}labels{os.sep}", f"{os.sep}images{os.sep}")
            )[0]
            img = next((cand + e for e in IMG_EXT if os.path.exists(cand + e)), None)
        if img is None:
            continue
        W, H = _img_size(img)
        boxes: List[Tuple[float, float]] = []
        with open(txt) as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    boxes.append((float(p[3]) * W, float(p[4]) * H))
        out[img] = (W, H, boxes)
    return out


def load_voc(root: str, keep=None, drop=frozenset()) -> BoxesByImage:
    """Pascal VOC XML (SeaShips)."""
    out: BoxesByImage = {}
    for xml in glob.glob(os.path.join(root, "**", "*.xml"), recursive=True):
        try:
            t = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        size = t.find("size")
        W = int(float(size.findtext("width"))) if size is not None else 0
        H = int(float(size.findtext("height"))) if size is not None else 0
        if not (W and H):
            fn = t.findtext("filename")
            ip = os.path.join(os.path.dirname(xml), fn) if fn else None
            if ip and os.path.exists(ip):
                W, H = _img_size(ip)
            else:
                continue
        boxes: List[Tuple[float, float]] = []
        for obj in t.findall("object"):
            if not _class_ok(obj.findtext("name", ""), keep, drop):
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x0 = float(bb.findtext("xmin")); y0 = float(bb.findtext("ymin"))
            x1 = float(bb.findtext("xmax")); y1 = float(bb.findtext("ymax"))
            boxes.append((abs(x1 - x0), abs(y1 - y0)))
        out[t.findtext("filename") or xml] = (W, H, boxes)
    return out


def load_coco(coco_json: str, keep=None, drop=frozenset()) -> BoxesByImage:
    """COCO JSON."""
    data = json.load(open(coco_json))
    imgs = {im["id"]: im for im in data["images"]}
    cats = {c["id"]: c["name"] for c in data["categories"]}
    by = defaultdict(list)
    for a in data["annotations"]:
        if not _class_ok(cats.get(a["category_id"], ""), keep, drop):
            continue
        _, _, w, h = a["bbox"]
        by[a["image_id"]].append((w, h))
    return {im["file_name"]: (im["width"], im["height"], by.get(i, []))
            for i, im in imgs.items()}


def load_csv_all(root: str, keep=None, drop=frozenset()) -> BoxesByImage:
    """CSV genérico (Roboflow/TF). Infere colunas de arquivo/bbox/classe.
    Pula com aviso CSVs de CLASSIFICAÇÃO (sem bbox)."""
    import pandas as pd

    fcands = ("filename", "file", "image", "image_name", "img", "image_id", "file_name")
    x0c = ("xmin", "x_min", "left", "x1"); y0c = ("ymin", "y_min", "top", "y1")
    x1c = ("xmax", "x_max", "right", "x2"); y1c = ("ymax", "y_max", "bottom", "y2")
    bwc = ("w", "bbox_width", "box_width", "bw"); bhc = ("h", "bbox_height", "box_height", "bh")
    wc = ("width", "img_width", "image_width"); hc = ("height", "img_height", "image_height")
    classc = ("class", "label", "category", "class_id", "category_id", "class_name")

    out: BoxesByImage = {}
    for csv in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        try:
            df = pd.read_csv(csv)
        except Exception as e:  # noqa: BLE001
            print(f"   [csv] erro lendo {os.path.basename(csv)}: {e}")
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        pick = lambda cands: next((cols[c] for c in cands if c in cols), None)  # noqa: E731
        fcol = pick(fcands); ccol = pick(classc)
        X0, Y0, X1, Y1 = pick(x0c), pick(y0c), pick(x1c), pick(y1c)
        BW, BH = pick(bwc), pick(bhc); WCOL, HCOL = pick(wc), pick(hc)
        has_box = (X0 and X1 and Y0 and Y1) or (BW and BH)
        print(f"   [csv] {os.path.basename(csv)}: file={fcol} "
              f"box={'sim' if has_box else 'NÃO'} wh={'sim' if (WCOL and HCOL) else 'não'}")
        if fcol is None or not has_box:
            print("   [csv] -> sem bbox utilizável, pulando (provável CLASSIFICAÇÃO)")
            continue
        cdir = os.path.dirname(csv)
        for fn, g in df.groupby(fcol):
            if WCOL and HCOL:
                W, H = float(g.iloc[0][WCOL]), float(g.iloc[0][HCOL])
            else:
                ip = os.path.join(cdir, str(fn))
                if not os.path.exists(ip):
                    hit = glob.glob(os.path.join(root, "**", str(fn)), recursive=True)
                    ip = hit[0] if hit else None
                if ip is None:
                    continue
                W, H = _img_size(ip)
            boxes: List[Tuple[float, float]] = []
            for _, r in g.iterrows():
                if ccol and not _class_ok(str(r[ccol]), keep, drop):
                    continue
                if X0 and X1 and Y0 and Y1:
                    boxes.append((abs(float(r[X1]) - float(r[X0])),
                                  abs(float(r[Y1]) - float(r[Y0]))))
                else:
                    boxes.append((float(r[BW]), float(r[BH])))
            out[str(fn)] = (W, H, boxes)
    return out


def load_aboships(root: str, img_w: int = 1280, img_h: int = 720,
                  drop=frozenset({"seamark", "miscellaneous"})) -> BoxesByImage:
    """ABOShips (Vesibussi_Labels.csv). Atenção: as colunas 'width'/'height' do
    CSV são o tamanho da CAIXA, não da imagem — a resolução real (fixa) é passada
    por img_w/img_h. Coords: xmin/xmax/ymin/ymax em pixels."""
    import pandas as pd

    csv = glob.glob(os.path.join(root, "**", "Vesibussi_Labels.csv"), recursive=True)[0]
    df = pd.read_csv(csv)
    drop_l = {d.lower() for d in drop}
    by: Dict[str, List[Tuple[float, float]]] = {}
    for _, r in df.iterrows():
        if str(r["class"]).strip().lower() in drop_l:
            continue
        bw = abs(float(r["xmax"]) - float(r["xmin"]))
        bh = abs(float(r["ymax"]) - float(r["ymin"]))
        if bw <= 0 or bh <= 0:
            continue
        by.setdefault(str(r["filename"]), []).append((bw, bh))
    return {k: (img_w, img_h, v) for k, v in by.items()}


def auto_load(root: str, fmt: str = "auto", keep=None, drop=frozenset(),
              img_w: int = 1280, img_h: int = 720) -> BoxesByImage:
    """Despacha pelo formato. fmt='auto' detecta VOC/COCO/CSV/YOLO pela pasta."""
    if fmt == "yolo":
        return load_yolo(root, keep, drop)
    if fmt == "voc":
        return load_voc(root, keep, drop)
    if fmt == "aboships":
        return load_aboships(root, img_w, img_h, drop or frozenset({"seamark", "miscellaneous"}))
    if fmt == "coco":
        cocos = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
        return load_coco(cocos[0], keep, drop)

    # auto
    if glob.glob(os.path.join(root, "**", "*.xml"), recursive=True):
        print("  formato: VOC XML"); return load_voc(root, keep, drop)
    cocos = [j for j in glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
             if "annot" in j.lower() or "instances" in j.lower()] or \
        glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    if cocos:
        print(f"  formato: COCO ({os.path.basename(cocos[0])})")
        return load_coco(cocos[0], keep, drop)
    if glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        print("  formato: CSV"); return load_csv_all(root, keep, drop)
    if glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        print("  formato: YOLO txt"); return load_yolo(root, keep, drop)
    print("  [aviso] formato não reconhecido em", root)
    return {}


def peek_classes(root: str) -> "dict[str, int]":
    """Lista nomes de classe presentes (VOC/COCO/CSV) para calibrar keep/drop."""
    from collections import Counter
    c: "Counter[str]" = Counter()
    xmls = glob.glob(os.path.join(root, "**", "*.xml"), recursive=True)
    if xmls:
        for x in xmls:
            try:
                for o in ET.parse(x).getroot().findall("object"):
                    c[(o.findtext("name") or "").strip().lower()] += 1
            except ET.ParseError:
                pass
        return dict(c)
    return dict(c)
