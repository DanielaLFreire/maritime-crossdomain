"""
Composição sintética in-place (fiel ao método do artigo, fonte = ABOShips).

Dois estágios:
  1) extract_crops   — SAM ViT-B com bbox como prompt -> crops RGBA (GPU)
  2) compose_inplace — substitui cada embarcação real do CITRA por um crop
                       redimensionado à bbox e alpha-blended (CPU); labels herdados

Parâmetros preservados do artigo:
  SAM ViT-B (sam_vit_b_01ec64); maior componente conexo; filtro de qualidade
  (cobertura da máscara 25–95% da bbox, dim mínima >= 50 px, aspect 0.2–8.0);
  crop RGBA; resize à bbox exata; 13 variações por imagem; labels idênticos.
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# filtro de qualidade (valores do artigo)
COV_MIN, COV_MAX = 0.25, 0.95
MIN_DIM = 20
ASPECT_MIN, ASPECT_MAX = 0.2, 8.0


# ---------------------------------------------------------------------------
# util: pares imagem/label YOLO
# ---------------------------------------------------------------------------
def _img_for_label(txt: str) -> Optional[str]:
    stem = os.path.splitext(txt)[0]
    img = next((stem + e for e in IMG_EXT if os.path.exists(stem + e)), None)
    if img:
        return img
    cand = os.path.splitext(txt.replace(f"{os.sep}labels{os.sep}",
                                        f"{os.sep}images{os.sep}"))[0]
    return next((cand + e for e in IMG_EXT if os.path.exists(cand + e)), None)


def _yolo_boxes(txt: str, W: int, H: int) -> List[Tuple[int, int, int, int]]:
    """Lê YOLO normalizado e devolve caixas em pixels (x0,y0,x1,y1)."""
    out = []
    for line in open(txt):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = map(float, p[1:5])
        x0 = int((cx - w / 2) * W); y0 = int((cy - h / 2) * H)
        x1 = int((cx + w / 2) * W); y1 = int((cy + h / 2) * H)
        out.append((max(0, x0), max(0, y0), min(W, x1), min(H, y1)))
    return out


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Mantém o maior componente conexo (usa scipy se disponível)."""
    try:
        from scipy import ndimage
        lbl, n = ndimage.label(mask)
        if n <= 1:
            return mask
        sizes = ndimage.sum(mask, lbl, range(1, n + 1))
        return lbl == (int(np.argmax(sizes)) + 1)
    except Exception:  # noqa: BLE001
        return mask


# ---------------------------------------------------------------------------
# estágio 1 — extração de crops com SAM (GPU)
# ---------------------------------------------------------------------------
def extract_crops(images_labels_dirs: List[Tuple[str, str]], out_dir: str,
                  sam_checkpoint: str, device: str = "cuda",
                  min_dim: int = MIN_DIM, cov_min: float = COV_MIN,
                  cov_max: float = COV_MAX, aspect_min: float = ASPECT_MIN,
                  aspect_max: float = ASPECT_MAX,
                  drop_prefixes: Tuple[str, ...] = ()) -> dict:
    """Segmenta cada bbox com SAM e salva crops RGBA filtrados.

    images_labels_dirs: lista de (images_dir, labels_dir) — p.ex. o split
    ABOShips/train. out_dir: pasta dos crops. Filtros de qualidade
    parametrizáveis (min_dim, cobertura, aspect). Retorna manifesto."""
    import torch  # noqa: F401
    from segment_anything import SamPredictor, sam_model_registry

    os.makedirs(out_dir, exist_ok=True)
    print(f"[filtros] min_dim={min_dim}px | cobertura {cov_min:.2f}-{cov_max:.2f} "
          f"| aspect {aspect_min}-{aspect_max}")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    kept = dropped = 0
    for images_dir, labels_dir in images_labels_dirs:
        txts = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
        for i, txt in enumerate(txts):
            img_path = _img_for_label(txt)
            if img_path is None:
                continue
            try:
                im = Image.open(img_path).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            W, H = im.size
            arr = np.array(im)
            predictor.set_image(arr)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            for j, (x0, y0, x1, y1) in enumerate(_yolo_boxes(txt, W, H)):
                bw, bh = x1 - x0, y1 - y0
                if bw <= 1 or bh <= 1:
                    continue
                masks, _, _ = predictor.predict(
                    box=np.array([x0, y0, x1, y1]), multimask_output=False)
                mask = _largest_cc(masks[0])
                sub = mask[y0:y1, x0:x1]
                cov = float(sub.mean()) if sub.size else 0.0
                # filtro de qualidade
                if not (cov_min <= cov <= cov_max):
                    dropped += 1; continue
                if min(bw, bh) < min_dim:
                    dropped += 1; continue
                ar = bw / bh
                if not (aspect_min <= ar <= aspect_max):
                    dropped += 1; continue
                # crop RGBA
                rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
                rgba[..., :3] = arr[y0:y1, x0:x1]
                rgba[..., 3] = (sub.astype(np.uint8) * 255)
                Image.fromarray(rgba, "RGBA").save(
                    os.path.join(out_dir, f"{stem}_{j}.png"))
                kept += 1
            if (i + 1) % 200 == 0:
                print(f"   ...{i+1}/{len(txts)} imgs | crops {kept} | descartados {dropped}")

    manifest = {"crops": kept, "dropped": dropped,
                "yield": round(kept / max(kept + dropped, 1), 3),
                "filters": {"min_dim": min_dim, "cov_min": cov_min, "cov_max": cov_max,
                            "aspect_min": aspect_min, "aspect_max": aspect_max}}
    with open(os.path.join(out_dir, "_crops_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[crops] {kept} salvos | {dropped} descartados | yield {manifest['yield']:.1%}")
    return manifest


def filter_crops(crops_dir: str, min_opacity: float = 0.20,
                 min_side: int = 24) -> dict:
    """Limpeza pós-segmentação: remove crops 'magros' (máscara ocupa pouco da
    bbox -> muito fundo transparente) ou minúsculos. Roda sobre os PNGs já
    gerados, sem re-executar o SAM. CPU.

    min_opacity: fração mínima de pixels opacos (alpha>127) na bbox.
    min_side:    lado mínimo do crop em px.
    Retorna manifesto (antes/removidos/restantes) e o grava ao lado dos crops."""
    crops = glob.glob(os.path.join(crops_dir, "*.png"))
    antes = len(crops)
    removidos = 0
    for p in crops:
        try:
            im = Image.open(p)
            a = np.array(im)
            opac = float((a[..., 3] > 127).mean()) if a.ndim == 3 and a.shape[2] == 4 else 1.0
            if opac < min_opacity or min(im.size) < min_side:
                os.remove(p); removidos += 1
        except Exception:  # noqa: BLE001
            os.remove(p); removidos += 1  # arquivo ilegível também sai
    manifest = {"antes": antes, "removidos": removidos, "restantes": antes - removidos,
                "min_opacity": min_opacity, "min_side": min_side,
                "pct_removido": round(100 * removidos / max(antes, 1), 1)}
    with open(os.path.join(crops_dir, "_filter_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[filter] antes {antes} | removidos {removidos} "
          f"({manifest['pct_removido']}%) | restantes {manifest['restantes']}")
    return manifest


# ---------------------------------------------------------------------------
# estágio 2 — composição in-place (CPU)
# ---------------------------------------------------------------------------
def compose_inplace(citra_images: str, citra_labels: str, crops_dir: str,
                    out_images: str, out_labels: str,
                    n_variations: int = 13, seed: int = 42) -> int:
    """Para cada imagem do CITRA, gera n_variations substituindo cada bbox por
    um crop aleatório (redimensionado à bbox, alpha-blend). Labels herdados."""
    os.makedirs(out_images, exist_ok=True); os.makedirs(out_labels, exist_ok=True)
    crops = [p for p in glob.glob(os.path.join(crops_dir, "*.png"))]
    if not crops:
        raise RuntimeError(f"nenhum crop em {crops_dir} — rode a extração antes.")
    rng = random.Random(seed)
    n_out = 0

    for txt in sorted(glob.glob(os.path.join(citra_labels, "*.txt"))):
        stem = os.path.splitext(os.path.basename(txt))[0]
        img = next((os.path.join(citra_images, stem + e) for e in IMG_EXT
                    if os.path.exists(os.path.join(citra_images, stem + e))), None)
        if img is None:
            continue
        base = Image.open(img).convert("RGB")
        W, H = base.size
        boxes = _yolo_boxes(txt, W, H)
        if not boxes:
            continue
        for k in range(n_variations):
            scene = base.copy()
            for (x0, y0, x1, y1) in boxes:
                bw, bh = x1 - x0, y1 - y0
                if bw <= 0 or bh <= 0:
                    continue
                crop = Image.open(rng.choice(crops)).convert("RGBA").resize(
                    (bw, bh), Image.LANCZOS)
                scene.paste(crop, (x0, y0), crop.split()[3])  # alpha como máscara
            scene.save(os.path.join(out_images, f"{stem}_v{k}.jpg"), quality=95)
            # label herdado (idêntico ao original)
            with open(os.path.join(out_labels, f"{stem}_v{k}.txt"), "w") as f:
                f.write(open(txt).read())
            n_out += 1
    print(f"[compose] {n_out} imagens sintéticas geradas em {out_images}")
    return n_out
