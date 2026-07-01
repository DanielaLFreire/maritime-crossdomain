"""
Perfil estrutural (escala / densidade / %small) e distância composta ao domínio
operacional de referência (CITRA-3D-Real), no espírito da Tabela I do artigo.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .loaders import BoxesByImage

EVAL_SIZE = 640          # lado maior após letterbox (convenção do artigo)
SMALL_PX = 32 * 32       # limiar COCO para "small" após reescala p/ EVAL_SIZE


def profile(boxes_by_img: BoxesByImage, eval_size: int = EVAL_SIZE,
            small_px: int = SMALL_PX) -> dict:
    """Retorna métricas dos três eixos do gap + contagens."""
    areas, objs, small = [], [], []
    for _, (W, H, boxes) in boxes_by_img.items():
        objs.append(len(boxes))
        s = eval_size / max(W, H)
        for (bw, bh) in boxes:
            areas.append((bw * bh) / (W * H))
            small.append(1 if (bw * s) * (bh * s) < small_px else 0)
    a = np.array(areas); o = np.array(objs)
    return dict(
        n_imgs=len(boxes_by_img), n_boxes=int(a.size),
        area_median=float(np.median(a)), area_mean=float(a.mean()),
        objs_mean=float(o.mean()), objs_median=float(np.median(o)),
        objs_max=int(o.max()), pct_small=100.0 * float(np.mean(small)),
    )


def distance(profiles: Dict[str, dict], ref: str = "CITRA-3D-Real"):
    """Distância composta nos 3 eixos (log-área, log-densidade, %small),
    normalizada pelo desvio entre datasets. Retorna DataFrame ordenado."""
    import pandas as pd

    la = {k: np.log10(v["area_median"]) for k, v in profiles.items()}
    ld = {k: np.log10(v["objs_mean"]) for k, v in profiles.items()}
    sm = {k: v["pct_small"] for k, v in profiles.items()}
    sa = np.std(list(la.values())) or 1.0
    sd = np.std(list(ld.values())) or 1.0
    ss = np.std(list(sm.values())) or 1.0

    rows = []
    for k in profiles:
        da = (la[k] - la[ref]) / sa
        dd = (ld[k] - ld[ref]) / sd
        ds = (sm[k] - sm[ref]) / ss
        rows.append(dict(dataset=k, d_escala=abs(da), d_densidade=abs(dd),
                         d_small=abs(ds), distancia=float(np.sqrt(da**2 + dd**2 + ds**2))))
    return pd.DataFrame(rows).sort_values("distancia").reset_index(drop=True)


def fig_profile(profiles: Dict[str, dict], ref: str, out_path: str):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(profiles).T.reset_index().rename(columns={"index": "dataset"})
    eixos = [("area_median", "Área mediana bbox (norm.)", True),
             ("objs_mean", "Objetos por imagem", False),
             ("pct_small", "% small (COCO@640)", False)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a_, (col, tit, logy) in zip(ax, eixos):
        cores = ["#c0392b" if d == ref else "#7f8c8d" for d in df["dataset"]]
        a_.bar(df["dataset"], df[col], color=cores); a_.set_title(tit, fontsize=10)
        if logy:
            a_.set_yscale("log")
        a_.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle(f"Perfil estrutural por dataset ({ref} destacado)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def fig_axis(dist_df, ref: str, out_path: str, near: str = ""):
    import matplotlib.pyplot as plt

    cores = ["#c0392b" if x == ref else "#27ae60" if x == near else "#2980b9"
             for x in dist_df["dataset"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(dist_df["dataset"], dist_df["distancia"], color=cores); ax.invert_yaxis()
    ax.set_xlabel("Distância estrutural ao domínio-alvo (adimensional)")
    ax.set_title("Eixo de compatibilidade estrutural (mais perto = melhor fonte)")
    for i, v in enumerate(dist_df["distancia"]):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path
