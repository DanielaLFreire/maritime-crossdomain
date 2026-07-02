"""
Treino dos braços do experimento (fiel ao protocolo do artigo).

Hiperparâmetros (validados por HPO Optuna no artigo, mantida a baseline):
  YOLOv11m, single-class, init COCO; AdamW; lr0=0.001, lrf=0.01, cosine;
  momentum 0.937; weight_decay 0.0005; warmup 3 ep; img 640; batch 16.

Regimes:
  - sequencial (C-pre): 100 ep pré-treino (patience 20) -> 300 ep FT (patience 30)
  - joint (A'joint+ABO): estágio único, real repetido 13x p/ balancear as 13
    variações sintéticas, 300 ep (patience 30), 50/50 real+sintético.

3 seeds (42, 123, 2024); resultados como mean ± std.
"""
from __future__ import annotations

import os
from typing import Dict, List

HP = dict(optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,
          momentum=0.937, weight_decay=0.0005,
          warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.01,
          imgsz=640, batch=16, cache=True, deterministic=True)
SEEDS = (42, 123, 2024)


def _train(data: str, weights: str, epochs: int, patience: int, seed: int,
           project: str, name: str, freeze: int = 0):
    """Treina um estágio. Se um run com esse nome já tem last.pt (sessão caiu no
    meio), RETOMA de onde parou (resume=True) em vez de recomeçar."""
    from ultralytics import YOLO
    last = os.path.join(project, name, "weights", "last.pt")
    best = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(best) and not os.path.exists(last):
        return best  # já concluído
    if os.path.exists(last):
        print(f"[resume] retomando {name} de {last}")
        model = YOLO(last)
        model.train(resume=True)
    else:
        model = YOLO(weights)
        model.train(data=data, epochs=epochs, patience=patience, seed=seed,
                    project=project, name=name, exist_ok=True, freeze=freeze,
                    save_period=10, **HP)
    return best


def train_arm(arm: str, cfg: dict, seed: int) -> dict:
    """Treina UM braço com UM seed. Retorna métricas do teste do CITRA."""
    from ultralytics import YOLO

    tr = cfg["train"]; sc = cfg["synth"]; citra = cfg["prepare"].get("citra_local", "/content/cross_domain/citra_sc")
    project = tr["project"]; coco = tr.get("coco_weights", "yolo11m.pt")
    yamls = tr["yamls"]           # pasta com os data.yaml gerados na Fase 1
    name = f"{arm}_seed{seed}"

    if arm == "B2":               # baseline COCO -> CITRA
        best = _train(f"{yamls}/citra.yaml", coco, 300, 30, seed, project, name)
    elif arm == "C-pre":          # COCO -> ABOShips (100) -> CITRA (300)
        pre = _train(f"{yamls}/aboships_pretrain.yaml", coco, 100, 20, seed,
                     project, f"{name}_pre")
        best = _train(f"{yamls}/citra.yaml", pre, 300, 30, seed, project, name)
    elif arm == "C-joint":        # COCO -> CITRA + ABOShips real (joint)
        best = _train(f"{yamls}/citra_aboships_joint.yaml", coco, 300, 30, seed,
                      project, name)
    elif arm == "A_joint_ABO":    # COCO -> CITRA + sintético (50/50, real 13x)
        from . import prepare
        prepare.write_balanced_trainlist(
            f"{citra}/train/images", f"{tr['synth_images']}/train/images",
            f"{yamls}/joint_trainlist.txt", repeat_real=sc.get("n_variations", 13))
        best = _train(f"{yamls}/citra_synth_joint.yaml", coco, 300, 30, seed,
                      project, name)
    else:
        raise ValueError(f"braço desconhecido: {arm}")

    # avaliação no teste do CITRA
    metrics = YOLO(best).val(data=f"{yamls}/citra.yaml", split="test",
                             project=project, name=f"{name}_eval", exist_ok=True)
    return {"arm": arm, "seed": seed, "weights": best,
            "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr)}


def summarize(rows: List[dict]) -> "Dict[str, dict]":
    """Agrega por braço: mean ± std das métricas across seeds."""
    import numpy as np
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    out = {}
    for arm, rs in by.items():
        out[arm] = {m: (float(np.mean([r[m] for r in rs])),
                        float(np.std([r[m] for r in rs])))
                    for m in ("mAP50", "mAP50_95", "precision", "recall")}
    return out