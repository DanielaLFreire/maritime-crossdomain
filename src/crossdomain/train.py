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


def _sync_run_to_drive(run_dir: str):
    """Copia um run (pesos, results.csv, plots) do disco local para o Drive
    imediatamente após concluir. Cópia simples de arquivos — compatível com o
    FUSE do Drive (ao contrário das escritas atômicas do Ultralytics). Protege
    contra perda de modelo se a sessão cair antes do sync do fim do lote."""
    drive_project = _DRIVE_PROJECT[0]
    if not drive_project:
        return
    name = os.path.basename(run_dir)
    dst = os.path.join(drive_project, name)
    try:
        import shutil
        os.makedirs(drive_project, exist_ok=True)
        shutil.copytree(run_dir, dst, dirs_exist_ok=True)
        print(f"[sync] {name} -> Drive (imediato)")
    except Exception as e:  # noqa: BLE001
        print(f"[sync] aviso: falha ao sincronizar {name}: {e}")


# destino Drive p/ o sync imediato (setado por train_arm a cada braço)
_DRIVE_PROJECT = [None]


def _train(data: str, weights: str, epochs: int, patience: int, seed: int,
           project: str, name: str, freeze: int = 0):
    """Treina um estágio. Retoma de last.pt SÓ se for um checkpoint válido e
    incompleto (tem estado de época/otimizador). Caso contrário treina do zero.
    Sincroniza o run ao Drive imediatamente ao terminar."""
    from ultralytics import YOLO
    run_dir = os.path.join(project, name)
    last = os.path.join(run_dir, "weights", "last.pt")
    best = os.path.join(run_dir, "weights", "best.pt")

    # já concluído? (best existe e não há treino incompleto pendente)
    if os.path.exists(best) and not _is_resumable(last):
        print(f"[skip] {name} já concluído (best.pt presente)")
        return best

    if _is_resumable(last):
        print(f"[resume] retomando {name} de {last}")
        YOLO(last).train(resume=True)
        _sync_run_to_drive(run_dir)
        return best

    # treino novo — remove run parcial/órfão para não confundir o Ultralytics
    if os.path.exists(run_dir):
        import shutil as _sh
        _sh.rmtree(run_dir, ignore_errors=True)
    YOLO(weights).train(data=data, epochs=epochs, patience=patience, seed=seed,
                        project=project, name=name, exist_ok=True, freeze=freeze,
                        save_period=10, **HP)
    _sync_run_to_drive(run_dir)
    return best


def _is_resumable(last_pt: str) -> bool:
    """True se last.pt é um checkpoint de treino incompleto (retomável):
    contém 'train_args' e época salva < épocas totais."""
    if not os.path.exists(last_pt):
        return False
    try:
        import torch
        ck = torch.load(last_pt, map_location="cpu", weights_only=False)
        ep = ck.get("epoch", -1)
        return isinstance(ck, dict) and "train_args" in ck and ep is not None and ep >= 0
    except Exception:  # noqa: BLE001
        return False


def train_arm(arm: str, cfg: dict, seed: int) -> dict:
    """Treina UM braço com UM seed. Retorna métricas do teste do CITRA.
    Treina em disco LOCAL (Drive via FUSE não suporta as escritas atômicas do
    Ultralytics — Errno 95). O 05_train sincroniza o resultado para o Drive."""
    import shutil
    from ultralytics import YOLO

    tr = cfg["train"]; sc = cfg["synth"]
    citra = cfg["prepare"].get("citra_local", "/content/cross_domain/citra_sc")
    project = tr.get("project_local", "/content/runs_local")   # LOCAL (rápido)
    drive_project = tr["project"]                              # Drive (destino)
    _DRIVE_PROJECT[0] = drive_project                          # habilita sync imediato
    coco = tr.get("coco_weights", "yolo11m.pt")
    yamls = tr["yamls"]
    name = f"{arm}_seed{seed}"

    # restaura run parcial do Drive (permite resume após queda de sessão)
    drive_run = os.path.join(drive_project, name)
    local_run = os.path.join(project, name)
    if os.path.isdir(drive_run) and not os.path.isdir(local_run):
        try:
            shutil.copytree(drive_run, local_run)
            print(f"[restore] {name} restaurado do Drive para resume local")
        except Exception:  # noqa: BLE001
            pass

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