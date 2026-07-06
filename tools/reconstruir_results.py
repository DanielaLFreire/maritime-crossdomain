"""
Reconstrói o results.csv COMPLETO dos 6 modelos (2 braços x 3 seeds).
Prioriza avaliar os best.pt presentes no Drive; para os ausentes, usa os
valores já medidos em sessões anteriores (evals confirmados).
Rode no Colab após o setup (02_prepare_data gera o citra.yaml).
"""
import glob, os
import pandas as pd

RUNS = "/content/drive/MyDrive/PROJETO_MARINHA/Experimento_CrossDomain/runs"
YAML = "/content/cross_domain/yamls/citra.yaml"

# valores já medidos (evals das sessões anteriores) — fallback se o .pt sumiu
CONHECIDOS = {
    ("B2", 42):          dict(mAP50=0.8275, mAP50_95=0.4970, precision=0.8403, recall=0.7682, epochs=148),
    ("B2", 123):         dict(mAP50=0.8336, mAP50_95=0.4923, precision=0.8213, recall=0.7843, epochs=127),
    ("B2", 2024):        dict(mAP50=0.8335, mAP50_95=0.5093, precision=None,   recall=0.7860, epochs=182),
    ("A_joint_ABO", 42): dict(mAP50=0.8413, mAP50_95=0.5097, precision=0.8726, recall=0.7955, epochs=43),
    ("A_joint_ABO", 123):dict(mAP50=0.8382, mAP50_95=0.5130, precision=0.8570, recall=0.7923, epochs=45),
    ("A_joint_ABO", 2024):dict(mAP50=0.8357, mAP50_95=0.5097, precision=None,  recall=0.7960, epochs=42),
}

rows = []
try:
    from ultralytics import YOLO
    have_yolo = os.path.exists(YAML)
except ImportError:
    have_yolo = False

for (arm, seed), known in CONHECIDOS.items():
    best = os.path.join(RUNS, f"{arm}_seed{seed}", "weights", "best.pt")
    if have_yolo and os.path.exists(best):
        m = YOLO(best).val(data=YAML, split="test", verbose=False)
        rows.append(dict(arm=arm, seed=seed, epochs=known["epochs"],
                         mAP50=round(float(m.box.map50),4),
                         mAP50_95=round(float(m.box.map),4),
                         precision=round(float(m.box.mp),4),
                         recall=round(float(m.box.mr),4), fonte="best.pt"))
        print(f"{arm}_seed{seed}: reavaliado (mAP50={m.box.map50:.4f})")
    else:
        rows.append(dict(arm=arm, seed=seed, **known, fonte="eval_anterior"))
        print(f"{arm}_seed{seed}: usando valor conhecido (best.pt ausente)")

df = pd.DataFrame(rows).sort_values(["arm","seed"])
df.to_csv(os.path.join(RUNS, "results_final.csv"), index=False)

# resumo mean±std
import numpy as np
print("\n=== RESUMO (mean ± std, 3 seeds) ===")
for arm in ["B2","A_joint_ABO"]:
    sub = df[df.arm==arm]
    print(f"{arm:14} mAP50={sub.mAP50.mean():.4f}±{sub.mAP50.std():.4f} | "
          f"recall={sub.recall.mean():.4f}±{sub.recall.std():.4f}")
print(df[["arm","seed","epochs","mAP50","recall","fonte"]].to_string(index=False))
print("\nsalvo em:", os.path.join(RUNS, "results_final.csv"))
