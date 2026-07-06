#!/usr/bin/env python3
"""
Dashboard de resultados — Experimento Cross-Domain (CITRA-3D / ABOShips).

  streamlit run app/streamlit_resultados.py

Lê os runs do Ultralytics (results.csv agregado + results.csv por época de cada
run) e, opcionalmente, roda os best.pt para inspeção visual de detecções.
Aponte a pasta de runs na barra lateral (por padrão, a do Drive montado).
"""
from __future__ import annotations

import glob
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cross-Domain · Resultados", layout="wide")

DEFAULT_RUNS = "results"   # pasta local com results_final_3seeds.csv
BASELINE = "B2"          # braço de referência para os deltas
METRICS = ["mAP50", "mAP50_95", "precision", "recall"]


# ---------------------------------------------------------------- carregamento
@st.cache_data(show_spinner=False)
def load_summary(runs_dir: str) -> pd.DataFrame:
    """Carrega o CSV agregado (uma linha por braço/seed).
    Aceita: um caminho .csv direto, OU uma pasta com um dos nomes conhecidos
    (results_final_3seeds.csv, results_final.csv, results.csv)."""
    if runs_dir.endswith(".csv"):
        candidatos = [runs_dir]
    else:
        candidatos = [os.path.join(runs_dir, n) for n in
                      ("results_final_3seeds.csv", "results_final.csv",
                       "results.csv")]
    p = next((c for c in candidatos if os.path.exists(c)), None)
    if p is None:
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df.drop_duplicates(subset=["arm", "seed"], keep="last")
    return df


@st.cache_data(show_spinner=False)
def load_curve(runs_dir: str, arm: str, seed: int) -> pd.DataFrame:
    """results.csv por época do Ultralytics (curvas de treino)."""
    p = os.path.join(runs_dir, f"{arm}_seed{seed}", "results.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def agg_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega por braço: mean ± std across seeds + delta vs baseline."""
    g = df.groupby("arm")[METRICS].agg(["mean", "std"]).round(4)
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    if BASELINE in g.index:
        for m in METRICS:
            g[f"Δ{m}_vs_{BASELINE}"] = (g[f"{m}_mean"]
                                        - g.loc[BASELINE, f"{m}_mean"]).round(4)
    return g.reset_index()


# ------------------------------------------------------------------- sidebar
import glob as _glob

st.sidebar.title("Configuração")
st.sidebar.markdown("**1. Pasta de resultados**")
results_dir = st.sidebar.text_input(
    "Caminho da pasta com os CSVs de resultados",
    value=DEFAULT_RUNS,
    placeholder="/home/usuario/maritime-crossdomain/results")

# lista todos os CSVs da pasta (novos experimentos aparecem aqui)
csvs = sorted(_glob.glob(os.path.join(results_dir, "*.csv"))) \
    if os.path.isdir(results_dir) else []

if not os.path.isdir(results_dir):
    st.sidebar.error(f"Pasta não encontrada:\n`{results_dir}`")
    summary = pd.DataFrame()
elif not csvs:
    st.sidebar.warning(f"Nenhum arquivo .csv em `{results_dir}`.")
    summary = pd.DataFrame()
else:
    nomes = [os.path.basename(c) for c in csvs]
    st.sidebar.success(f"{len(csvs)} arquivo(s) encontrado(s):")
    st.sidebar.write("\n".join(f"- {n}" for n in nomes))
    combinar = st.sidebar.checkbox(
        "Combinar todos", value=False,
        help="Une todos os CSVs da pasta (um por experimento) numa visão só.")
    if combinar:
        dfs = []
        for c in csvs:
            d = pd.read_csv(c)
            d["_arquivo"] = os.path.basename(c)
            dfs.append(d)
        summary = pd.concat(dfs, ignore_index=True) \
            .drop_duplicates(subset=["arm", "seed"], keep="last")
    else:
        escolha = st.sidebar.selectbox("Visualizar arquivo", nomes,
                                       index=len(nomes) - 1)
        summary = load_summary(os.path.join(results_dir, escolha))

# pasta de runs (curvas/detecções) — separada, costuma ficar no Drive
st.sidebar.markdown("**2. Pasta de runs** (opcional)")
runs_dir = st.sidebar.text_input(
    "Caminho dos runs (curvas por época e detecções)",
    value="",
    placeholder="/content/drive/.../Experimento_CrossDomain/runs")
st.sidebar.caption("A tabela de resultados vem da pasta de resultados (item 1). "
                   "Curvas por época e detecções precisam dos runs "
                   "(subdiretórios <arm>_seed<seed>/), normalmente no Drive.")

st.title("Aumento Cross-Domain do CITRA-3D — Resultados")
st.caption("Fonte primária: ABOShips (distância estrutural 0,95 ao CITRA-3D). "
           "Baseline de referência: B2 (COCO → CITRA).")

if summary.empty:
    st.warning(f"Nenhum results.csv encontrado em `{runs_dir}`. "
               "O treino ainda não gerou resultados, ou o caminho está errado.")
    st.stop()


# ----------------------------------------------------------- 1) comparativo
st.header("1 · Comparativo por braço")
agg = agg_mean_std(summary)

show = agg[["arm"] + [f"{m}_mean" for m in METRICS]].copy()
show.columns = ["Braço"] + METRICS
if f"ΔmAP50_vs_{BASELINE}" in agg:
    show[f"Δ mAP50 vs {BASELINE}"] = agg[f"ΔmAP50_vs_{BASELINE}"]
    show[f"Δ recall vs {BASELINE}"] = agg[f"Δrecall_vs_{BASELINE}"]

# tabela sem styler (compatível com qualquer versão de pandas/streamlit)
show_fmt = show.copy()
for m in METRICS:
    if m in show_fmt.columns:
        show_fmt[m] = show_fmt[m].map(lambda v: f"{v:.4f}")
for c in show_fmt.columns:
    if c.startswith("Δ"):
        show_fmt[c] = show_fmt[c].map(lambda v: f"{v:+.4f}")
try:
    st.dataframe(show_fmt, use_container_width=True, hide_index=True)
except TypeError:
    st.table(show_fmt)

c1, c2 = st.columns(2)
c1.bar_chart(agg.set_index("arm")[["mAP50_mean"]])
c2.bar_chart(agg.set_index("arm")[["recall_mean"]])

# leitura automática do veredito
if BASELINE in agg["arm"].values:
    base = agg.loc[agg.arm == BASELINE, "mAP50_mean"].iloc[0]
    for _, r in agg.iterrows():
        if r["arm"] == BASELINE:
            continue
        d = r["mAP50_mean"] - base
        verdct = "supera" if d > 0 else "não supera"
        st.write(f"**{r['arm']}** {verdct} o baseline "
                 f"({r['mAP50_mean']:.4f} vs {base:.4f}, Δ={d:+.4f} mAP50).")


# --------------------------------------------------------- 2) curvas de treino
st.header("2 · Curvas de treino")
runs_disp = sorted(summary["arm"] + "_seed" + summary["seed"].astype(str))
sel = st.multiselect("Runs", runs_disp, default=runs_disp[:2])
metric_map = {"mAP50": "metrics/mAP50(B)", "mAP50-95": "metrics/mAP50-95(B)",
              "recall": "metrics/recall(B)", "precision": "metrics/precision(B)",
              "box_loss": "train/box_loss"}
metric_key = st.selectbox("Métrica", list(metric_map), index=0)

curves = {}
for run in sel:
    arm, seed = run.rsplit("_seed", 1)
    d = load_curve(runs_dir, arm, int(seed))
    col = metric_map[metric_key]
    if not d.empty and col in d.columns:
        curves[run] = d.set_index("epoch")[col]
if curves:
    st.line_chart(pd.DataFrame(curves))
else:
    st.info("Sem curvas por época para os runs selecionados "
            "(procuro results.csv dentro de cada <arm>_seed<seed>/).")


# ------------------------------------------------------- 3) detecções (opcional)
st.header("3 · Inspeção de detecções")
st.caption("Roda o best.pt de um braço em imagens de teste do CITRA. "
           "Requer ultralytics instalado no ambiente do dashboard.")

with st.expander("Abrir visualizador"):
    test_dir = st.text_input(
        "Imagens de teste",
        "/content/cross_domain/citra_sc/test/images")
    run_pick = st.selectbox("Modelo (best.pt)", runs_disp)
    n = st.slider("Nº de imagens", 1, 8, 3)
    if st.button("Rodar inferência"):
        try:
            from ultralytics import YOLO
            arm, seed = run_pick.rsplit("_seed", 1)
            w = os.path.join(runs_dir, f"{arm}_seed{seed}", "weights", "best.pt")
            imgs = sorted(glob.glob(os.path.join(test_dir, "*")))[:n]
            if not imgs:
                st.error("Nenhuma imagem encontrada nesse caminho.")
            else:
                model = YOLO(w)
                for img in imgs:
                    res = model(img, verbose=False)[0]
                    st.image(res.plot()[:, :, ::-1],
                             caption=f"{os.path.basename(img)} — "
                                     f"{len(res.boxes)} detecções", width=640)
        except ImportError:
            st.error("ultralytics não está instalado neste ambiente. "
                     "Rode `pip install ultralytics` onde o Streamlit executa.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Erro: {e}")

st.divider()
st.caption("Projeto CASNAV/DMarSup · extensão de *Visual Similarity Is Not "
           "Enough*. Dados: results.csv gerado por scripts/05_train.py.")