#!/usr/bin/env python3
"""
Dashboard de resultados — Experimento Cross-Domain (CITRA-3D / ABOShips).

  streamlit run app/streamlit_resultados.py

Seleção de pastas por navegador de árvore (clicando), sem digitar caminho.
Lê os CSVs agregados da pasta de resultados e, opcionalmente, as curvas por
época e os best.pt da pasta de runs (para curvas e detecções).
"""
from __future__ import annotations

import glob
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cross-Domain · Resultados", layout="wide")

# --- compatibilidade com Streamlit antigo (Python 3.8) ---
def _cache(func=None, **kwargs):
    """st.cache_data (>=1.18) com fallback para st.cache (antigo) ou no-op."""
    deco = getattr(st, "cache_data", None) or getattr(st, "cache", None)
    if deco is None:
        return func if func else (lambda f: f)
    try:
        return deco(func, **kwargs) if func else deco(**kwargs)
    except TypeError:
        # st.cache antigo não aceita show_spinner etc.
        return deco(func) if func else deco

def _rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()

def _divider():
    if hasattr(st, "divider"):
        st.divider()
    else:
        st.markdown("---")

def _sb_divider():
    if hasattr(st.sidebar, "divider"):
        st.sidebar.divider()
    else:
        st.sidebar.markdown("---")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE = "B2"
METRICS = ["mAP50", "mAP50_95", "precision", "recall"]


# ------------------------------------------------------- navegador de pastas
def folder_picker(label: str, key: str, default_sub: str = "") -> str:
    """Seletor de pasta robusto (sem rerun recursivo).
    Lista as subpastas do projeto e permite escolher uma; ou digitar um
    caminho absoluto (ex.: uma pasta no Drive montado). Sem recursão."""
    # subpastas diretas do projeto + o próprio projeto
    opcoes = ["(raiz do projeto)"]
    try:
        for d in sorted(os.listdir(PROJECT_ROOT)):
            full = os.path.join(PROJECT_ROOT, d)
            if os.path.isdir(full) and not d.startswith("."):
                opcoes.append(d)
                # um nível abaixo também (ex.: runs/B2_seed42 não; só pastas-mãe)
    except OSError:
        pass
    opcoes.append("✎ digitar caminho…")

    # default: a subpasta indicada, se existir
    idx = opcoes.index(default_sub) if default_sub in opcoes else 0
    escolha = st.sidebar.selectbox(label, opcoes, index=idx, key=f"{key}_sel")

    if escolha == "(raiz do projeto)":
        return PROJECT_ROOT
    if escolha == "✎ digitar caminho…":
        return st.sidebar.text_input("Caminho completo", key=f"{key}_txt",
                                     placeholder="/caminho/para/a/pasta")
    return os.path.join(PROJECT_ROOT, escolha)


# ------------------------------------------------------------- carregamento
@_cache(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop_duplicates(subset=["arm", "seed"], keep="last")


@_cache(show_spinner=False)
def load_curve(runs_dir: str, arm: str, seed: int) -> pd.DataFrame:
    # aceita dois layouts: subpasta <arm>_seed<seed>/results.csv
    # ou arquivo plano results_<arm>_seed<seed>.csv na própria pasta
    candidatos = [
        os.path.join(runs_dir, f"{arm}_seed{seed}", "results.csv"),
        os.path.join(runs_dir, f"results_{arm}_seed{seed}.csv"),
        os.path.join(runs_dir, f"{arm}_seed{seed}.csv"),
    ]
    p = next((c for c in candidatos if os.path.exists(c)), None)
    if p is None:
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def _curve_exists(runs_dir: str, run: str) -> bool:
    """run = '<arm>_seed<seed>'. Confere os dois layouts de nome."""
    return any(os.path.exists(os.path.join(runs_dir, c)) for c in (
        os.path.join(run, "results.csv"),
        f"results_{run}.csv",
        f"{run}.csv"))


def agg_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("arm")[METRICS].agg(["mean", "std"]).round(4)
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    if BASELINE in g.index:
        for m in METRICS:
            g[f"Δ{m}"] = (g[f"{m}_mean"] - g.loc[BASELINE, f"{m}_mean"]).round(4)
    return g.reset_index()


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Configuração")

st.sidebar.markdown("### 1. Resultados (tabela)")
results_dir = folder_picker("Pasta com os CSVs de resultados",
                            key="results_dir", default_sub="results")

csvs = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
summary = pd.DataFrame()
if not csvs:
    st.sidebar.warning("Nenhum .csv nesta pasta.")
else:
    nomes = [os.path.basename(c) for c in csvs]
    st.sidebar.success(f"{len(csvs)} CSV(s): " + ", ".join(nomes))
    combinar = st.sidebar.checkbox("Combinar todos", value=False)
    if combinar:
        dfs = []
        for c in csvs:
            d = pd.read_csv(c); d["_arquivo"] = os.path.basename(c); dfs.append(d)
        summary = pd.concat(dfs, ignore_index=True) \
            .drop_duplicates(subset=["arm", "seed"], keep="last")
    else:
        escolha = st.sidebar.selectbox("Arquivo", nomes, index=len(nomes) - 1)
        summary = load_csv(os.path.join(results_dir, escolha))

_sb_divider()
st.sidebar.markdown("### 2. Runs (curvas/detecções, opcional)")
runs_dir = folder_picker("Pasta dos runs (<arm>_seed<seed>/)",
                         key="runs_dir", default_sub="runs")

# ------------------------------------------------------------------- corpo
st.title("Aumento Cross-Domain do CITRA-3D — Resultados")
st.caption("Fonte primária: ABOShips (distância estrutural 0,95 ao CITRA-3D). "
           "Baseline: B2 (COCO → CITRA).")

if summary.empty:
    st.info("Selecione, no item 1 da barra lateral, a pasta com os CSVs de "
            "resultados (ex.: a pasta `results/` do projeto).")
    st.stop()

# 1) comparativo
st.header("1 · Comparativo por braço")
agg = agg_mean_std(summary)
show = agg[["arm"] + [f"{m}_mean" for m in METRICS]].copy()
show.columns = ["Braço"] + METRICS
if "ΔmAP50" in agg:
    show["Δ mAP50 vs B2"] = agg["ΔmAP50"]
    show["Δ recall vs B2"] = agg["Δrecall"]

_sty = show.style.format({m: "{:.4f}" for m in METRICS if m in show.columns})
try:
    _sty = _sty.highlight_max(subset=["mAP50"], color="#c6efce")
except Exception:  # noqa: BLE001
    pass
try:
    st.dataframe(_sty, use_container_width=True, hide_index=True)
except TypeError:
    st.dataframe(_sty)

c1, c2 = st.columns(2)
c1.bar_chart(agg.set_index("arm")[["mAP50_mean"]])
c2.bar_chart(agg.set_index("arm")[["recall_mean"]])

if BASELINE in agg["arm"].values:
    base = agg.loc[agg.arm == BASELINE, "mAP50_mean"].iloc[0]
    for _, r in agg.iterrows():
        if r["arm"] == BASELINE:
            continue
        d = r["mAP50_mean"] - base
        st.write(f"**{r['arm']}** {'supera' if d > 0 else 'não supera'} o baseline "
                 f"({r['mAP50_mean']:.4f} vs {base:.4f}, Δ={d:+.4f} mAP50).")

# 2) curvas de treino
st.header("2 · Curvas de treino")
if not os.path.isdir(runs_dir):
    st.info("Selecione a pasta de runs (item 2 da barra lateral) para ver as curvas.")
else:
    runs_disp = sorted(summary["arm"] + "_seed" + summary["seed"].astype(str))
    disponiveis = [r for r in runs_disp if _curve_exists(runs_dir, r)]
    if not disponiveis:
        st.info(f"Nenhum `<run>/results.csv` encontrado em `{runs_dir}`. "
                "Esperado: subpastas como `B2_seed42/results.csv`.")
    else:
        sel = st.multiselect("Runs", disponiveis, default=disponiveis)
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

# 3) detecções
st.header("3 · Inspeção de detecções")
with st.expander("Abrir visualizador (requer ultralytics + best.pt + imagens)"):
    test_dir = folder_picker("Pasta de imagens de teste", key="test_dir")
    runs_disp = sorted(summary["arm"] + "_seed" + summary["seed"].astype(str))
    run_pick = st.selectbox("Modelo (best.pt)", runs_disp)
    n = st.slider("Nº de imagens", 1, 8, 3)
    if st.button("Rodar inferência"):
        try:
            from ultralytics import YOLO
            arm, seed = run_pick.rsplit("_seed", 1)
            w = os.path.join(runs_dir, f"{arm}_seed{seed}", "weights", "best.pt")
            if not os.path.exists(w):
                st.error(f"best.pt não encontrado: {w}")
            else:
                imgs = sorted(glob.glob(os.path.join(test_dir, "*")))[:n]
                model = YOLO(w)
                for img in imgs:
                    res = model(img, verbose=False)[0]
                    st.image(res.plot()[:, :, ::-1],
                             caption=f"{os.path.basename(img)} — {len(res.boxes)} det.")
        except ImportError:
            st.error("ultralytics não instalado neste ambiente.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Erro: {e}")

_divider()
st.caption("Projeto CASNAV/DMarSup · extensão de *Visual Similarity Is Not Enough*.")