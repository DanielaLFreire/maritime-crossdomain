# =============================================================================
#  FASE 1 — treino do braço A_joint_InaTech (3 seeds) + avaliações
#  Pré-requisito: rodar a célula de SETUP (tools/colab_setup_sessao.py) antes.
#  GPU necessária (idealmente A100). Retomável: se a sessão cair, rode o setup
#  e esta célula de novo — o treino continua do last.pt.
# =============================================================================
import os, glob, zipfile, subprocess

DRIVE = "/content/drive/MyDrive/PROJETO_MARINHA"
ZIPS  = f"{DRIVE}/Datasets/_zips"
REPO  = "/content/maritime-crossdomain"
os.chdir(REPO)

# --- 1) sintéticas do InaTech: extrair do zip -------------------------------
SYNTH_LOCAL = "/content/synth_inatech"
zips_candidatos = [f"{ZIPS}/synth_inatech.zip",
                   f"{DRIVE}/Experimento_CrossDomain/synth_inatech.zip"]  # fallback pré-organização
if not os.path.isdir(f"{SYNTH_LOCAL}/train/images"):
    z = next((p for p in zips_candidatos if os.path.exists(p)), None)
    assert z, f"synth_inatech.zip não encontrado em: {zips_candidatos}"
    print(f"extraindo {z} -> {SYNTH_LOCAL} ...")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(SYNTH_LOCAL)   # zip sem pasta raiz: train/, val/ na raiz

# --- 2) sanity checks antes de gastar GPU ------------------------------------
n_img = len(glob.glob(f"{SYNTH_LOCAL}/train/images/*"))
n_lbl = len(glob.glob(f"{SYNTH_LOCAL}/train/labels/*.txt"))
n_real = len(glob.glob("/content/cross_domain/citra_sc/train/images/*"))
print(f"sintéticas InaTech: {n_img} imagens / {n_lbl} labels (train)")
print(f"CITRA real (train): {n_real} imagens")
assert n_img > 0 and n_img == n_lbl, "sintéticas incompletas — confira o zip!"
assert n_real > 0, "citra_sc ausente — rode a célula de setup primeiro!"
# volume esperado: n_real * 13 variações (paridade com o braço ABO)
print(f"esperado ~{n_real * 13} sintéticas (13 variações) — "
      f"{'OK' if abs(n_img - n_real * 13) < n_real else 'CONFIRA'}")

# --- 3) treino: A_joint_InaTech × 3 seeds (o passo caro) ---------------------
# trainlist balanceada (real 13x + sintético) é gerada automaticamente pelo
# train_arm; skip/resume automáticos por seed.
r = subprocess.run(
    "PYTHONPATH=src python scripts/05_train.py --config configs/datasets.yaml "
    "--arms A_joint_InaTech --seeds 42 123 2024", shell=True)
assert r.returncode == 0, "treino falhou — veja o log acima"

# --- 4) zero-shot no held-out SeaShips (acumula com os 4 braços anteriores) --
subprocess.run(
    "PYTHONPATH=src python scripts/06_eval_heldout.py --config configs/datasets.yaml "
    "--arms A_joint_InaTech --seeds 42 123 2024", shell=True)

# --- 5) comparação de fontes consolidada (in-domain) -------------------------
import pandas as pd
runs = f"{DRIVE}/Experimento_CrossDomain/runs"
df = pd.read_csv(f"{runs}/results.csv")
print("\n=== COMPARAÇÃO DE FONTES — teste CITRA (mean ± std, ddof=0) ===")
g = df.groupby("arm")
base = g.get_group("B2")["mAP50"].mean() if "B2" in g.groups else None
ordem = sorted(g.groups, key=lambda a: -g.get_group(a)["mAP50"].mean())
for arm in ordem:
    m = g.get_group(arm)
    mu, sd = m["mAP50"].mean(), m["mAP50"].std(ddof=0)
    d = f"  Δ={100*(mu-base):+.2f}pp" if base and arm != "B2" else ""
    print(f"  {arm:16s} mAP50={mu:.4f}±{sd:.4f}  "
          f"R={m['recall'].mean():.4f}±{m['recall'].std(ddof=0):.4f}{d}")
print(f"\nCSVs: {runs}/results.csv | {runs}/results_heldout_seaships.csv")
print("Pergunta da Fase 1: A_joint_InaTech supera A_joint_ABO no CITRA?")
