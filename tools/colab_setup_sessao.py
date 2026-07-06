# =============================================================================
#  SETUP DE SESSÃO — cole no INÍCIO de toda sessão do Colab.
#  /content é apagado a cada sessão; o Drive não. Esta célula reconstrói tudo
#  a partir do Drive/GitHub (rápido) e verifica o que precisa ser resolvido.
# =============================================================================
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile, glob, subprocess

DRIVE   = "/content/drive/MyDrive/PROJETO_MARINHA"
ZIPS    = f"{DRIVE}/Datasets/_zips"
CROSSD  = f"{DRIVE}/Experimento_CrossDomain"
REPO    = "/content/maritime-crossdomain"

# 1) repo (some a cada sessão) --------------------------------------------------
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone",
                    "https://github.com/DanielaLFreire/maritime-crossdomain.git",
                    REPO], check=True)
os.chdir(REPO)
subprocess.run(["git", "pull", "-q"])
print("repo:", REPO)

# 2) dependências ---------------------------------------------------------------
subprocess.run("pip -q install segment-anything scipy ultralytics".split())

# 3) datasets locais a partir dos zips (ABOShips/SeaShips) -----------------------
for name, dest in [("ABOships.zip", "/content/datasets/ABOships"),
                   ("seaship.zip",  "/content/datasets/SeaShips_zip")]:
    os.makedirs(dest, exist_ok=True)
    z = f"{ZIPS}/{name}"
    if os.path.exists(z) and not os.listdir(dest):
        with zipfile.ZipFile(z) as zf: zf.extractall(dest)

# 3b) SeaShips VOC — preferir a cópia salva no Drive (detecção, com XML) ---------
SEA_DRIVE = f"{CROSSD}/SeaShips_voc"
SEA_LOCAL = "/content/datasets/SeaShips"
if os.path.isdir(SEA_DRIVE):
    subprocess.run(["cp", "-r", SEA_DRIVE, SEA_LOCAL])
    print("SeaShips VOC copiado do Drive.")
else:
    print("[ATENÇÃO] SeaShips VOC não está no Drive. Rebaixe do Roboflow e salve:")
    print("  rf...download('voc', location='/content/datasets/SeaShips')")
    print(f"  !cp -r /content/datasets/SeaShips {SEA_DRIVE}")

# 4) SAM checkpoint — do Drive, senão baixa ------------------------------------
SAM = f"{REPO}/sam_vit_b_01ec64.pth"
SAM_DRIVE = f"{DRIVE}/sam_vit_b_01ec64.pth"
if os.path.exists(SAM_DRIVE):
    subprocess.run(["cp", SAM_DRIVE, SAM]); print("SAM copiado do Drive.")
elif not os.path.exists(SAM):
    subprocess.run(["wget", "-q", "-O", SAM,
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"])
    subprocess.run(["cp", SAM, SAM_DRIVE]); print("SAM baixado e salvo no Drive.")

# 5) crops SAM — se já existem no Drive, traz de volta (pula estágio 1) ----------
CROPS_DRIVE = f"{CROSSD}/crops_abo"
CROPS_LOCAL = "/content/cross_domain/crops_abo"
if os.path.isdir(CROPS_DRIVE):
    os.makedirs("/content/cross_domain", exist_ok=True)
    subprocess.run(["cp", "-r", CROPS_DRIVE, CROPS_LOCAL])
    n = len(glob.glob(f"{CROPS_LOCAL}/*.png"))
    print(f"crops do Drive: {n} PNGs (estágio 1 já feito — pule para composição)")

# 6) refazer splits (rápido; mesmo split via manifesto) -------------------------
subprocess.run("PYTHONPATH=src python scripts/02_prepare_data.py "
               "--config configs/datasets.yaml", shell=True)

# 7) diagnóstico ----------------------------------------------------------------
import torch
print("\n=== DIAGNÓSTICO ===")
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—")
print("SeaShips XMLs:", len(glob.glob(f"{SEA_LOCAL}/**/*.xml", recursive=True)))
print("SAM:", os.path.exists(SAM))
print("crops presentes:", len(glob.glob(f"{CROPS_LOCAL}/*.png")))
print("\nPróximo passo:")
print("  estágio 1 (se crops=0): !PYTHONPATH=src python scripts/03_extract_crops.py "
      "--config configs/datasets.yaml --sam sam_vit_b_01ec64.pth")
print("  estágio 2:              !PYTHONPATH=src python scripts/04_compose.py "
      "--config configs/datasets.yaml")
