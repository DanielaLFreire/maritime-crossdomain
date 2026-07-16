# =============================================================================
#  ORGANIZAR DRIVE — consolida datasets em PROJETO_MARINHA/Datasets/_zips
#  Cole esta célula no Colab. NADA é apagado: o script só cria zips e move a
#  pasta InaTechShips. Sugestões de limpeza são impressas no final, para você
#  executar manualmente depois de conferir.
# =============================================================================
from google.colab import drive
drive.mount('/content/drive')

import os, glob, shutil, subprocess

DRIVE  = "/content/drive/MyDrive"
PROJ   = f"{DRIVE}/PROJETO_MARINHA"
DSETS  = f"{PROJ}/Datasets"
ZIPS   = f"{DSETS}/_zips"
CROSSD = f"{PROJ}/Experimento_CrossDomain"
os.makedirs(ZIPS, exist_ok=True)

# Ajuste aqui se quiser mudar o comportamento:
MOVER_INATECH  = True    # MyDrive/InaTechShips  ->  Datasets/InaTechShips
ZIP_BACKUP_CITRA = False # backup do CITRA em _zips (pesado; ligue se quiser)

def tam_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try: total += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return total / 1e9

def zipar_para_zips(origem, nome_zip):
    """Zipa em /content (rápido) e move para _zips (Drive FUSE é lento p/ escrever zip direto)."""
    destino = f"{ZIPS}/{nome_zip}"
    if os.path.exists(destino):
        print(f"  [ok] {nome_zip} já existe em _zips — pulado")
        return
    if not os.path.isdir(origem):
        print(f"  [AVISO] origem inexistente: {origem} — pulado")
        return
    print(f"  zipando {origem} ({tam_gb(origem):.2f} GB) ...")
    local = shutil.make_archive(f"/content/{nome_zip[:-4]}", "zip",
                                root_dir=os.path.dirname(origem),
                                base_dir=os.path.basename(origem))
    shutil.move(local, destino)
    print(f"  [criado] {destino} ({os.path.getsize(destino)/1e9:.2f} GB)")

# ---------------------------------------------------------------------------
# 1) AUDITORIA — o que o experimento espera encontrar
# ---------------------------------------------------------------------------
print("=== AUDITORIA DOS CAMINHOS DO EXPERIMENTO ===")
esperados = {
    "CITRA-3D-Real (pasta)":        f"{DSETS}/CITRA-3D-Real",
    "ABOships.zip":                 f"{ZIPS}/ABOships.zip",
    "seaship.zip":                  f"{ZIPS}/seaship.zip",
    "SMD smd_clean (pasta)":        f"{PROJ}/Experimento_Dataset_Similar/smd_clean",
    "SeaShips VOC (config)":        f"{CROSSD}/SeaShips_voc_completo",
    "SeaShips VOC (setup sessão)":  f"{CROSSD}/SeaShips_voc",
    "InaTech na raiz (script 08)":  f"{DRIVE}/InaTechShips",
    "InaTech crops_sam":            f"{DRIVE}/InaTechShips/crops_sam",
    "InaTech dataset (config)":     f"{PROJ}/Experimento YOLOStarLS-adapted/Datasets/InaTechShips",
    "crops_abo":                    f"{CROSSD}/crops_abo",
    "synth_abo":                    f"{CROSSD}/synth_abo",
    "runs":                         f"{CROSSD}/runs",
    "SAM checkpoint":               f"{PROJ}/sam_vit_b_01ec64.pth",
}
for nome, p in esperados.items():
    print(f"  {'✓' if os.path.exists(p) else '✗ FALTA'}  {nome}: {p}")

# ---------------------------------------------------------------------------
# 2) MOVER InaTechShips da raiz para Datasets/ (mv dentro do Drive = instantâneo)
# ---------------------------------------------------------------------------
print("\n=== MOVENDO InaTechShips ===")
INA_ANTIGO, INA_NOVO = f"{DRIVE}/InaTechShips", f"{DSETS}/InaTechShips"
if MOVER_INATECH and os.path.isdir(INA_ANTIGO) and not os.path.exists(INA_NOVO):
    subprocess.run(["mv", INA_ANTIGO, INA_NOVO], check=True)
    print(f"  [movido] {INA_ANTIGO} -> {INA_NOVO}")
    print("  [ATENÇÃO] atualize o repo (datasets.yaml e scripts/08) — patch já preparado.")
elif os.path.exists(INA_NOVO):
    print(f"  [ok] já está em {INA_NOVO}")
else:
    print("  [pulado]")

# ---------------------------------------------------------------------------
# 3) BACKUPS EM _zips (o que ainda não tem zip)
# ---------------------------------------------------------------------------
print("\n=== CRIANDO ZIPS DE BACKUP EM _zips ===")
# SMD limpo (usado no perfil estrutural)
zipar_para_zips(f"{PROJ}/Experimento_Dataset_Similar/smd_clean", "smd_clean.zip")

# SeaShips VOC (held-out de detecção — CRÍTICO: veio do Roboflow, difícil rebaixar)
sea_voc = next((p for p in (f"{CROSSD}/SeaShips_voc_completo", f"{CROSSD}/SeaShips_voc")
                if os.path.isdir(p)), None)
if sea_voc:
    zipar_para_zips(sea_voc, "SeaShips_voc.zip")
else:
    print("  [AVISO] nenhum SeaShips VOC encontrado no Drive!")

# Crops SAM do InaTech (caros de regenerar: horas de GPU)
ina_base = INA_NOVO if os.path.isdir(INA_NOVO) else INA_ANTIGO
zipar_para_zips(f"{ina_base}/crops_sam", "InaTechShips_crops_sam.zip")

# Crops SAM do ABOShips (idem)
zipar_para_zips(f"{CROSSD}/crops_abo", "crops_abo.zip")

# CITRA (opcional — dataset operacional restrito; backup local também recomendado)
if ZIP_BACKUP_CITRA:
    zipar_para_zips(f"{DSETS}/CITRA-3D-Real", "CITRA-3D-Real.zip")

# ---------------------------------------------------------------------------
# 4) RESUMO + limpeza manual sugerida
# ---------------------------------------------------------------------------
print("\n=== CONTEÚDO FINAL DE _zips ===")
for z in sorted(glob.glob(f"{ZIPS}/*.zip")):
    print(f"  {os.path.basename(z):35s} {os.path.getsize(z)/1e9:6.2f} GB")

print("""
=== LIMPEZA MANUAL (só depois de conferir os zips!) ===
Nada foi apagado. Se os zips acima estão íntegros, você PODE remover:
  - Experimento_Dataset_Similar/smd_clean       (tem smd_clean.zip)
  - Experimento_CrossDomain/synth_abo           (regenerável do crops_abo.zip em ~min)
NÃO remova: CITRA-3D-Real (pasta ativa), runs/ (resultados), crops_* originais
até validar os zips numa extração de teste.
""")
