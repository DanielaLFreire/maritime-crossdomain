#!/usr/bin/env python3
"""
00_download.py — instruções/atalhos de download das fontes externas.

Datasets grandes e/ou atrás de login não podem ser baixados de forma 100%
automática; este script imprime os comandos e baixa o que é direto (ABOShips).
Rode no Colab (rede do datacenter) baixando direto para o Drive.

  python scripts/00_download.py --zips /content/drive/.../Datasets/_zips
"""
import argparse
import os
import subprocess

ABOSHIPS_URL = "https://zenodo.org/records/4736931/files/ABOshipsDataset.zip?download=1"
SEASHIPS_KAGGLE = "tangwenyang/seaship"          # atenção: versão de CLASSIFICAÇÃO
SEASHIPS_ROBOFLOW = ("ships-tznqe", "seaships7000-yxbuv")  # detecção (VOC/YOLO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True, help="pasta no Drive para os .zip")
    args = ap.parse_args()
    os.makedirs(args.zips, exist_ok=True)

    abo = os.path.join(args.zips, "ABOships.zip")
    if not os.path.exists(abo):
        print("Baixando ABOShips (8,2 GB) do Zenodo...")
        subprocess.run(["wget", "-c", "-O", abo, ABOSHIPS_URL], check=True)
    else:
        print("ABOShips.zip já presente.")

    print("\n--- SeaShips (DETECÇÃO, VOC/YOLO) ---")
    print("A versão Kaggle 'tangwenyang/seaship' é de CLASSIFICAÇÃO (sem bbox).")
    print("Use o Roboflow (detecção) — guarde a API key como Colab Secret ROBOFLOW_KEY:")
    print("  from google.colab import userdata; from roboflow import Roboflow")
    print(f"  rf = Roboflow(api_key=userdata.get('ROBOFLOW_KEY'))")
    print(f"  rf.workspace('{SEASHIPS_ROBOFLOW[0]}').project('{SEASHIPS_ROBOFLOW[1]}')"
          ".versions()[0].download('voc', location='/content/datasets/SeaShips')")


if __name__ == "__main__":
    main()
