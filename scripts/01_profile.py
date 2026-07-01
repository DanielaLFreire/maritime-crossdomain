#!/usr/bin/env python3
"""
01_profile.py — Passo Zero: perfil estrutural + distância ao domínio-alvo.
CPU only. Gera CSVs e figuras.

  python scripts/01_profile.py --config configs/datasets.yaml --out /content/drive/.../_passo_zero
"""
import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crossdomain import loaders, profiling  # noqa: E402

# classes de embarcação do SeaShips (colapsadas em 'vessel')
SEA_KEEP = {"ore carrier", "bulk cargo carrier", "general cargo ship",
            "container ship", "fishing boat", "passenger ship"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--out", required=True, help="pasta de saída (csv + figuras)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))
    ref = cfg["reference"]
    prof = {}

    for name, d in cfg["datasets"].items():
        if not os.path.isdir(d["dir"]):
            print(f"[pulado] {name}: pasta inexistente ({d['dir']})"); continue
        print(f"[carregando] {name} ...")
        keep = SEA_KEEP if name == "SeaShips" else None
        drop = set(d.get("drop", []))
        bb = loaders.auto_load(d["dir"], fmt=d.get("fmt", "auto"), keep=keep, drop=drop,
                               img_w=d.get("img_w", 1280), img_h=d.get("img_h", 720))
        if not bb:
            print(f"[aviso] {name}: 0 anotações lidas."); continue
        prof[name] = profiling.profile(bb, cfg["eval_size"], cfg["small_px"])
        p = prof[name]
        print(f"   {p['n_imgs']} imgs | {p['n_boxes']} boxes | "
              f"area_med={p['area_median']:.5f} | obj/img={p['objs_mean']:.2f} | "
              f"small={p['pct_small']:.1f}%")

    assert ref in prof, f"{ref} (referência) não foi carregado."

    tab = pd.DataFrame(prof).T
    tab.to_csv(os.path.join(args.out, "perfil_estrutural.csv"))
    dist = profiling.distance(prof, ref)
    dist.to_csv(os.path.join(args.out, "distancia_citra.csv"), index=False)

    print("\n=== DISTÂNCIA AO", ref, "(menor = mais próximo) ===")
    print(dist.round(3).to_string(index=False))

    cand = dist[dist.dataset != ref].reset_index(drop=True)
    near = cand.iloc[0]["dataset"]
    print(f"\n>>> Fonte mais próxima: {near}")
    nao_smd = cand[~cand.dataset.str.contains("SMD")]
    if len(nao_smd):
        print(f">>> Held-out não-SMD (mais distante): {nao_smd.iloc[-1]['dataset']}")

    profiling.fig_profile(prof, ref, os.path.join(args.out, "fig_perfil_estrutural.png"))
    profiling.fig_axis(dist, ref, os.path.join(args.out, "fig_eixo_compatibilidade.png"), near)
    print(f"\n[ok] saídas em {args.out}")


if __name__ == "__main__":
    main()
