# Maritime Cross-Domain Augmentation (CITRA-3D-Real)

Experimento de **aumento cross-domain** do dataset operacional CITRA-3D-Real,
extensão do artigo *"Visual Similarity Is Not Enough: Domain-Adapted Synthetic
Data for Maritime Vessel Detection"*. Projeto CASNAV/DMarSup — Termo 66/2025.

## Ideia central

A utilidade de uma fonte externa para aumentar um dataset operacional é predita
pela **compatibilidade estrutural** (escala, densidade, %small), não pela
similaridade visual. Medimos essa distância, atribuímos papéis às fontes, e
usamos composição sintética in-place (SAM) para aumentar o CITRA-3D.

## Resultado principal (3 seeds, teste CITRA-3D)

| Braço | mAP50 | Recall |
|---|---|---|
| B2 (baseline COCO→CITRA) | 0,8315 ± 0,0029 | 0,7795 ± 0,0080 |
| **A_joint_ABO** (+ sintético ABOShips) | **0,8384 ± 0,0023** | **0,7946 ± 0,0016** |
| Δ | **+0,69 pp** | **+1,51 pp** |

A_joint_ABO supera o baseline nos **três seeds**, com desvio-padrão menor.
Combinado ao caso negativo do artigo original (InaTechShips, fonte distante →
transferência negativa), ancora a tese: **estrutura prediz transferência**.

Distância estrutural ao CITRA-3D (menor = mais próximo):
ABOShips 0,95 (fonte primária) · SMD on-shore 1,84 (secundária) ·
SeaShips 4,08 (held-out) · InaTechShips ≫ (âncora "longe").

## Estrutura

```
maritime-crossdomain/
├── configs/datasets.yaml       # caminhos e parâmetros (edite aqui)
├── src/crossdomain/            # pacote reutilizável
│   ├── loaders.py              # leitura YOLO/VOC/COCO/CSV/ABOShips
│   ├── profiling.py            # perfil estrutural + distância
│   ├── prepare.py              # conversão YOLO + splits disjuntos + CITRA classe única
│   ├── synth.py                # composição sintética in-place (SAM)
│   └── train.py                # treino dos braços (local + sync Drive)
├── scripts/                    # pipeline (entry-points)
│   ├── 00_download.py          # download das fontes
│   ├── 01_profile.py           # Passo zero (perfil)              [CPU]
│   ├── 02_prepare_data.py      # Fase 1 (splits + citra_sc)       [CPU]
│   ├── 03_extract_crops.py     # Fase 2a (crops SAM)              [GPU]
│   ├── 03b_filter_crops.py     # Fase 2b (limpeza de crops)       [CPU]
│   ├── 04_compose.py           # Fase 2 (composição in-place)     [CPU]
│   └── 05_train.py             # Fase 3 (treino dos braços)       [GPU]
├── app/streamlit_resultados.py # dashboard de resultados (Streamlit)
├── tools/
│   ├── colab_setup_sessao.py   # setup de sessão do Colab (cole no início)
│   └── reconstruir_results.py  # reconstrói results.csv a partir dos best.pt
├── docs/                       # plano (md/pdf/docx), changelog, figuras
└── results/results_final_3seeds.csv
```

## Uso

```bash
pip install -r requirements.txt
# edite os caminhos em configs/datasets.yaml

python scripts/01_profile.py --config configs/datasets.yaml --out <saida>   # CPU
python scripts/02_prepare_data.py --config configs/datasets.yaml            # CPU
python scripts/03_extract_crops.py --config configs/datasets.yaml --sam sam_vit_b_01ec64.pth  # GPU
python scripts/03b_filter_crops.py --config configs/datasets.yaml           # CPU
python scripts/04_compose.py --config configs/datasets.yaml                 # CPU
python scripts/05_train.py --config configs/datasets.yaml --arms B2 A_joint_ABO --seeds 42 123 2024  # GPU
```

No Colab: use `tools/colab_setup_sessao.py` no início de cada sessão
(`/content` é efêmero — só Drive/GitHub persistem). Rode com `PYTHONPATH=src`.

## Dashboard (Streamlit)

```bash
streamlit run app/streamlit_resultados.py
```
Aponte a barra lateral para a pasta de runs. Mostra comparativo por braço,
curvas de treino por época e visualizador de detecções.

## Convenções metodológicas

- Classe única `vessel`; labels do CITRA vêm de `labels_single_class/`.
- Splits disjuntos por origem (ABOShips por sequência, SMD por vídeo);
  splits pré-fabricados do Roboflow **não** são usados.
- Composição fiel ao artigo: 13 variações, joint balanceado 50/50.
- Adaptações à fonte ABOShips (MIN_DIM 20 px, filtro de opacidade) — ver
  `docs/CHANGELOG_metodologico.md`.
- Treino em disco local + sync ao Drive (FUSE quebra escrita atômica — Errno 95).

## Dados e licenças

Datasets não versionados (`.gitignore`). ABOShips (Zenodo 4736931, CC BY 4.0);
SMD (Prasad et al., 2017); SeaShips (Shao et al., 2018); InaTechShips
(Teixeira et al., Ocean Engineering 2025, doi 10.1016/j.oceaneng.2025.120823).
CITRA-3D-Real é operacional e restrito.

## Estado

- [x] Passo zero (perfil → papéis) · [x] Fase 1 (splits) · [x] Fase 2 (síntese)
- [x] Experimento núcleo: B2 vs A_joint_ABO, 3 seeds (ver Resultado principal)
- [ ] C-pre / C-joint (H1/H2 + curva H4) · [ ] Held-out SeaShips (generalização)
