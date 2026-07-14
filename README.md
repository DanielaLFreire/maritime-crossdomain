# Maritime Cross-Domain Augmentation (CITRA-3D-Real)

Experimento de **aumento cross-domain** do dataset operacional CITRA-3D-Real,
extensão do artigo *"Visual Similarity Is Not Enough: Domain-Adapted Synthetic
Data for Maritime Vessel Detection"* (SIBGRAPI 2026). Projeto CASNAV/DMarSup —
Termo 66/2025.

## Ideia central

A utilidade de uma fonte externa para aumentar um dataset operacional é predita
pela **compatibilidade estrutural** (escala, densidade, %small), não pela
similaridade visual. Medimos essa distância, atribuímos papéis às fontes, e
usamos composição sintética in-place (SAM) para aumentar o CITRA-3D.

Distância estrutural ao CITRA-3D (menor = mais próximo):
ABOShips 0,95 (fonte primária) · SMD on-shore 1,84 (secundária) ·
SeaShips 4,08 (held-out) · InaTechShips (mais distante; âncora "longe").

## Resultados — 4 braços, 3 seeds (teste CITRA-3D)

| Braço | Descrição | mAP50 | Recall |
|---|---|---|---|
| **A_joint_ABO** | + sintético ABOShips (joint 50/50) | **0,8384 ± 0,0023** | **0,7945 ± 0,0016** |
| B2 | baseline COCO→CITRA | 0,8315 ± 0,0029 | 0,7795 ± 0,0080 |
| C-joint | + ABOShips **real** (joint 50/50) | 0,8310 ± 0,0009 | 0,7732 ± 0,0042 |
| C-pre | pré-treino ABOShips → fine-tune CITRA | 0,8279 ± 0,0007 | 0,7632 ± 0,0045 |

Fonte: `results/results_completo_4arms.csv` (per-seed em `runs/`).

- **Só a síntese supera o baseline** (faixas mean±std não sobrepostas nos 3
  seeds): +0,69 pp mAP50 e +1,51 pp recall sobre B2.
- Co-treino com ABOShips **real** (C-joint) não ajuda no domínio CITRA e
  degrada recall.
- **Pré-treino (C-pre) é a pior opção** em todas as métricas — reforça o caso
  negativo do artigo (InaTechShips → transferência negativa).

## Generalização zero-shot (held-out SeaShips, sem fine-tune)

| Braço | mAP50 | Δ vs B2 |
|---|---|---|
| **C-joint** | **0,4395 ± 0,0117** | **+4,75 pp** |
| A_joint_ABO | 0,4184 ± 0,0146 | +2,63 pp |
| B2 | 0,3920 ± 0,0493 | — |
| C-pre | 0,3041 ± 0,0068 | −8,79 pp |

Fonte: `results/results_heldout_seaships.csv` (protocolo em `scripts/06_eval_heldout.py`).

**Inversão fora do domínio — hipótese de escala:** os objetos do CITRA são
~36× menores que os do SeaShips (mediana da área: 0,101% vs 3,635% da imagem;
ver `scripts/07_profile_heldout_sizes.py` e `docs/perfil_datasets_640.csv`).
A síntese in-place otimiza para os objetos minúsculos do CITRA; o co-treino
com dados reais preserva a detecção de objetos grandes, que casa com a
distribuição do SeaShips. Ou seja: **a síntese é o método comprovado para o
cenário operacional CITRA**; qual fonte é ótima (ABOShips vs InaTechShips)
é o objeto da Fase 1 multi-fonte.

## Estrutura

```
maritime-crossdomain/
├── configs/
│   ├── datasets.yaml           # caminhos e parâmetros (edite aqui)
│   └── split_manifest.json     # splits disjuntos congelados
├── src/crossdomain/            # pacote reutilizável
│   ├── loaders.py              # leitura YOLO/VOC/COCO/CSV/ABOShips
│   ├── profiling.py            # perfil estrutural + distância
│   ├── prepare.py              # conversão YOLO + splits disjuntos + CITRA classe única
│   ├── synth.py                # composição sintética in-place (SAM)
│   └── train.py                # treino dos braços (local + sync Drive)
├── scripts/                    # pipeline (entry-points)
│   ├── 00_download.py          # download das fontes
│   ├── 01_profile.py           # Passo zero (perfil)                    [CPU]
│   ├── 02_prepare_data.py      # splits + citra_sc + held-out YOLO      [CPU]
│   ├── 03_extract_crops.py     # crops SAM                              [GPU]
│   ├── 03b_filter_crops.py     # limpeza de crops                       [CPU]
│   ├── 04_compose.py           # composição in-place                    [CPU]
│   ├── 05_train.py             # treino dos braços                      [GPU]
│   ├── 06_eval_heldout.py      # avaliação zero-shot no SeaShips        [GPU]
│   ├── 07_profile_heldout_sizes.py  # perfil de tamanhos (hipótese de escala)
│   ├── 08_compose_multisource.py    # composição com volume controlado (Fase 1)
│   └── 09_profile_table.py     # tabela de perfil estrutural p/ o artigo
├── app/streamlit_resultados.py # dashboard de resultados (Streamlit)
├── tools/
│   ├── colab_setup_sessao.py   # setup de sessão do Colab (cole no início)
│   └── reconstruir_results.py  # reconstrói results.csv a partir dos best.pt
├── docs/                       # plano/relatório (md/pdf/docx), changelog, figuras
├── results/                    # CSVs consolidados (4 braços + held-out)
├── runs/                       # results.csv por braço/seed
└── test/images/                # amostras CITRA para o visualizador de detecções
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
python scripts/05_train.py --config configs/datasets.yaml --arms B2 A_joint_ABO C-pre C-joint --seeds 42 123 2024  # GPU
python scripts/06_eval_heldout.py --config configs/datasets.yaml --arms B2 A_joint_ABO C-pre C-joint --seeds 42 123 2024  # GPU

# Fase 1 multi-fonte (volume controlado, ex.: 16.016 crops InaTech = volume ABO):
python scripts/08_compose_multisource.py --config configs/datasets.yaml --sources InaTech --volume 16016 --tag inatech
```

No Colab: use `tools/colab_setup_sessao.py` no início de cada sessão
(`/content` é efêmero — só Drive/GitHub persistem). Rode com `PYTHONPATH=src`.

## Dashboard (Streamlit)

```bash
streamlit run app/streamlit_resultados.py
```
Aponte a barra lateral para a pasta de runs. Mostra comparativo por braço,
curvas de treino por época e visualizador de detecções (amostras em
`test/images/`).

## Convenções metodológicas

- Classe única `vessel`; labels do CITRA vêm de `labels_single_class/`.
- Splits disjuntos por origem (ABOShips por sequência, SMD por vídeo);
  splits pré-fabricados do Roboflow **não** são usados.
- Composição fiel ao artigo: 13 variações, joint balanceado 50/50.
- Multi-fonte com **volume controlado** (mesma quantidade de crops por fonte;
  ver `08_compose_multisource.py`) — só a fonte muda entre braços.
- Estatística: com n=3 seeds, reportamos mean±std e faixas não sobrepostas
  (bootstrap CI é instável nesse n e não é reportado).
- Distâncias estruturais **não** são renormalizadas após a submissão
  (consistência com o artigo).
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
- [x] Experimento completo: B2, A_joint_ABO, C-pre, C-joint × 3 seeds
- [x] Held-out SeaShips (zero-shot) + perfil de tamanhos (hipótese de escala)
- [ ] Fase 1 multi-fonte: ABOShips vs InaTechShips como fonte de síntese
      (crops volume-matched via `08_compose_multisource.py`)
- [ ] Extensão para periódico (PRL / JBCS; EAAI como fallback)
