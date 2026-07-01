# Maritime Cross-Domain Augmentation (CITRA-3D-Real)

Código do experimento de **aumento cross-domain** do dataset operacional
CITRA-3D-Real, extensão do artigo *"Visual Similarity Is Not Enough:
Domain-Adapted Synthetic Data for Maritime Vessel Detection"*.

Projeto CASNAV/DMarSup — Termo 66/2025.

## Ideia central

A utilidade de uma fonte externa para aumentar um dataset operacional é
prevista pela **compatibilidade estrutural** (escala, densidade, %small), não
pela similaridade visual. Medimos essa distância e a usamos para atribuir
papéis às fontes candidatas.

Resultado do perfil (Passo Zero) — distância ao CITRA-3D (menor = mais próximo):

| Fonte | distância | papel |
|---|---|---|
| ABOShips | 0.95 | fonte primária de aumento |
| SMD on-shore | 1.84 | fonte secundária |
| SeaShips | 4.08 | held-out independente |

## Estrutura

```
maritime-crossdomain/
├── configs/
│   └── datasets.yaml          # caminhos e parâmetros (edite aqui)
├── src/crossdomain/
│   ├── loaders.py             # leitura YOLO/VOC/COCO/CSV/ABOShips unificada
│   ├── profiling.py           # perfil estrutural + distância composta
│   └── prepare.py             # conversão YOLO + splits disjuntos
├── scripts/
│   ├── 00_download.py         # download das fontes (Colab)
│   ├── 01_profile.py          # Passo Zero (perfil + figuras)  [CPU]
│   └── 02_prepare_data.py     # Fase 1 (splits disjuntos)       [CPU]
├── docs/
│   └── Plano_Experimento_CrossDomain_SMD.md
├── requirements.txt
└── .gitignore
```

## Uso

```bash
pip install -r requirements.txt

# 1) editar caminhos em configs/datasets.yaml

# 2) perfil estrutural (decide papéis das fontes) — CPU
python scripts/01_profile.py --config configs/datasets.yaml \
    --out /content/drive/MyDrive/PROJETO_MARINHA/Experimento_CrossDomain/_passo_zero

# 3) preparar dados: YOLO classe única + splits disjuntos — CPU
python scripts/02_prepare_data.py --config configs/datasets.yaml
```

Os scripts também rodam no Colab (cole numa célula ou `!python scripts/...`).
Runtime **CPU** basta para profiling e preparação; a GPU/A100 só é necessária
na fase de treino.

## Convenções metodológicas

- Classe única `vessel`; classes não-embarcação excluídas por dataset
  (ex.: `seamark`, `miscellaneous` no ABOShips).
- `%small`: convenção COCO após reescala para 640 (lado maior).
- **Splits disjuntos por origem** (anti-vazamento): ABOShips por sequência de
  captura, SMD por vídeo (`MVI_id`). Splits pré-fabricados do Roboflow **não**
  são usados. Um manifesto JSON persiste as listas de IDs por split.
- SeaShips é reservado como held-out; jamais entra em treino/val/tuning.

## Dados

Datasets **não** são versionados (ver `.gitignore`). Fontes:
ABOShips (Zenodo 4736931, CC BY 4.0), SeaShips (Shao et al., 2018),
SMD (Prasad et al., 2017). CITRA-3D-Real é operacional e restrito.

## Estado

- [x] Passo Zero (perfil estrutural + atribuição de papéis)
- [x] Fase 1 (splits disjuntos + manifesto)
- [ ] Composição sintética in-place (crops SAM do ABOShips)
- [ ] Fase de treino (braços B2 / C-pre / C-joint / A′joint) — requer
      autorização do supervisor
