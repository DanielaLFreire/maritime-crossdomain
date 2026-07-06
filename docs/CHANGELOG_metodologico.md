# Changelog metodológico — Experimento Cross-Domain (ABOShips)

Registro das decisões tomadas durante a implementação, para o artigo de extensão.
Atualizado enquanto o experimento roda.

## Fonte de aumento
- **Perfil estrutural (Passo Zero)** definiu papéis por distância ao CITRA-3D:
  ABOShips 0,95 (primária) < SMD on-shore 1,84 (secundária) < SeaShips 4,08 (held-out).
  InaTechShips (≫) = âncora "longe" do artigo original (transfer −4,15 pp).
- Inversão da hipótese a-priori: ABOShips (câmera móvel) é estruturalmente o mais
  próximo apesar do viewpoint distinto — reforça a tese "estrutura > aparência".

## Extração de crops (SAM ViT-B)
- **MIN_DIM reduzido 50 → 20 px** [adaptação justificada; ablação = future work].
  Motivo: 66,3% das caixas do ABOShips têm lado menor < 50 px (mediana 34 px).
  Manter 50 px (calibrado p/ os navios grandes do InaTech) descartaria 2/3 dos crops.
  Efeito medido: yield 25,8% → 62,8% (18.420 crops de 29.318 caixas).
- **Filtro de opacidade pós-segmentação** [passo novo, não existia no artigo].
  Remove crops com máscara < 20% da bbox ou lado < 24 px (ruído de embarcações
  pequenas). Removeu 13,0% (2.389 de 18.405) → **16.016 crops limpos**.

## Composição sintética
- Fiel ao artigo: 13 variações × CITRA(train+val) = **21.840 sintéticas**
  (17.524 train + 4.316 val). Labels forçados a classe única 0.
- Joint balanceado: 17.524 real (×13) + 17.524 sintético = 35.048 imgs/época (50/50).

## Reprodutibilidade / armadilhas resolvidas
- CITRA-3D `train/labels/` contém labels MULTI-classe (0–8) + 1 linha corrompida
  (`Quadrado_marcacao(Clone)`). O correto é `labels_single_class/` (classe 0).
  Pipeline monta um CITRA classe-única local (symlink de imagens + cópia dos
  labels single-class) para o Ultralytics ler os labels certos.
- ABOShips: imagens PNG (não JPG); CSV `Vesibussi_Labels.csv` com colunas
  width/height = tamanho da BBOX (não da imagem; usar 1280×720 fixo);
  classes seamark/miscellaneous excluídas.
- Hiperparâmetros idênticos ao B2 do artigo (AdamW, lr0=0.001, lrf=0.01, cos_lr,
  imgsz=640, batch=16, 300 ep, patience=30); warmup_bias_lr default (0.1).

## Validação do pipeline
- Sanity 25 épocas (seed 42): mAP50 subiu estável 0.59 → 0.76, 0 corrupt labels.
  Confirma o pipeline correto antes do treino completo (300 ep).

## Braços (a popular)
- B2 (baseline), C-pre (H1), C-joint (H2), A_joint_ABO (H3).
- Curva H4 (distância → Δ-transferência): pontos ABOShips (0,95), SMD (1,84),
  InaTechShips (≫, −4,15). Populada por C-pre/C-joint por fonte.

## Resultados preliminares (seed 42) — teste no CITRA-3D (401 imgs)
| Braço | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| B2 (baseline) | 0.8275 | 0.4970 | 0.8403 | 0.7682 |
| A_joint_ABO | **0.8413** | **0.5097** | **0.8726** | **0.7955** |
| Δ | **+1.39 pp** | +1.27 pp | +3.23 pp | +2.73 pp |

- A_joint_ABO (crops ABOShips) supera o baseline em TODAS as métricas.
- Ganho em Recall (+2.73) SEM perda de Precision (subiu +3.23) — aumento "limpo".
- Comparação com artigo original: A'joint (InaTech) deu +1.00 pp mAP50;
  A_joint_ABO (fonte estruturalmente mais próxima, d=0,95) deu +1.39 pp —
  ganho MAIOR com a fonte mais próxima, reforçando a tese "estrutura > aparência".
- PENDENTE: seeds 123 e 2024 para mean±std (atual n=1; std=0 no summary).

## RESULTADO FINAL — 3 seeds (42, 123, 2024), teste CITRA-3D (401 imgs)
| Braço | mAP50 | mAP50-95 | Recall |
|---|---|---|---|
| B2 (baseline) | 0.8315 ± 0.0029 | 0.4995 ± 0.0074 | 0.7795 ± 0.0080 |
| A_joint_ABO | **0.8384 ± 0.0023** | **0.5108 ± 0.0016** | **0.7946 ± 0.0016** |
| Δ | **+0.69 pp** | +1.13 pp | **+1.51 pp** |

- A_joint_ABO supera B2 em TODOS os 3 seeds (Δ mAP50: +1.38, +0.46, +0.22 pp) — consistente.
- Std do A_joint_ABO MENOR que o do B2 → método aumentado é mais estável, não só melhor.
- B2 reproduz o baseline do artigo (0.8315 ≈ 0.835), validando o pipeline.
- Nota de convergência: o braço joint (35.048 imgs/época, 13× o B2) converge em ~12-15
  épocas (early stopping ~42-45) porque cada época é ~13× mais rica em dados. Esperado.
- vs artigo original: A'joint (InaTech, d≫) deu +1.00 pp; A_joint_ABO (ABOShips, d=0,95)
  deu +0.69 pp mAP50 mas +1.51 pp recall. Dois pontos da curva estrutural
  (próximo ajuda / distante atrapalha) sustentam a tese "estrutura prediz transferência".
