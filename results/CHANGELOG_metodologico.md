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

## Terminologia — cadeia de transfer learning (para o texto do artigo)
Cada braço é uma cadeia de treinos sequenciais partindo de pesos prévios.
Convenção adotada (alinhada à literatura):
- **B2** (baseline): COCO (pré-treino base) → **fine-tune** CITRA-3D (300 ép).
- **C-pre** (H1): COCO → **pré-treino** ABOShips (100 ép) → **fine-tune** CITRA-3D (300 ép).
  Três elos: COCO (base de fábrica) → ABOShips (pré-treino de domínio) → CITRA (fine-tune alvo).
- **C-joint** (H2): COCO → fine-tune conjunto CITRA-3D + ABOShips real (300 ép).
- **A_joint_ABO** (H3): COCO → fine-tune conjunto CITRA-3D + sintético-ABOShips (300 ép).

Nota: "pré-treino" reservado ao estágio intermediário (ABOShips); "fine-tune" ao
estágio final no alvo (CITRA). O COCO é sempre a base pré-treinada de fábrica.
As 300 épocas finais no CITRA são idênticas em todos os braços — isola o efeito
da fonte/estratégia, não do número de épocas.

## Resultados C-pre / C-joint (seed 42) — ABOShips REAL vs sintético
Teste CITRA-3D (401 imgs), seed 42:
| Braço | ABOShips via | mAP50 | Recall | Δ mAP50 vs B2 |
|---|---|---|---|---|
| B2 (baseline) | — | 0,8275 | 0,7682 | — |
| A_joint_ABO | sintético | 0,8413 | 0,7955 | **+1,38 pp** |
| C-joint | real (co-treino) | 0,8298 | 0,7674 | +0,23 pp |
| C-pre | real (pré-treino) | 0,8275 | 0,7650 | +0,00 pp |

**Achado central:** o ABOShips só gera ganho substancial quando entra via
**síntese domain-adapted** (A_joint_ABO). Como dados REAIS — pré-treino (C-pre)
ou co-treino (C-joint) — o ganho é nulo/marginal, e o recall fica ABAIXO do
baseline. Refina a tese: proximidade estrutural é NECESSÁRIA (ABOShips funciona,
InaTech distante não) mas NÃO SUFICIENTE — a adaptação de domínio via composição
in-place é o ingrediente crítico. Justifica a contribuição metodológica (síntese)
sobre "apenas adicionar mais dados reais".
PENDENTE: seeds 123, 2024 de C-pre/C-joint para mean±std (atual n=1).

## SeaShips (held-out para generalização) — fonte das anotações
Problema: a cópia no Drive tinha só JPEGImages (7000 JPGs, 0 XMLs).
Fonte oficial COM anotações VOC (Shao et al., 2018 — usar esta, não Roboflow):
- WHU (direto): http://www.lmars.whu.edu.cn/prof_web/shaozhenfeng/datasets/SeaShips(7000).zip
- Repo: github.com/jiaming-wang/SeaShips | Baidu: pan.baidu.com/s/1iQ-JG-4JhyBkdUFwIclprg (senha 986Z)
Estrutura: JPEGImages/ + Annotations/ (XMLs) + ImageSets/. O prepare_seaships_heldout
converte VOC→YOLO classe única. Apontar configs SeaShips.dir para a pasta completa.
Citação: Shao et al., IEEE Trans. Multimedia 20(10):2593-2604, 2018.

### SeaShips — atualização: WHU falhou, usar Roboflow
O link direto da WHU retorna página de erro (~53KB HTML, servidor instável).
Via que funciona no Colab — Roboflow API (formato VOC com XMLs):
  rf.workspace("ships-tznqe").project("seaships7000-yxbuv").version(1).download("voc")
Requer API key grátis (roboflow.com). Roboflow é só o MEIO de obtenção;
CITAR a fonte original Shao et al. 2018 (não o re-upload).

### SeaShips — RESOLVIDO
Baixado via Roboflow (ships-tznqe/seaships7000-yxbuv, formato VOC): 6.979 imagens
com XMLs. Salvo no Drive: Experimento_CrossDomain/SeaShips_voc_completo.
Apontar configs SeaShips.dir para essa pasta. Conferir se o Roboflow redimensionou
as imagens (documentar se sim). Citar Shao et al. 2018 (fonte original).

## RESULTADO FINAL — 4 braços × 3 seeds (teste CITRA-3D)
| Braço | ABOShips via | mAP50 | Recall | Δ mAP50 |
|---|---|---|---|---|
| A_joint_ABO | sintético | 0,8384 ± 0,0023 | 0,7946 ± 0,0016 | +0,69 pp |
| B2 (baseline) | — | 0,8315 ± 0,0029 | 0,7795 ± 0,0080 | — |
| C-joint | real (co-treino) | 0,8310 ± 0,0009 | 0,7732 ± 0,0042 | −0,05 pp |
| C-pre | real (pré-treino) | 0,8279 ± 0,0007 | 0,7631 ± 0,0044 | −0,37 pp |

CONFIRMADO com 3 seeds (std ≤ 0,003): só A_joint_ABO (síntese) supera o baseline.
ABOShips REAL — pré-treino (C-pre) ou co-treino (C-joint) — NÃO ajuda; C-pre chega
a degradar o recall (−1,64 pp). Ordenamento A_joint_ABO > B2 > C-joint > C-pre
estável nos 3 seeds. Tese refinada: proximidade estrutural é NECESSÁRIA mas NÃO
SUFICIENTE; a adaptação de domínio via composição sintética é o ingrediente crítico.
Hipótese p/ o texto: pré-treino/co-treino com ABOShips real enviesa o modelo para
embarcações grandes (a escala do ABOShips), degradando o recall nas pequenas do CITRA;
a síntese in-place preserva a escala/contexto do CITRA, por isso funciona.

## Figura H4 — interação via×distância (interpretação confirmada com o artigo)
Ler o artigo original corrigiu a tese. O artigo decompõe o MECANISMO com UMA
fonte (InaTech, distante): −4,15 (real) → −1,31 (síntese seq) → +1,00 (síntese joint).
A extensão ABOShips adiciona o EIXO DISTÂNCIA (2ª fonte, próxima 0,95):
| Fonte | dist | via real (pré-treino) | via síntese joint |
|---|---|---|---|
| ABOShips | 0,95 | −0,37 (C-pre) / −0,05 (C-joint) | +0,69 (A_joint_ABO) |
| InaTech  | ≫    | −4,15 (Arm A, artigo)          | +1,00 (A' joint, artigo) |

ACHADO da extensão: a síntese joint é ROBUSTA à distância (ambas > 0: +0,69 e +1,00);
o uso de dados reais DEGRADA com a distância (−0,37 perto → −4,15 longe). As duas
vias divergem conforme a fonte se distancia. Tese: a composição in-place neutraliza
a distância estrutural; o uso real não. Estende (não contradiz) o artigo — confirma
que o mecanismo síntese-joint vale variando a fonte, e isola a distância como o que
governa a severidade do transfer negativo real.
Fig: docs/fig_h4_interacao.{png,pdf} (gerada por docs/gerar_fig_h4.py).
NOTA: distância do InaTech plotada como "≫" (sem valor numérico do Passo Zero).

### SeaShips held-out — notas do formato (Roboflow)
- Imagens REDIMENSIONADAS para 640×640 pelo Roboflow (documentar no artigo:
  "SeaShips images resized to 640×640 by the distribution source"). O parser lê
  <size> do XML, então as bboxes convertem corretamente.
- SeaShips é MULTI-CLASSE (bulk cargo carrier, container ship, etc.);
  prepare_seaships_heldout colapsa TODAS as classes em classe 0 ("vessel") —
  correto para a avaliação single-class.
- Contagem de XMLs (13.120) pode incluir duplicatas do merge train+test com
  hashes do Roboflow; o parser gera 1 label por imagem válida. Documentar o total
  real de imagens avaliadas (n do seaships_heldout/labels).
- Citar Shao et al. 2018 (fonte original), não o Roboflow.

## GENERALIZAÇÃO zero-shot SeaShips (held-out, dist 4,08, 6.979 imgs)
| Braço | mAP50 | Recall | Δ mAP50 vs B2 |
|---|---|---|---|
| C-joint | 0,4395 ± 0,0143 | 0,3872 | +4,75 pp |
| A_joint_ABO | 0,4184 ± 0,0179 | 0,4445 | +2,63 pp |
| B2 (baseline) | 0,3920 ± 0,0604 | 0,3781 | — |
| C-pre | 0,3041 ± 0,0083 | 0,2887 | −8,79 pp |

LEITURA HONESTA (resultado matizado — NÃO force-fit na narrativa in-domain):
- In-domain (CITRA): A_joint_ABO (síntese) lidera.
- Zero-shot distante (SeaShips): C-joint (co-treino REAL) ligeiramente à frente do
  A_joint_ABO; AMBOS superam o B2. A síntese NÃO domina no held-out distante.
- A_joint_ABO tem o MELHOR RECALL (0,4445) — relevante p/ vigilância.
- C-pre despenca em ambos (−8,79 aqui) — pré-treino direto prejudica sempre.
- Variância do B2 alta (±0,0604): consistente com o artigo (SMD: "seed variance grows").
TESE DEFENSÁVEL: a fonte próxima (ABOShips) ajuda a generalização quando usada de
forma BALANCEADA (joint), seja síntese ou real — não por pré-treino. A vantagem
específica da síntese é clara IN-DOMAIN; no held-out muito distante, síntese e
co-treino real convergem. Reportar honestamente; NÃO afirmar que a síntese sempre vence.
PENDENTE: discutir com Cmte. Moreira como enquadrar (o C-joint>A_joint no held-out
é inesperado; pode virar ponto de discussão rico, não fraqueza).
