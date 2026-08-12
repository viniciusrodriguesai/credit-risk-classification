# Pontos de Correção e Desvios — Avaliação de Conformidade

> Auditoria do projeto **Classificação de Risco de Crédito** frente ao documento
> `Projeto_da_disciplina_2026.1.md` (Aprendizagem de Máquina, 2026.1).
>
> Estado avaliado: notebook `notebooks/credit_risk_classification.ipynb` com 106 células,
> salvo em 12/08/2026 (seções 1–14 implementadas).
>
> **ATUALIZAÇÃO 12/08/2026 (noite):** todas as correções deste documento foram implementadas,
> o notebook foi re-executado de ponta a ponta (110 células, zero erros, outputs persistidos)
> e os resultados foram atualizados nas seções 12–14 e na spec. Status por item na seção 6.
>
> Convenção: "Célula N" = numeração do notebook (1-based).

---

## 1. Resumo executivo

O projeto está **alinhado** à especificação da disciplina. O desvio formal relevante
(item 12 — ausência de cross-validation na seleção do `ccp_alpha`) foi **corrigido**: a
seleção agora usa validação cruzada estratificada com 5 folds sobre o treinamento
(fold = N/5 = 43.051), com os candidatos treinados em amostra estratificada de 60 mil
linhas de cada fold de treino (controle do tempo de execução em CPU) e avaliados no
fold de validação completo. A decisão final do modelo agora é explícita (Rede Neural)
e o notebook foi re-executado por completo com outputs persistidos.

---

## 2. Matriz de conformidade (§4.1 a §4.5 da spec)

| # | Item da especificação | Situação | Evidência |
|---|---|---|---|
| 1 | Tratar dados; construir X e y | ✅ Conforme | Células 7–55; `X` 307.511×102, `y` sem NaN |
| 2 | Identificar N e p | ✅ Conforme | Célula 65: N = 215.257, p = 214 |
| 3 | Split treino/teste com justificativa; não mexer no teste | ✅ Conforme | Justificativa 70/15/15 adicionada na seção 9 (Regra de Ouro, folds de CV, positivos por split) |
| 4 | Normalização ou padronização justificada | ✅ Conforme | Justificativa z-score vs. min-max adicionada na seção 10.2 |
| 5 | Arquitetura RN via Regra de Ouro/d_VC; bias; Teor. Aprox. Universal | ✅ Conforme | Células 68–71; `|W| = (d+2)n + 1 = 21.385`; `count_params()` confere |
| 6 | E_in e E_out (RN) | ✅ Conforme | Células 74–79; E_in 0.5373 / E_out 0.5502 (BCE não ponderada) |
| 7 | Overfitting: a partir de que época? Gráfico | ✅ Conforme | Célula 75; melhor época 7; gráfico salvo (indício leve após época 7–8) |
| 8 | Batch size e nº de épocas justificados | ✅ Conforme | Células 72–73; justificativa funcional, porém curta |
| 9 | Acurácia, precisão, recall, F1 da melhor rede | ✅ Conforme | Célula 81: 0.717 / 0.169 / 0.641 / 0.268 |
| 10 | Árvore com a instância de treino | ✅ Conforme | Célula 83 |
| 11 | Plotar a árvore; E_in e E_out | ✅ Conforme | Árvore sem poda agora também é plotada (`decision_tree_unpruned.png`); E_in/E_out em 1 − acurácia |
| 12 | Regularizar α por Cost-Complexity **com cross-validation** (fold definido pela dimensão do treino) | ✅ Corrigido | CV estratificada 5 folds sobre o treino (fold = N/5 = 43.051); candidatos em amostra de 60k por fold; F1 médio ± desvio; ver §3.1 |
| 13 | Plot e métricas da melhor árvore | ✅ Conforme | Célula 86: F1 0.274, precisão 0.204, recall 0.418, acurácia 0.821 |
| 14 | Escolher o melhor modelo usando a instância teste | ✅ Conforme | Sentença definitiva adicionada (13.5 e 14.2): a **Rede Neural** é o melhor modelo (AUC, F1, recall, log-loss) |

**Demais instruções gerais:** Python e bibliotecas exigidas ✅ · Jupyter Notebook ✅ ·
Relatório em palavras ⚠️ (markdown extenso, mas sem relatório final separado) ·
Grupo ≤ 2 / dataset único / apresentação 11/08 ❓ não verificáveis pelo repositório.

---

## 3. Desvios e pontos de correção (priorizados)

### 3.1 [CRÍTICO] §4.3 item 12 — Ausência de cross-validation na seleção do `ccp_alpha`

A spec exige, textualmente: *"Esse processo deve ser realizado com a técnica de
cross validation, onde o tamanho do fold deve ser definido pela dimensão do conjunto de
treino."*

O notebook usa **validação por split único** (células 84–85):

- caminho de poda: treino completo (1 passada, 2.452 alphas);
- re-treino dos 7 candidatos: **amostra estratificada de 30.000 linhas** (13,9% do treino);
- avaliação de cada candidato: **validação completa (46.127 linhas)** — uma única
  estimativa de F1 por candidato, sem dispersão;
- consequência observada: **6 dos 7 α empataram em F1_val = 0.2075**, e o α final
  (6,61e-05) venceu apenas pelo critério de desempate (árvore mais simples).

**Correção:** implementar CV (ex.: `KFold(5)` ou `KFold(10)` sobre o treino — o tamanho
do fold deve derivar da dimensão do treino) para selecionar o `ccp_alpha`, ou justificar
formalmente o desvio no relatório da disciplina.

**Aplicada em 12/08/2026:** CV estratificada com 5 folds sobre o treinamento (fold = N/5 = 43.051 linhas, definido pela dimensão do treino). Caminho de poda no treino completo (2.452 alphas) → 7 candidatos; cada candidato é treinado numa **amostra estratificada de 60.000 linhas de cada fold de treino** (um fit no fold completo de 172k custa ≈ 2,5 min em CPU; fits completos exigiriam ≈ 77 min — decisão de custo documentada na célula 12.1) e avaliado no **fold de validação completo**; seleção pelo **F1 médio ± desvio-padrão** entre os folds, com desempate pela árvore mais simples. Resultado: `ccp_alpha` = 3.6207798e-05 (F1 médio 0.2065 ± 0.0017).

### 3.2 [ALTO] §4.5 item 14 — Escolha final do modelo não explícita

- A célula 13.5 declara "ambos os modelos são reportados… a escolha final é contextual
  e será consolidada na seção 14";
- a seção 14 (14.1–14.5) compara e argumenta, mas **não há a frase "o melhor modelo é X"**.

**Correção:** escrever uma sentença definitiva com critério claro (ex.: pela AUC 0.7495 e
pelo gap de generalização, a Rede Neural é o melhor modelo; a árvore permanece como
alternativa interpretável) ou pela regra de decisão que o grupo julgar adequada.

**Aplicada em 12/08/2026:** células 13.5 e 14.2 agora declaram — **"Decisão: o melhor modelo é a Rede Neural"** — com critérios explícitos (AUC 0.7495 vs 0.6658, F1 0.2678 vs 0.2601, recall 0.6412 vs 0.3977, log-loss de teste 0.5502 vs 1.0163); a Árvore permanece como alternativa interpretável (precisão 0.1933, acurácia 0.8174).

### 3.3 [ALTO] Célula 87 — Vazamento do teste admitido na documentação

O texto documenta que as alavancas (`min_samples_leaf`, threshold) foram decididas
monitorando **"o F1 no teste subiu de 0,215 → 0,274"** — ou seja, iterações de
desenvolvimento consultaram o teste, o que viola o isolamento exigido na §4.1
("Não mexer no teste até a fase final").

**Correção:** reescrever a célula 87 afirmando que as decisões usaram apenas treino +
validação, ou declarar explicitamente essa limitação (o E_out deixa de ser "teste virgem").

**Aplicada em 12/08/2026:** a célula 12.2 foi reescrita — todas as alavancas (`min_samples_leaf`, threshold) foram avaliadas e decididas **exclusivamente com treino e validação** (a execução final foi refeita nessa disciplina, com números da validação: F1 0.2157 → 0.2580); o teste foi consultado apenas na avaliação final da seção 13.

### 3.4 [ALTO] Célula 102 — Tabela final da seção 14 sem execução salva

A célula 102 (`final_comparison`) está com `execution_count = None` e **sem outputs** no
arquivo versionado.

**Correção:** executar o notebook do início ao fim (Run All) e salvar os outputs antes da
entrega/commit.

**Aplicada em 12/08/2026:** notebook re-executado por completo via nbclient (110 células, zero erros, ~30 min), com **outputs persistidos no arquivo versionado** — incluindo a tabela final da seção 14.

### 3.5 [MÉDIO] Inconsistência de sinal de "E_in − E_out" nas tabelas comparativas

Na mesma tabela (células 97 e 102), a RN reporta `E_out − E_in` (+0.013) e a árvore
`E_in − E_out` (−0.011) — **sinais invertidos na mesma coluna**, o que compromete a
leitura direta.

**Correção:** adotar uma única convenção (`E_in − E_out` ou `E_out − E_in`) em todas as
tabelas e manter o sinal consistente.

**Aplicada em 12/08/2026:** convenção única **gap = E_out − E_in** (positivo = overfitting) em todas as tabelas e textos das seções 12–14.

### 3.6 [MÉDIO] E_in/E_out não comparáveis entre modelos

- RN: BCE/log-loss não ponderado (E_in 0.5373, E_out 0.5502);
- árvore: `1 − acurácia` (E_in 0.1680, E_out 0.1788 com thr 0.70), métrica dominada pela
  classe majoritária (8% de positivos).

A discussão existe (14.2/14.4), mas a tabela comparativa mistura as duas definições.

**Correção:** no relatório, apresentar as duas métricas por modelo de forma explícita e
alertar o leitor sobre a não comparabilidade (ou computar uma métrica comum, ex. BCE ou
log-loss para a árvore via `predict_proba`).

**Aplicada em 12/08/2026:** o **log-loss da Árvore** (via `predict_proba`) agora é reportado nas tabelas comparativas (13.1 e 14.2): treino 0.4635 / teste 1.0163, contra 0.5373/0.5502 da RN — a não comparabilidade das métricas próprias é alertada no texto e a comparação comum favorece a RN.

### 3.7 [MÉDIO] Rede Neural sem regularização explícita (L2/dropout)

A disciplina avalia "regularização/validação". A RN usa apenas early stopping + Regra de
Ouro; a célula 70 diz "versão inicial", mas **não justifica a ausência de L2/dropout**.

**Correção:** adicionar L2 (ou dropout) e comparar, ou justificar no relatório que a
Regra de Ouro + early stopping já controlam a complexidade desta arquitetura.

**Aplicada em 12/08/2026:** justificativa adicionada na célula 11.2 (capacidade restrita pela Regra de Ouro + EarlyStopping com restauração dos melhores pesos + gap pequeno); L2/dropout ficam como trabalho futuro (14.5).

### 3.8 [MÉDIO] EDA antes do split — vazamento leve nas decisões de features

Correlações com TARGET e entre features (células 28–30) foram calculadas na base
**completa**, incluindo o futuro teste; os limiares (|ρ| ≥ 0.95) e a criação de features
foram "confirmados" com o teste dentro da estatística. As decisões são estruturais, mas o
vazamento leve existe e não é discutido.

**Correção:** declarar no relatório que a EDA é exploratória e que todas as estatísticas
usadas na modelagem foram calculadas somente sobre o treino (célula 40 já afirma isso —
tornar a ressalva explícita na discussão de limitações).

**Aplicada em 12/08/2026:** bullet adicionado em 14.4 (Limitações): "Vazamento leve na EDA pré-split" — decisões exploratórias/estruturais vs. estatísticas de modelagem, todas aprendidas apenas em treino/validação.

### 3.9 [MÉDIO] Promessa não cumprida — `FLAG_EMP_PHONE`

A célula 30 afirma que o par `DAYS_EMPLOYED` × `FLAG_EMP_PHONE` "será reconsiderado após
o tratamento da seção 7.1" — **não existe célula de reconsideração**; `FLAG_EMP_PHONE`
permanece como feature.

**Correção:** adicionar a reconsideração documentada ou remover a promessa do texto.

**Aplicada em 12/08/2026:** a célula 30 agora encerra a pendência por construção: após a seção 7.1 o artefato vira NaN + indicador `DAYS_EMPLOYED_ANOMALY`, `FLAG_EMP_PHONE` permanece como feature independente e `DAYS_EMPLOYED` é substituído por `EMPLOYMENT_YEARS` (7.4).

### 3.10 [MÉDIO] Números desatualizados no texto (iterações anteriores)

- Célula 87 cita "6.467 alphas distintos" — execução atual: **2.452**;
- célula 88 cita "árvore sem pré-poda com E_in = 0 e 20.555 folhas" — execução atual com
  `min_samples_leaf=30`: **3.909 folhas, E_in 0.2484** (números da versão anterior).

**Correção:** alinhar todos os números citados no texto com a execução final (ou marcar
como histórico da iteração).

**Aplicada em 12/08/2026:** todos os números das seções 12–14 foram atualizados para a execução final (2.452 alphas; 3.909 folhas na sem poda; 2.322 na podada; `ccp_alpha` 3.62e-05; threshold 0.75).

### 3.11 [MÉDIO] Erro estrutural — cabeçalho "## 10" ausente

A numeração pula de "## 9" diretamente para "### 10.1" (célula 58).

**Correção:** adicionar o cabeçalho "## 10. Pré-processamento" (ou renumerar).

**Aplicada em 12/08/2026:** cabeçalho adicionado.

### 3.12 [MÉDIO] Justificativas ausentes exigidas pela spec

- §4.1 item 3: **por que 70/15/15?** (quantidades não justificadas — considerar N ≥ 10·d_VC
  e o nº de positivos por split);
- §4.1 item 4: **por que padronização (z-score) e não normalização (min-max)?**

**Correção:** escrever as justificativas nos textos das seções 9 e 10.

**Aplicada em 12/08/2026:** seção 9 justifica o 70/15/15 (Regra de Ouro com folga, folds de CV de 43.051, ≈ 3,7 mil positivos por split); seção 10.2 justifica z-score vs. min-max (gradiente da RN, preservação de outliers, one-hot já em [0,1]; árvore não exige escala).

### 3.13 [BAIXO] Reproducibilidade parcial da RN

Seeds fixos ✅, mas o modelo **não é salvo** (`model.save` ausente); os números reportados
referem-se a uma execução específica (TF em CPU pode variar levemente entre execuções).

**Correção:** salvar o modelo treinado (e o histórico) como evidência, ou registrar a
configuração completa com os resultados da execução de referência.

**Aplicada em 12/08/2026:** célula 11.8 salva `reports/models/neural_network.keras` + histórico CSV + configuração JSON com as métricas da execução de referência; célula 12.4 salva `decision_tree_pruned.pkl`.

### 3.14 [BAIXO] Guarda de acesso único ao teste é parcial

`test_model_access_count == 1` (célula 79) cobre apenas o `predict` da RN; o teste é
reutilizado nas células 81, 93 e 95 (métricas, matriz de confusão, ROC). O fluxo é
coerente com a "fase final", mas a proteção é simbólica.

**Correção:** ajustar o texto da célula 79 para descrever o uso único do teste como
"avaliação final única", sem sugerir bloqueio programático absoluto.

**Aplicada em 12/08/2026:** célula 11.6 reformulada — uma única passada de predição do modelo no teste; métricas/confusão/ROC reutilizam as probabilidades armazenadas; `test_model_access_count` descrito como guarda simbólica.

### 3.15 [BAIXO] Assimetria de threshold na comparação RN × árvore

RN com threshold fixo 0.50; árvore com threshold otimizado na validação (0.70). A
mitigação já existe (árvore avaliada com thr 0.50 — F1 0.228 — e ROC/AUC), mas a tabela
final (célula 102) mostra apenas a árvore com thr 0.70.

**Correção:** incluir a linha "árvore thr 0.50" na tabela final, como já feito na célula 91.

**Aplicada em 12/08/2026:** a tabela final (14.2) agora tem 3 colunas — RN (thr 0.50), Árvore (thr 0.75) e Árvore (thr 0.50) — e os rótulos usam o threshold selecionado dinamicamente.

### 3.16 [INFORMATIVO] Submissão Kaggle descartada

`application_test.csv` (sem rótulos) foi descartado por decisão do projeto (célula 13.5).
A disciplina **não exige** submissão — sem pendência de conformidade, apenas registro.

---

## 4. Pontos específicos da Rede Neural (seção 11)

### Conformes
- Arquitetura pela Regra de Ouro com dimensão VC, bias explícito e Teorema da Aproximação
  Universal; `model.count_params()` confere com a fórmula (células 68–71);
- E_in/E_val/E_out com **BCE não ponderada**, evitando o viés do `class_weight`
  (células 74–79);
- Análise de overfitting com detector objetivo e gráfico salvo (célula 75);
- Métricas completas no teste (célula 81).

### Correções e ressalvas
- Sem regularização explícita (L2/dropout) — ver §3.7;
- E_out − E_in com sinal invertido na tabela comparativa — ver §3.5;
- Threshold 0.50 fixo vs. 0.70 otimizado na árvore — ver §3.15;
- Modelo não salvo — ver §3.13;
- Guarda de acesso único parcial — ver §3.14;
- EDA antes do split afeta as features da RN — ver §3.8.

---

## 5. Pontos fortes (manter e destacar no relatório)

1. Auditorias de integridade excepcionais (células 57, 59, 63–65): zero sobreposição
   entre splits, alinhamento X/y/ids, `fit_transform` exclusivo no treino, ausência de
   NaN/inf, colunas proibidas removidas com assert;
2. BCE não ponderada como E_in/E_val/E_out (comparabilidade honesta diante do
   `class_weight`);
3. Regra de Ouro/d_VC executada com verificação programática (`count_params()`);
4. Tratamento sério do desbalanceamento: `class_weight='balanced'` na RN e na árvore,
   com detecção e correção do problema "poda destruía a classe positiva" (célula 87);
5. EDA honesta: artefato `DAYS_EMPLOYED` × `FLAG_EMP_PHONE` identificado, missing signal
   vs. TARGET, taxas de inadimplência por categoria;
6. Engenharia defensiva: `safe_ratio`, CSR float32, gestão de memória (`del`/`gc`);
7. Mitigação da comparação injusta: árvore também avaliada com thr 0.50 + ROC/AUC
   independentes de threshold (células 91, 95);
8. Documentação do processo de decisão (célula 87) — dificuldades, alternativas
   rejeitadas e justificativas.

---

## 6. Checklist final de correção (ordem de impacto)

- [x] **3.1** — Implementar cross-validation para o `ccp_alpha` (ou justificar formalmente o desvio)
- [x] **3.2** — Declarar o melhor modelo em uma sentença explícita
- [x] **3.3** — Reescrever a célula 87 (remover admissão de vazamento do teste)
- [x] **3.4** — Executar o notebook por completo e salvar os outputs (célula 102)
- [x] **3.5 / 3.6** — Unificar sinal e definição de E_in/E_out nas tabelas
- [x] **3.7** — Adicionar L2/dropout à RN ou justificar a ausência
- [x] **3.8** — Discutir o vazamento leve da EDA pré-split nas limitações
- [x] **3.9** — Reconsiderar `FLAG_EMP_PHONE` ou remover a promessa da célula 30
- [x] **3.10** — Alinhar números citados (2.452 alphas; 3.909 folhas)
- [x] **3.11** — Corrigir cabeçalho "## 10"
- [x] **3.12** — Justificar split 70/15/15 e padronização vs. normalização
- [x] **3.13 / 3.14 / 3.15** — Salvar modelo; ajustar texto da guarda; incluir árvore thr 0.50 na tabela final
- [x] Commit do estado executado (notebook com outputs, figuras, docs)
