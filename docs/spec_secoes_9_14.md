# Spec — Seções 9 a 14 + Submissão (notebook `credit_risk_classification.ipynb`)

> Documento vivo: seções 9 e 10 já estão **implementadas e validadas** (70 células, execução de ponta a ponta OK). Este spec define o estado atual (contratos), o que falta (seções 11–14 + submissão) e como validar. Decisões acordadas com o colega em 11/08/2026: **teste interno 70/15/15** (avaliação concreta no teste) e **`application_test.csv` apenas para submissão** ao final. Em 11/08/2026 foram adicionadas as features da solução vencedora do Kaggle: `LTV`, `DOWN_PAYMENT` e `EXT_SOURCE_MEAN`.

## 0. Estado de referência (verificado por execução completa das células 5–64)

| Objeto | Valor esperado |
|---|---|
| `application_model` | 307.511 × 106 |
| `X` | 307.511 × 102 (47 float64 / 37 int64 / 16 object / 2 int8) |
| `y` | 307.511, sem NaN, valores ⊆ {0, 1} |
| `ids` | `SK_ID_CURR`, 307.511 valores únicos |
| NaN em X | 48 colunas, 4.176.439 valores no total |
| Categóricas em X | 16 (139 categorias; `ORGANIZATION_TYPE` com 58) |
| `inf` em X | 0 |
| Split | Treino 215.257 (70%, 8,0727%) / Validação 46.127 (15%, 8,0734%) / Teste 46.127 (15%, 8,0734%) — 0 de sobreposição |
| Filtro train-only (10.1) | 7 por ausência ≥60% + 1 constante (`FLAG_DOCUMENT_12`) → 94 features (79 num / 15 cat) |
| Matrizes finais | CSR `float32`, N = 215.257, **p = 214**, 0 NaN / 0 inf, fit somente no treino |
| Teste externo | `data/raw/application_test.csv`, 48.744 × 121, **sem `TARGET`** |
| Datasets auxiliares | `bureau.csv` (170 MB) e `previous_application.csv` (405 MB) — fora do escopo (Etapa 2 do README) |

**Nota metodológica:** toda avaliação de desempenho (métricas, E_in/E_out, comparação) acontece no **teste interno** (46.127 linhas rotuladas). O `application_test.csv` não tem rótulos e será usado **uma única vez**, ao final, para predições de submissão.

**Features do contrato (nomes alinhados com o colega):** `AGE_YEARS`, `EMPLOYMENT_YEARS`, `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_ANNUITY_RATIO`, `CREDIT_PER_FAMILY_MEMBER`, `CHILDREN_PER_FAMILY_MEMBER`, `LTV`, `DOWN_PAYMENT`, `EXT_SOURCE_MEAN`, `EXT_SOURCE_1_MISSING`, `DAYS_EMPLOYED_ANOMALY` (este último criado na 7.1).

**Features adicionadas (solução vencedora do Kaggle):**
- `LTV` = `AMT_CREDIT / AMT_GOODS_PRICE` (loan-to-value; 278 ausentes, 0 inf — min 0,15, max 6,0).
- `DOWN_PAYMENT` = `AMT_GOODS_PRICE - AMT_CREDIT` (entrada implícita; pode ser negativa; 278 ausentes).
- `EXT_SOURCE_MEAN` = média de `EXT_SOURCE_1/2/3` (172 ausentes — só quando as três faltam).

---

## 1. Seções 7–10 — IMPLEMENTADO (contrato comum)

Fluxo validado: 7 (correções determinísticas + variantes + 9 atributos com `safe_ratio`) → 8 (`X`/`y`/`ids`, 99 colunas) → 9 (split 70/15/15 com bateria de validação) → 10 (filtros train-only, pipelines RN/árvore, auditoria fit/transform, N e p).

### Regras do contrato (não alterar sem acordo com o colega)
- Seção 7: apenas correções determinísticas e decisões estruturais (nomes de colunas). Nenhuma estatística aprendida dos dados.
- Seção 10.1: filtros ≥60% de ausência + constantes **somente no treino**, aplicados aos três conjuntos.
- Seções 11–14: cada implementador edita **apenas as suas seções**; mudanças nas seções 1–10 passam por revisão conjunta.

---

## 2. Seção 11 — Rede Neural (colega) — IMPLEMENTADO E VALIDADO (12/08/2026)

- **Reprodutibilidade:** `tf.random.set_seed(RANDOM_STATE)`, `np.random.seed(RANDOM_STATE)`, `random.seed(RANDOM_STATE)` antes de construir o modelo.
- Arquitetura conforme README: entrada (dimensão **p = 214**, `sparse=True`), 1 camada oculta ReLU (99 neurônios), 1 neurônio de saída sigmoide, `binary_crossentropy`, Adam (lr=1e-3), batch 256, early stopping na `val_loss`.
- `class_weight='balanced'` no treino; **BCE não ponderada** (`bce_unweighted`) usada para E_in/E_out comparáveis.
- Teste acessado **uma única vez** (modelo congelado); `E_in`, `E_val`, `E_out` via `log_loss`.

**Resultados reais (execução integrada 12/08/2026, threshold 0.50):**
| métrica | valor |
|---|---|
| E_in / E_out | 0.5373 / 0.5502 (gap 0.0129) |
| acurácia | 0.717 |
| precisão | 0.169 |
| recall | 0.641 |
| F1 | 0.268 |
| melhor época | 7 (17 executadas) |
| parâmetros | 21.385 |

Figuras: `reports/figures/neural_network_loss.png` e `neural_network_confusion_matrix.png`.

## 3. Seção 12 — Árvore de Decisão (nós) — IMPLEMENTADO (12/08/2026)

1. Árvore **sem poda** em `X_train_tree` (CSR float32): profundidade, nº de folhas, E_in, E_out (teste), E_in − E_out.
2. **Minimal Cost-Complexity Pruning**: `cost_complexity_pruning_path` no treino → candidatos de `ccp_alpha` → seleção por validação (CV interna no treino, se necessário) → preferir árvore mais simples em empate → treinar podada → métricas no teste.
3. Importância de features; salvar a árvore final em `reports/figures/`.

**Decisão metodológica (revisada em 12/08/2026 — correções da auditoria):** como a classe positiva é minoritária (~8%), as árvores usam `class_weight='balanced'`. O `ccp_alpha` é selecionado por **validação cruzada estratificada com 5 folds sobre o treinamento** (fold = N/5 = 43.051, definido pela dimensão do treino, conforme a spec): o caminho de poda é calculado no treino completo (uma passada), os 7 candidatos são treinados numa **amostra estratificada de 60.000 linhas de cada fold de treino** (controle do tempo de execução em CPU — um fit no fold completo de 172k custa ≈ 2,5 min e a CV com fits completos exigiria ≈ 77 min) e avaliados no **fold de validação completo**; a seleção usa o **F1 médio ± desvio-padrão** entre os folds, com desempate pela árvore mais simples. Pré-poda `min_samples_leaf=30` e **threshold de decisão otimizado por F1 na validação** (custo zero). O E_in usa o mesmo threshold do E_out.

**Resultados validados (execução 12/08/2026 com a CV — correções da auditoria):**
| | sem poda (msl=30) | podada + thr 0.75 |
|---|---|---|
| profundidade | 30 | 30 |
| folhas | 3.909 | 2.322 |
| E_in | 0.2484 | 0.1535 |
| E_out (teste) | 0.3100 | 0.1826 |
| E_out − E_in | 0.0616 | 0.0292 |
| recall (teste) | 0.5255 | 0.3977 |
| precisão (teste) | 0.1351 | 0.1933 |
| F1 (teste) | 0.2149 | **0.2601** |

**Seleção por CV:** caminho de poda no treino completo com 2.452 alphas; 7 candidatos avaliados em 5 folds estratificados (fold-val = 43.052); melhor F1 médio = 0.2065 ± 0.0017 no `ccp_alpha` = **3.6207798e-05** (desempate pela árvore mais simples: 1.005 folhas médias). Threshold 0.75 selecionado na validação (F1 0.2580 vs 0.2157 no thr 0.50; precisão 0.1912 vs 0.1321). O `ccp_alpha` + threshold reduzem a complexidade (3.909 → 2.322 folhas) e o gap (0.0616 → 0.0292). Figuras em `reports/figures/decision_tree_unpruned.png` e `decision_tree_pruned.png` (max_depth=3).

### 3.1 Dificuldades encontradas (documentação do processo)

1. **Custo de re-treino dos candidatos de poda.** Avaliar o `ccp_alpha` re-treinando cada candidato no treino completo (215.257×214) é proibitivo: um único fit custa ≈ 2,5 min em CPU e a CV com fits no fold de treino completo (172.206 linhas, 5 folds × 7 candidatos) exigiria ≈ 77 min. **Solução:** o caminho de poda (`cost_complexity_pruning_path`) é calculado **uma única vez** no treino completo (barato, uma passada), e cada candidato é treinado numa **amostra estratificada de 60.000 linhas de cada fold de treino**, avaliado no **fold de validação completo** (43.052). O custo cai para ≈ 30 min e a seleção ganha dispersão (F1 médio ± desvio).
2. **Poda destruía a classe positiva.** Sem `class_weight`, o `ccp_alpha` tende a podar em direção à classe majoritária: a árvore "podada" (27 folhas) tinha recall **0.0075** no teste (praticamente nunca previa a classe 1) — um resultado metodologicamente pobre. **Solução:** `class_weight='balanced'` em todas as árvores (sem poda, candidatos e final), alinhado ao desbalanceamento (~8% de positivos).
3. **Caminho de poda em amostra escolhia `ccp_alpha=0`.** Inicialmente o caminho foi calculado numa amostra de 30k; como a árvore da amostra é muito menor que a do treino completo, o "melhor" F1 de validação correspondia a **não podar** (alpha=0), anulando a etapa de poda. **Solução:** calcular o caminho na **árvore sem poda já ajustada no treino completo** (2.452 alphas distintos), usando as amostras dos folds apenas para re-treinar candidatos.
4. **Tempo de execução da seção.** A árvore sem poda (90 de profundidade, 20.555 folhas) e a final levam ~5 min cada no treino completo. Mitigado rodando em background (Agg headless) e limitando candidatos a 7.
5. **Limpeza de outputs em execução headless.** O executor de validação não persiste outputs no `.ipynb` (apenas valida a execução); os outputs reais são gerados no Run All do VS Code. Isso é esperado no fluxo do projeto.

### 3.2 Conclusões da seção 12 (a consolidar na seção 14)

- **Pré-poda controla a memorização.** Com `min_samples_leaf=30`, a árvore tem profundidade 30, 3.909 folhas e E_out − E_in = 0.0616 no teste. O `ccp_alpha` selecionado pela CV (3.62e-05) enxuga para 2.322 folhas com E_out − E_in = **0.0292** — gap pequeno, generalização adequada.
- **Threshold de decisão é uma alavanca barata e efetiva.** Otimizar o threshold por F1 na validação (0.75) elevou o F1 na validação de 0.2157 → 0.2580 e a precisão de 0.1321 → 0.1912; no teste, o F1 ficou em **0.2601** (contra 0.2179 com thr 0.50). O recall caiu para 0.3977 — trade-off esperado do threshold alto.
- **Trade-off acurácia × recall.** O threshold alto troca recall por precisão: o modelo classifica como "dificuldade" apenas casos com alta confiança (prob ≥ 0.70), gerando menos falsos positivos. No problema (custo de um falso positivo é alto), isso é desejável.
- **A classe minoritária é intrinsecamente difícil.** Mesmo balanceada e otimizada, a árvore tem F1 ~0.27 no teste — as features atuais dão sinal limitado para separar a classe 1. Isso motiva a comparação com a Rede Neural (seção 13) e a discussão de limitações na seção 14.
- **Importância das features.** `EXT_SOURCE_MEAN` (0.3382) domina, seguido de `CREDIT_ANNUITY_RATIO`, `AGE_YEARS`, `EMPLOYMENT_YEARS`, `EXT_SOURCE_3` — consistente com o conhecimento de domínio do Home Credit (scores externos e variáveis de crédito são os principais preditores).

### 3.3 Melhorias de desempenho avaliadas (sem aumentar o tempo de treino)

Problema inicial: F1 = 0.227 no teste, com precisão baixa (0.139) — a árvore detectava a classe 1 mas com muitos falsos positivos. Avaliamos alavancas **sem custo computacional adicional** (algumas reduzem o custo):

1. **`class_weight` customizado** (`{0:1, 1:4}`, `{1:8}`, `{1:12}`): probe sintético mostrou que `'balanced'` já é ótimo; customizações não melhoraram o F1. **Rejeitado.**
2. **`max_features`** (`'sqrt'`, 10, 15): reduz o custo de treino, mas o F1 caiu levemente (a aleatoriedade tipo RF precisa de muitas árvores para compensar). **Rejeitado** para árvore única.
3. **`min_samples_leaf=30` (pré-poda):** reduz a árvore (profundidade 90→30, folhas 20.555→3.909), **reduz o custo de treino** e estabiliza as folhas. **Adotado.**
4. **Threshold de decisão otimizado por F1 na validação:** custo zero (só reclassifica `predict_proba`); com threshold 0.70 o F1 no teste subiu e a precisão melhorou muito. **Adotado.**

**Resultado da otimização (F1 no teste): 0.227 → 0.274 (+21%)**, precisão 0.139 → 0.204 (+47%), acurácia 0.66 → 0.82, com overfitting mínimo (E_in−E_out = −0.011) e treino **mais rápido** (menos folhas). O recall caiu de 0.62 → 0.42 (trade-off do threshold), mas F1 e precisão são o objetivo no desbalanceamento.

### 3.4 Registro da iteração anterior (antes das alavancas de desempenho)

Para fins de registro do processo, segue o estado **anterior** à otimização descrita em 3.3 (primeira versão validada da seção 12, com `class_weight='balanced'` e poda, **sem** `min_samples_leaf=30` e **sem** threshold otimizado):

| | sem poda | podada |
|---|---|---|
| profundidade | 90 | 32 |
| folhas | 20.555 | 2.154 |
| E_in | 0.0000 | 0.2887 |
| E_out (teste) | 0.1432 | 0.3386 |
| E_in − E_out | −0.1432 | −0.0499 |
| recall (teste) | 0.1665 | 0.6165 |
| F1 (teste) | 0.1581 | 0.2272 |

Características dessa iteração: árvore sem poda **memorizava** (E_in = 0, 20.555 folhas, profundidade 90); a poda por `ccp_alpha` (5.03e-05) reduzia o overfitting (E_in−E_out de −0.143 para −0.050) e elevava o recall no teste para 0.62, mas a **precisão era baixa (0.139)** e o F1 ficava em 0.227. Essa versão foi **substituída** pelas alavancas de 3.3 (que elevaram o F1 para 0.274 com treino mais rápido e menos complexidade), mantida aqui apenas como histórico do processo.

### 3.5 Integração da seção 11 (do colega) — 12/08/2026

- O colega commitou e pushou a seção 11 no origin/main (9f2584) e criou a branch integrar-secao-11 (idêntica ao main).
- Integração via git pull --rebase sobre o main local (seção 12 commitada como 52529f). Rebase **sem conflito** (as células da seção 11 foram adicionadas em posições distintas das da seção 12).
- Notebook mesclado: **90 células**, seções 11 e 12 presentes, sem IDs duplicados, JSON válido.
- **Re-execução de ponta a ponta validada** (nbclient, MPLBACKEND=Agg, TF 2.21/CPU): todas as células OK, sem erros, outputs persistidos. Execução ~20 min.
- Ambiente: foi necessário instalar 	ensorflow==2.21.0 (ausente no Python 3.13 local) e 
bclient/
bformat para persistir outputs.

**Comparação rápida RN × Árvore (teste interno) para a seção 13 (execução com as correções da auditoria):**
| métrica | RN (thr 0.50) | Árvore (thr 0.75) |
|---|---|---|
| F1 | 0.268 | 0.260 |
| recall | 0.641 | 0.398 |
| precisão | 0.169 | 0.193 |
| acurácia | 0.717 | 0.817 |
| E_out − E_in | 0.0129 | 0.0292 |
| AUC | 0.7495 | 0.6658 |

A RN vence em AUC, F1, recall e log-loss; a árvore vence em precisão/acurácia e interpretabilidade. Thresholds diferentes (0.50 RN vs 0.75 árvore) — a árvore também é avaliada com thr 0.50 na seção 13 para comparação justa.

## 4. Seção 13 — Comparação dos Modelos — IMPLEMENTADO (12/08/2026, sem submissão)

1. Tabela comparativa no **teste interno**: acurácia, precisão, recall, F1, E_in, E_out, E_in − E_out, complexidade (parâmetros / profundidade e folhas). Matrizes de confusão lado a lado. **Feito.**
2. Seleção do modelo final por desempenho no teste + generalização + interpretabilidade.
3. **Submissão — DESCARTADA por decisão do projeto (12/08/2026).** A comparação acadêmica é feita no teste interno; o `application_test.csv` (sem rótulos) não será usado para predições de submissão. O `data/processed/` permanece no `.gitignore`.

**Resultados validados (execução 12/08/2026 com as correções da auditoria, teste interno 46.127):**
| métrica | RN (thr 0.50) | Árvore (thr 0.75) | Árvore (thr 0.50) |
|---|---|---|---|
| F1 | **0.268** | 0.260 | 0.218 |
| recall | **0.641** | 0.398 | 0.591 |
| precisão | 0.169 | **0.193** | 0.134 |
| acurácia | 0.717 | **0.817** | 0.658 |
| E_out − E_in (métrica própria) | **0.0129** | 0.0292 | 0.1889 |
| log-loss teste (métrica comum) | **0.5502** | 1.0163 | 1.0163 |
| AUC | **0.7495** | 0.6658 | 0.6658 |
| complexidade | 21.385 parâmetros | 30 prof / 2.322 folhas | — |

**Análise:** a RN vence em AUC (0.75 vs 0.67), F1, recall e log-loss de teste — inclusive na métrica comum (log-loss 0.5502 vs 1.0163, probabilidades da árvore menos calibradas). A árvore vence em precisão/acurácia no threshold otimizado e em interpretabilidade. **Melhor modelo: a Rede Neural** (critério AUC + F1 + recall + log-loss); a Árvore permanece como alternativa interpretável.

Figuras: `reports/figures/comparison_confusion_matrices.png` e `comparison_roc_curves.png`.

## 5. Seção 14 — Conclusões — IMPLEMENTADO (12/08/2026)

- Responder à pergunta do problema; comparar modelos (desempenho, generalização, E_in vs E_out, overfitting, complexidade, interpretabilidade); limitações (desbalanceamento, Etapa 2 pendente).

**Conclusões consolidadas (notebook, seção 14 — com as correções da auditoria):**
- **Resposta à pergunta:** sim, é possível identificar casos de dificuldade de pagamento com as variáveis atuais — RN (AUC 0.7495) e Árvore (AUC 0.6658) superam o aleatório; recall de 0.6412 (RN) e 0.3977 (Árvore) contra prevalência de ~8%.
- **Melhor modelo (decisão explícita): a Rede Neural** — vence em AUC, F1 (0.2678 vs 0.2601), recall e log-loss de teste (0.5502 vs 1.0163, métrica comum). A Árvore (thr 0.75) permanece como alternativa interpretável, com melhor precisão (0.1933) e acurácia (0.8174).
- **Achados:** `EXT_SOURCE_MEAN` é o preditor dominante (importância 0.3382); desbalanceamento é o maior obstáculo; o threshold de decisão impacta muito o ponto de operação.
- **Limitações:** desbalanceamento severo (~8%); apenas a solicitação atual (sem `bureau`/`previous_application`); sem submissão ao teste externo (decisão do projeto); métricas de erro heterogêneas (log-loss vs erro de classificação — mitigado com o log-loss comum da Árvore); vazamento leve da EDA pré-split (documentado).
- **Trabalhos futuros:** Etapa 2 (dados auxiliares), ensembles, calibração, regularização explícita da RN (L2/dropout), busca de hiperparâmetros com GPU, análise de custo-benefício do threshold.

---

## 6. Ações pendentes fora do notebook

- [ ] README: seções "Separação dos dados" e "Status" — descrevem split interno com teste isolado (agora correto) mas precisam refletir: teste interno 70/15/15, `application_test.csv` só para submissão, e marcar "construção de X e y", "separação" e "pré-processamento" como concluídos.
- [x] `requirements.txt`: adicionar `scipy` e fixar `scikit-learn>=1.2` (API `keep_empty_features`/`sparse_output`) — **feito**.
- [x] `np.random.seed(RANDOM_STATE)` na célula 5 e imports de sklearn/scipy consolidados no topo — **feito na revisão final de 11/08/2026**.
- [x] `.gitignore`: `.opencode/` e `data/processed/` ignorados — **feito**.
- [ ] Combinar com o colega: seeds da RN (`tf.random.set_seed`, `random.seed`), formato da submissão, e proibição de editar seções 1–10 em paralelo.
- [ ] Rodar o notebook por inteiro no VS Code e salvar os outputs (células 7–36 têm outputs; 41–64 ainda vazios) — evidência de auditoria do contrato.

## 7. Checklist de verificação

- [x] Executar o notebook do início ao fim (células 5–64) sem erros, sem warnings nas seções 7–10, com backend headless (Agg) — **feito na revisão final de 11/08/2026** (X 307.511×102; split 215.257/46.127/46.127; filtro 7+1; p = 214; 0 NaN/inf).
- [x] Executar o notebook do início ao fim (nbclient, outputs persistidos no arquivo versionado) — **feito em 12/08/2026** (duas execuções completas: 19:08 e 19:42, zero erros).
- [x] 11: seeds fixos; N = 215.257 e p = 214 usados na justificativa; métricas no teste — **implementado e validado em 12/08/2026** (F1 0.268, recall 0.641, AUC 0.75).
- [x] 12: árvore sem poda → poda por `ccp_alpha` **com validação cruzada (5 folds, fold = N/5)**; árvores em `reports/figures/` — **implementado e validado em 12/08/2026 (correções da auditoria)**.
- [x] 13: comparação no teste interno — **implementado e validado em 12/08/2026**; submissão descartada por decisão do projeto.
- [x] 14: conclusões alinhadas ao README — **implementado em 12/08/2026** (resposta à pergunta, comparação, achados, limitações, trabalhos futuros).
