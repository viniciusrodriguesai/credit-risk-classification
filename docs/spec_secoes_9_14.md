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

**Decisão metodológica:** como a classe positiva é minoritária (~8%), as árvores usam `class_weight='balanced'`. O `ccp_alpha` é escolhido na validação com re-treino dos candidatos numa amostra estratificada (30k), com o caminho de poda calculado no treino completo; o teste é reservado para a avaliação final. **Otimização adicional (12/08/2026):** pré-poda `min_samples_leaf=30` (estabiliza folhas e reduz custo de treino) e **threshold de decisão otimizado por F1 na validação** (custo zero, apenas reclassifica probabilidades). O E_in também usa o threshold selecionado, para a comparação E_in/E_out ser consistente.

**Resultados validados (execução 12/08/2026, com as 3 alavancas):**
| | sem poda (msl=30) | podada + thr 0.70 |
|---|---|---|
| profundidade | 30 | 25 |
| folhas | 3.909 | 773 |
| E_in | 0.2484 | 0.1680 |
| E_out (teste) | 0.3100 | 0.1788 |
| E_in − E_out | −0.0616 | −0.0108 |
| recall (teste) | 0.5255 | 0.4181 |
| precisão (teste) | 0.1351 | 0.2038 |
| F1 (teste) | 0.2149 | **0.2741** |

O `ccp_alpha` selecionado (6.61e-05) + threshold 0.70 reduzem a complexidade (3.909 → 773 folhas), o overfitting (E_in−E_out = −0.011) e melhoram o F1 no teste de 0.215 → **0.274** (+28%) com precisão de 0.135 → **0.204** (+51%) e acurácia 0.69 → 0.82. O recall caiu (0.53 → 0.42) — trade-off esperado do threshold alto. Figura em `reports/figures/decision_tree_pruned.png` (max_depth=3).

### 3.1 Dificuldades encontradas (documentação do processo)

1. **Custo de re-treino dos candidatos de poda.** Avaliar o `ccp_alpha` re-treinando cada candidato no treino completo (215.257×214) é proibitivo (≈67 min para ~20 candidatos). **Solução:** o caminho de poda (`cost_complexity_pruning_path`) é calculado no treino completo (barato, uma passada), mas os candidatos são **re-treinados numa amostra estratificada de 30.000 linhas** e avaliados na validação completa (46.127). Como a seleção de hiperparâmetro não precisa da base completa, o custo caiu para ~6 min.
2. **Poda destruía a classe positiva.** Sem `class_weight`, o `ccp_alpha` tende a podar em direção à classe majoritária: a árvore "podada" (27 folhas) tinha recall **0.0075** no teste (praticamente nunca previa a classe 1) — um resultado metodologicamente pobre. **Solução:** `class_weight='balanced'` em todas as árvores (sem poda, candidatos e final), alinhado ao desbalanceamento (~8% de positivos).
3. **Caminho de poda na amostra escolhia `ccp_alpha=0`.** Inicialmente o caminho foi calculado numa amostra de 30k; como a árvore da amostra é muito menor que a do treino completo, o "melhor" F1 de validação correspondia a **não podar** (alpha=0), anulando a etapa de poda. **Solução:** calcular o caminho na **árvore sem poda já ajustada no treino completo** (6467 alphas distintos), usando a amostra apenas para re-treinar candidatos.
4. **Tempo de execução da seção.** A árvore sem poda (90 de profundidade, 20.555 folhas) e a final levam ~5 min cada no treino completo. Mitigado rodando em background (Agg headless) e limitando candidatos a 7.
5. **Limpeza de outputs em execução headless.** O executor de validação não persiste outputs no `.ipynb` (apenas valida a execução); os outputs reais são gerados no Run All do VS Code. Isso é esperado no fluxo do projeto.

### 3.2 Conclusões da seção 12 (a consolidar na seção 14)

- **Pré-poda controla a memorização.** A árvore sem pré-poda atinge E_in = 0 com 20.555 folhas (overfitting). Com `min_samples_leaf=30` a árvore fica com 3.909 folhas e E_in − E_out = −0.062 no teste — a memorização é reduzida já na base, e o `ccp_alpha` (6.61e-05) enxuga ainda mais (773 folhas, E_in − E_out = **−0.011**).
- **Threshold de decisão é uma alavanca barata e efetiva.** Otimizar o threshold por F1 na validação (0.70) elevou o F1 no teste de 0.215 → **0.274** e a precisão de 0.135 → **0.204**, com acurácia 0.69 → 0.82. O recall caiu (0.53 → 0.42), mas o F1 — a métrica combinada — melhorou, que é o objetivo no desbalanceamento.
- **Trade-off acurácia × recall.** O threshold alto troca recall por precisão: o modelo classifica como "dificuldade" apenas casos com alta confiança (prob ≥ 0.70), gerando menos falsos positivos. No problema (custo de um falso positivo é alto), isso é desejável.
- **A classe minoritária é intrinsecamente difícil.** Mesmo balanceada e otimizada, a árvore tem F1 ~0.27 no teste — as features atuais dão sinal limitado para separar a classe 1. Isso motiva a comparação com a Rede Neural (seção 13) e a discussão de limitações na seção 14.
- **Importância das features.** `EXT_SOURCE_MEAN` (0.50) domina, seguido de `CREDIT_ANNUITY_RATIO`, `AGE_YEARS`, `LTV`, `EXT_SOURCE_3`, `EMPLOYMENT_YEARS` — consistente com o conhecimento de domínio do Home Credit (scores externos e variáveis de crédito são os principais preditores).

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

**Comparação rápida RN × Árvore (teste interno) para a seção 13:**
| métrica | RN (thr 0.50) | Árvore (thr 0.70) |
|---|---|---|
| F1 | 0.268 | 0.274 |
| recall | 0.641 | 0.418 |
| precisão | 0.169 | 0.204 |
| acurácia | 0.717 | 0.821 |
| E_in − E_out | 0.0129 | −0.0108 |

Perfis diferentes: RN tem mais recall, árvore mais precisão; F1 muito próximos. Thresholds diferentes (0.50 RN vs 0.70 árvore) — considerar normalizar para comparação justa na seção 13.

## 4. Seção 13 — Comparação dos Modelos + Submissão

1. Tabela comparativa no **teste interno**: acurácia, precisão, recall, F1, E_in, E_out, E_in − E_out, complexidade (parâmetros / profundidade e folhas). Matrizes de confusão lado a lado.
2. Seleção do modelo final por desempenho no teste + generalização + interpretabilidade.
3. **Submissão** (nova subseção 13.x):
   - Carregar `data/raw/application_test.csv` (48.744 × 121; validar ausência de `TARGET` e esquema do treino).
   - Aplicar a **mesma** limpeza da seção 7 como transform (anomalia 365243 → NaN + flag; XNA → NaN; drop das 28 variantes; 12 features com `safe_ratio`; drop de `DAYS_BIRTH`/`DAYS_EMPLOYED`) → shape (48.744, 102); `assert` de igualdade de colunas com `X`.
   - Aplicar os filtros e transformadores **fitados no treino** (10.1 + pipelines); `assert` de colunas vs. matriz de treino.
   - Prever com ambos os modelos → `data/processed/submission_arvore.csv` e `submission_rn.csv` com `SK_ID_CURR`, `TARGET` (probabilidade da classe positiva; formato Kaggle combinado com o colega).

## 5. Seção 14 — Conclusões

- Responder à pergunta do problema; comparar modelos (desempenho, generalização, E_in vs E_out, overfitting, complexidade, interpretabilidade); limitações (desbalanceamento, Etapa 2 pendente).

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
- [ ] Rodar no VS Code (Run All) e salvar os outputs — evidência no arquivo versionado.
- [ ] 11: seeds fixos; N = 215.257 e p = 214 usados na justificativa; métricas no teste.
- [x] 12: árvore sem poda → poda por `ccp_alpha`; árvore final em `reports/figures/` — **implementado e validado em 12/08/2026**.
- [ ] 13: comparação no teste interno; submissão com asserts de esquema idêntico.
- [ ] 14: conclusões alinhadas ao README.
