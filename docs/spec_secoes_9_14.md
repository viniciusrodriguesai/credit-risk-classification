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

## 2. Seção 11 — Rede Neural (colega, base alinhada)

- **Reprodutibilidade:** `tf.random.set_seed(RANDOM_STATE)`, `np.random.seed(RANDOM_STATE)`, `random.seed(RANDOM_STATE)` antes de construir o modelo.
- Arquitetura conforme README: entrada (dimensão **p = 214**), 1+ camadas ocultas ReLU, 1 neurônio de saída sigmoide, `binary_crossentropy`, Adam.
- Justificar com dimensão VC e Regra de Ouro (N = 215.257, p = 214), número de épocas, batch size, critério de parada (early stopping na perda de validação).
- Acompanhar perdas/acurácias de treino e validação; identificar overfitting; calcular E_in e E_out no teste.
- Métricas no teste interno: acurácia, precisão, recall (atenção à classe positiva), F1, matriz de confusão.
- Matrizes de entrada: `X_train_nn`, `X_val_nn`, `X_test_nn` (CSR float32).

## 3. Seção 12 — Árvore de Decisão (nós)

1. Árvore **sem poda** em `X_train_tree` (CSR float32): profundidade, nº de folhas, E_in, E_out (teste), E_in − E_out.
2. **Minimal Cost-Complexity Pruning**: `cost_complexity_pruning_path` no treino → candidatos de `ccp_alpha` → seleção por validação (CV interna no treino, se necessário) → preferir árvore mais simples em empate → treinar podada → métricas no teste.
3. Importância de features; salvar a árvore final em `reports/figures/`.

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
- [ ] 12: árvore sem poda → poda por `ccp_alpha`; árvore final em `reports/figures/`.
- [ ] 13: comparação no teste interno; submissão com asserts de esquema idêntico.
- [ ] 14: conclusões alinhadas ao README.
