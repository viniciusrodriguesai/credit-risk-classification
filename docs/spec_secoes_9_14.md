# Spec — Seções 9 a 14 + Submissão (notebook `credit_risk_classification.ipynb`)

> Documento de planejamento. Nenhuma célula das seções 9–14 existe ainda no notebook (apenas placeholders de título). Este spec define **o que** cada seção deve fazer e **como validar**; a implementação será feita em sessões posteriores sob nossa responsabilidade.

## 0. Estado de referência (verificado em 07/08/2026)

Contratos já estabelecidos e validados empiricamente (executar antes de iniciar a seção 9):

| Objeto | Valor esperado |
|---|---|
| `application_model` | 307.511 × 95 |
| `X` | 307.511 × 91 (37 int64 / 37 float64 / 15 str / 2 int8) |
| `y` | 307.511, sem NaN, valores ⊆ {0, 1} |
| `ids` | `SK_ID_CURR`, 307.511 valores únicos |
| NaN em X | 36 colunas, 2.710.773 valores no total |
| Categóricas em X | 15 (136 categorias ao todo; `ORGANIZATION_TYPE` com 58) |
| `inf` em X | 0 |
| Teste externo | `data/raw/application_test.csv`, 48.744 × 121, **sem `TARGET`** |
| Datasets auxiliares | `bureau.csv` (170 MB) e `previous_application.csv` (405 MB) — fora do escopo (Etapa 2 do README) |

**Nota metodológica:** o teste externo não tem rótulos. Toda avaliação de desempenho acontece na **validação**. O teste externo é usado **uma única vez**, no final, para gerar predições (submissão).

---

## 1. Seção 9 — Divisão em Treinamento e Validação (85/15)

### Objetivo
Separar `X`, `y` e `ids` em treinamento (85%) e validação (15%), estratificados por `TARGET`, com `RANDOM_STATE = 42`.

### Especificação
1. Cabeçalho md: explicar que **não há split de teste** — o teste é o `application_test.csv` externo (sem rótulos), usado somente ao final; documentar que a validação servirá para tuning e para a comparação final.
2. Código:
   ```python
   X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
       X, y, ids,
       test_size=0.15,
       stratify=y,
       random_state=RANDOM_STATE,
   )
   ```
3. **Asserts/checagens** (todas com valores esperados):
   - `len(X_train) == 261384`, `len(X_val) == 46127`, soma = 307.511;
   - proporção de `TARGET = 1` ≈ 8,07% nas duas partições (tolerância ~0,1 pp);
   - interseção vazia entre `ids_train` e `ids_val`;
   - `X_train.shape[1] == X_val.shape[1] == 91`.
4. Registrar tamanhos/proporções em variáveis reutilizáveis (ex.: `N_TRAIN`, `N_VAL`, `POS_RATE_TRAIN`).
5. **Célula de robustez (mitiga achado A1):** recomputar **somente no treino**:
   - missing rate ≥ 60% → conjunto de colunas deve ser idêntico ao removido na seção 7.3;
   - grupos `_AVG/_MODE/_MEDI` → seleção idêntica à da seção 7.2;
   - colunas constantes → nenhuma.
   Usar `assert` de igualdade de conjuntos. Isso evidencia que as regras definidas com a base completa não mudariam se calculadas só no treino.
6. Declaração de isolamento (md): a validação não participa de seleção de atributos nem de ajuste dos transformadores (fit só no treino); o teste externo é intocável até a submissão.

### Referências de valores
- Esperado: treino 261.384 (8,073% positivos), validação 46.127 (8,073% positivos).

---

## 2. Seção 10 — Pré-processamento dos Dados

### Objetivo
Transformar `X_train`/`X_val` (e, futuramente, o teste externo) em matrizes prontas para cada modelo, com **fit apenas no treino**.

### Regras gerais
- Dois pipelines distintos (README, "Pré-processamento"):
  - **RN**: imputação (numéricas: mediana; categóricas: `most_frequent`) → OneHotEncoder (`handle_unknown='ignore'`) → `StandardScaler`.
  - **Árvore**: imputação + codificação **sem escalonamento**. Para alta cardinalidade (`ORGANIZATION_TYPE` 58 cat., `OCCUPATION_TYPE` 18 cat.), avaliar agrupamento de categorias raras (ex.: min_frequency) ou manter OHE; árvore aceita ambos, mas agrupamento reduz ruído.
- Flags `int8` (`DAYS_EMPLOYED_ANOMALY`, `EXT_SOURCE_1_MISSING`) são numéricas completas — passam sem imputação.
- `EMPLOYED_YEARS` tem 18% de NaN (55.374) rotulados pelo flag: avaliar imputação pela mediana **condicionada ao flag** em vez da mediana global (distorção do sinal).
- Outliers finitos conhecidos (decisão documentada, estatística do treino): `CREDIT_INCOME_RATIO` máx. 84,7; `CREDIT_PER_FAMILY` máx. ~4 M; `ANNUITY_INCOME_RATIO` máx. 1,88. Para a RN, avaliar clipping (percentis 1–99 do treino); a árvore é robusta.
- Guardar os transformadores fitados em variáveis para reutilizar na submissão (RN) e registrar `X_train_processed.shape` como contrato para a seção 11.

### Verificações
- `transform` na validação não vaza estatísticas: conferir que `scaler.mean_`/medianas vêm do treino.
- Relatório pós-processamento: shape, ausência de NaN, dtypes numéricos.

---

## 3. Seção 11 — Rede Neural (responsabilidade do colega)

### Contratos exigidos
- **Reprodutibilidade:** `tf.random.set_seed(RANDOM_STATE)`, `np.random.seed(RANDOM_STATE)`, `random.seed(RANDOM_STATE)` antes de construir o modelo — **combinar com o colega agora**, não na seção 11.
- Usar `RANDOM_STATE` como seed global do notebook.
- Arquitetura conforme README: entrada (dimensão = `X_train_processed.shape[1]`), 1+ camadas ocultas ReLU, 1 neurônio de saída sigmoide, `binary_crossentropy`, Adam.
- Justificar: dimensão VC × Regra de Ouro (parâmetros treináveis), número de épocas, batch size, critério de parada (early stopping na perda de validação).
- Acompanhar perdas/acurácias de treino e validação; identificar overfitting (E_in ↓ enquanto E_val ↑).
- Calcular E_in e E_out na validação; métricas: acurácia, precisão, recall, F1, matriz de confusão (atenção ao recall da classe positiva).
- Custo de memória: o DataFrame bruto já foi liberado na seção 7.6.

---

## 4. Seção 12 — Árvore de Decisão e Poda por Complexidade de Custo

### Especificação
1. Árvore **sem poda** (`max_depth=None`) treinada em `X_train` (matriz codificada da seção 10):
   - registrar profundidade, nº de folhas, E_in, E_val, E_in − E_val (overfitting esperado);
2. **Minimal Cost-Complexity Pruning** (README):
   - `ccp_alpha` candidatos via `cost_complexity_pruning_path` no treino;
   - avaliar candidatos por validação cruzada (treino/validação disponíveis; CV interna em `X_train` se necessário);
   - selecionar melhor `ccp_alpha` **pelo desempenho de validação**; preferir árvore mais simples em empate;
   - treinar a árvore podada, calcular métricas na validação;
   - visualizar a árvore final (salvar em `reports/figures/`).
3. Importância de features (`feature_importances_`) como insumo da discussão.
4. Nada de tuning na validação além do `ccp_alpha` — o teste externo não participa.

---

## 5. Seção 13 — Comparação dos Modelos

### Especificação
- Tabela comparativa na **validação** (mesmas amostras, mesmas métricas): acurácia, precisão, recall, F1, E_in, E_out, E_in − E_out, complexidade (nº de parâmetros / profundidade e folhas).
- Matrizes de confusão lado a lado.
- Nota explícita: **o teste externo não tem rótulos; a comparação final é feita na validação**, e o teste externo serve apenas para predição/submissão.
- Selecionar o modelo final com base no desempenho de validação + generalização + interpretabilidade.

### Submissão (nova subseção 13.x — fechar a entrega do Kaggle)
1. Carregar `data/raw/application_test.csv` (48.744 × 121; validar `TARGET not in columns` e mesmo esquema do treino).
2. Aplicar a **mesma** limpeza da seção 7 como transform (nunca fit):
   - `DAYS_EMPLOYED == 365243` → NaN + flag `DAYS_EMPLOYED_ANOMALY`;
   - drop das 28 variantes + 7 de alta ausência + constantes;
   - criação das 8 features (incl. `DAYS_BIRTH`/`DAYS_EMPLOYED` → anos e remoção das originais, conforme seção 8);
   - **validação:** shape (48.744, 91) e colunas idênticas a `X` (assert de igualdade de `columns`).
3. Pré-processar com os transformadores **fitados no treino** (seção 10); assert de colunas vs. `X_train_processed`.
4. Prever com ambos os modelos → probabilidade `TARGET` (classe positiva):
   - salvar `data/processed/submission_arvore.csv` e `data/processed/submission_rn.csv` com colunas `SK_ID_CURR`, `TARGET`;
   - padronizar o nome da coluna de predição com o colega (formato Kaggle: `SK_ID_CURR`, `TARGET`).
5. Registrar também predições na validação para análise de calibração, se desejado.

---

## 6. Seção 14 — Conclusões

- Responder à pergunta do problema (README): as informações permitem identificar casos de dificuldade de pagamento?
- Comparar os modelos sob: desempenho, generalização, E_in vs E_out, overfitting, complexidade, interpretabilidade.
- Limitações (desbalanceamento, ausência de rótulos no teste externo, Etapa 2 pendente com bureau/previous_application).

---

## 7. Ações pendentes fora do notebook

- [ ] README: seção "Separação dos dados" e "Como executar" ainda descrevem split triplo com teste interno — atualizar para refletir `application_test.csv` externo (sem rótulos) e avaliação na validação.
- [ ] Combinar com o colega da RN: seeds (`tf.random.set_seed`), formato da predição de submissão e assinatura de `X_train_processed`.
- [ ] Decidir se `data/processed/` entra no `.gitignore` (submissões são geradas, não versionadas).

## 8. Checklist de verificação por seção

- [ ] 9: asserts de tamanho/proporção/overlap + célula de robustez (regras recomputadas no treino) passam.
- [ ] 10: transformadores com fit só no treino; shapes registrados; sem NaN pós-processamento.
- [ ] 11: seeds fixos; justificativas de arquitetura/épocas/batch; E_in/E_out na validação.
- [ ] 12: árvore sem poda → poda por `ccp_alpha` com CV; árvore final salva em `reports/figures/`.
- [ ] 13: comparação na validação; submissão com asserts de esquema idêntico ao treino.
- [ ] 14: conclusões alinhadas ao README.
