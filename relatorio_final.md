# Classificação de Risco de Crédito com Aprendizagem de Máquina

**Disciplina:** Aprendizagem de Máquina  
**Semestre:** 2026.1  
**Professores:** Bruno Jefferson de Sousa Pessoa e Gilberto Farias de Sousa Filho  
**Autores:** Gustavo Pereira e Vinicius Mangueira  

---

## Resumo

Este trabalho aplica técnicas de Aprendizagem de Máquina ao problema de classificação de risco de crédito utilizando o conjunto de dados **Home Credit Default Risk**. O objetivo é identificar solicitações associadas à dificuldade de pagamento e comparar dois modelos supervisionados: uma **Rede Neural Artificial** e uma **Árvore de Decisão regularizada por Minimal Cost-Complexity Pruning**.

O projeto inclui análise exploratória dos dados, tratamento de valores ausentes, correção de valores anômalos, redução de redundância entre atributos, engenharia de atributos, separação estratificada em treinamento, validação e teste, pré-processamento específico para cada modelo, análise de generalização, regularização e avaliação por acurácia, precisão, recall, F1-score, matriz de confusão e AUC.

Após o pré-processamento, o conjunto de treinamento contém **215.257 exemplos** e **214 atributos**. A Rede Neural possui uma camada escondida com **99 neurônios**, totalizando **21.385 parâmetros treináveis**, atendendo à Regra de Ouro adotada no projeto. A Árvore de Decisão foi regularizada por validação cruzada estratificada em cinco folds, resultando em `ccp_alpha = 3.6207798e-05`.

Na avaliação final, a Rede Neural apresentou **AUC = 0,7495**, **recall = 0,6412** e **F1 = 0,2678**, enquanto a Árvore apresentou **AUC = 0,6658**, **recall = 0,3977** e **F1 = 0,2601** no threshold otimizado de 0,75. Considerando capacidade de discriminação, F1, recall e log-loss, a **Rede Neural foi selecionada como melhor modelo**, enquanto a Árvore permanece como alternativa mais interpretável.

---

## 1. Introdução

A análise de risco de crédito é um problema relevante para instituições financeiras, pois envolve a identificação de solicitações com maior possibilidade de dificuldade de pagamento. Neste projeto, o problema foi formulado como uma tarefa de **classificação supervisionada binária**.

A variável-alvo é a coluna `TARGET`:

- `TARGET = 0`: solicitação não associada à classe de dificuldade de pagamento;
- `TARGET = 1`: solicitação associada à classe de dificuldade de pagamento.

A classe positiva é, portanto, `TARGET = 1`.

O objetivo principal é responder à seguinte pergunta:

> É possível utilizar informações do solicitante e da solicitação de crédito para identificar casos associados à dificuldade de pagamento?

Foram comparados dois modelos:

1. Rede Neural Artificial;
2. Árvore de Decisão regularizada por Minimal Cost-Complexity Pruning.

O trabalho foi desenvolvido em Python, utilizando principalmente Pandas, NumPy, Scikit-Learn, Matplotlib, TensorFlow e Keras.

---

## 2. Conjunto de dados

O conjunto utilizado é o **Home Credit Default Risk**, originalmente disponibilizado em uma competição do Kaggle.

A primeira versão do projeto utiliza o arquivo:

```text
application_train.csv
```

A base possui inicialmente:

- **307.511 registros**;
- **122 colunas**;
- variáveis numéricas e categóricas;
- informações financeiras, demográficas, familiares, de emprego, moradia e pontuações externas de risco.

O identificador `SK_ID_CURR` é preservado para rastreabilidade, mas não é utilizado como atributo preditivo. A coluna `TARGET` é removida da matriz de entrada e utilizada como vetor-alvo.

---

## 3. Análise exploratória dos dados

### 3.1 Distribuição da variável-alvo

A base é fortemente desbalanceada:

- `TARGET = 0`: aproximadamente **91,93%**;
- `TARGET = 1`: aproximadamente **8,07%**.

Esse desbalanceamento torna inadequado avaliar os modelos somente pela acurácia. Por isso, foram utilizadas também precisão, recall e F1-score, com atenção especial ao recall da classe positiva.

![Distribuição da variável-alvo](figures/target_distribution.png)

### 3.2 Valores ausentes

A análise exploratória mostrou que algumas colunas possuem percentuais de ausência próximos de 70%. O tratamento definitivo dessas colunas é realizado posteriormente, utilizando estatísticas aprendidas a partir do conjunto de treinamento.

![Colunas com maior percentual de valores ausentes](figures/top_missing_values.png)

### 3.3 Correlação com TARGET

As variáveis individuais mais correlacionadas com a variável-alvo foram as pontuações externas:

- `EXT_SOURCE_3`;
- `EXT_SOURCE_2`;
- `EXT_SOURCE_1`.

As correlações são negativas, indicando que menores pontuações externas estão associadas a maior ocorrência de `TARGET = 1`.

![Correlação com TARGET](figures/correlation_with_target.png)

Também foi analisada a correlação entre as variáveis mais relevantes para identificar redundâncias.

![Correlação entre as principais variáveis](figures/correlation_top_features.png)

### 3.4 Comparação entre as classes

Foram utilizados boxplots para comparar variáveis financeiras e pontuações externas entre `TARGET = 0` e `TARGET = 1`. As diferenças mais evidentes aparecem nos atributos `EXT_SOURCE`, indicando que essas pontuações possuem sinal relevante para separação das classes.

![Comparação das distribuições por classe](figures/target_comparison_boxplots.png)

---

## 4. Limpeza e engenharia de atributos

### 4.1 Tratamento de valores especiais

A variável `DAYS_EMPLOYED` contém o valor especial `365243`, que não representa um tempo de emprego plausível. Esse valor foi convertido em ausente e foi criada a variável indicadora:

```text
DAYS_EMPLOYED_ANOMALY
```

Também foram tratados valores especiais como `XNA` em `CODE_GENDER`.

### 4.2 Redução de redundância

Foram identificadas variáveis altamente correlacionadas, especialmente grupos com sufixos:

```text
_AVG
_MODE
_MEDI
```

Para reduzir redundância, uma representação foi mantida por grupo, com preferência pelas variáveis `_MEDI`.

### 4.3 Novos atributos

Foram criados atributos com interpretação financeira e demográfica, incluindo:

- `AGE_YEARS`;
- `EMPLOYMENT_YEARS`;
- `CREDIT_INCOME_RATIO`;
- `ANNUITY_INCOME_RATIO`;
- `CREDIT_ANNUITY_RATIO`;
- `CREDIT_PER_FAMILY_MEMBER`;
- `CHILDREN_PER_FAMILY_MEMBER`;
- `LTV`;
- `DOWN_PAYMENT`;
- `EXT_SOURCE_MEAN`;
- indicadores de ausência de pontuações externas.

As divisões foram realizadas com uma função segura para evitar divisão por zero e geração de valores infinitos.

Após essa etapa:

```text
X = 307.511 x 102
y = 307.511
```

---

## 5. Separação dos dados

Os dados rotulados foram divididos de forma estratificada em:

| Conjunto | Quantidade | Proporção |
|---|---:|---:|
| Treinamento | 215.257 | 70% |
| Validação | 46.127 | 15% |
| Teste | 46.127 | 15% |

A estratificação preserva aproximadamente a proporção de 8,07% da classe positiva em todos os conjuntos.

A separação em três partes permite utilizar:

- **treinamento** para ajuste dos parâmetros;
- **validação** para decisões de arquitetura e hiperparâmetros;
- **teste** para avaliação final.

As decisões de arquitetura, número de épocas, `batch_size`, `ccp_alpha`, threshold e demais hiperparâmetros foram baseadas em treinamento e validação.

---

## 6. Pré-processamento

As decisões de pré-processamento foram aprendidas a partir do conjunto de treinamento.

Foram removidas:

- 7 colunas com pelo menos 60% de valores ausentes;
- 1 coluna constante.

Após os filtros:

- 94 variáveis permaneceram antes do One-Hot Encoding;
- 79 eram numéricas;
- 15 eram categóricas;
- o One-Hot Encoding resultou em **214 atributos finais**.

### 6.1 Variáveis numéricas

Para a Rede Neural:

1. imputação pela mediana;
2. padronização por `StandardScaler`.

A padronização utilizada é:

\[
z = \frac{x-\mu}{\sigma}
\]

A escolha do z-score se deve à necessidade de colocar os atributos numéricos em escalas comparáveis para a otimização por gradiente.

### 6.2 Variáveis categóricas

Foram utilizadas:

1. imputação pela categoria mais frequente;
2. `OneHotEncoder(handle_unknown='ignore')`.

### 6.3 Pipeline da Árvore

A Árvore de Decisão não utiliza `StandardScaler`, pois seus cortes não dependem da escala numérica da mesma forma que os métodos baseados em gradiente.

Ao final:

\[
N = 215.257
\]

\[
p = 214
\]

---

## 7. Rede Neural

### 7.1 Arquitetura

A arquitetura da Rede Neural foi definida com base no número de exemplos de treinamento, número de atributos, dimensão VC aproximada e Regra de Ouro.

A rede possui:

```text
214 entradas
      ↓
Dense(99) + ReLU
      ↓
Dense(1) + Sigmoid
```

Para uma camada escondida com `n` neurônios e `d` entradas, o número de pesos e bias é:

\[
|W| = (d+1)n + (n+1)
\]

ou:

\[
|W| = (d+2)n + 1
\]

Com `d = 214` e `n = 99`:

\[
|W| = 21.385
\]

Foi adotada a aproximação:

\[
d_{VC} \approx |W|
\]

e a Regra de Ouro:

\[
N \geq 10d_{VC}
\]

Assim:

\[
10 \times 21.385 = 213.850
\]

Como:

\[
215.257 \geq 213.850
\]

a arquitetura satisfaz a regra adotada.

O Teorema da Aproximação Universal foi utilizado como justificativa para o uso de uma única camada escondida com função de ativação não linear.

### 7.2 Tratamento do desbalanceamento

Foram utilizados pesos de classe calculados somente sobre `y_train`:

- classe 0: aproximadamente **0,5439**;
- classe 1: aproximadamente **6,1937**.

Assim, os erros sobre a classe minoritária recebem maior importância durante o treinamento.

Não foram aplicados SMOTE, oversampling ou undersampling.

### 7.3 Treinamento

Configuração principal:

- otimizador: Adam;
- learning rate: 0,001;
- função de perda: Binary Cross-Entropy;
- `batch_size = 256`;
- máximo de 100 épocas;
- EarlyStopping com `patience = 10`;
- restauração dos melhores pesos.

O treinamento executou **17 épocas**, com melhor resultado de validação na **época 7**.

### 7.4 Overfitting e generalização

A melhor `val_loss` foi:

\[
E_{val} = 0,549509
\]

Após a melhor região, a BCE de treinamento continuou diminuindo enquanto a BCE de validação permaneceu acima do melhor valor por múltiplas épocas, indicando início de overfitting. O EarlyStopping evitou que os pesos finais fossem mantidos nessa região e restaurou os melhores pesos.

![Evolução da BCE da Rede Neural](figures/neural_network_loss.png)

Foram obtidos:

\[
E_{in} = 0,537344
\]

\[
E_{out} = 0,550239
\]

\[
E_{out} - E_{in} = 0,012895
\]

O pequeno gap indica boa capacidade de generalização da configuração final.

### 7.5 Métricas da Rede Neural

No conjunto de teste, com threshold de 0,50:

| Métrica | Resultado |
|---|---:|
| Acurácia | 0,716912 |
| Precisão | 0,169242 |
| Recall | 0,641246 |
| F1-score | 0,267803 |
| AUC | 0,7495 |

O principal destaque é o recall de aproximadamente **64,12%**, indicando que a Rede Neural identifica uma parcela maior dos casos positivos.

![Matriz de confusão da Rede Neural](figures/neural_network_confusion_matrix.png)

---

## 8. Árvore de Decisão

### 8.1 Árvore de referência

A Árvore de Decisão utiliza:

```python
class_weight='balanced'
min_samples_leaf=30
```

A pré-poda por `min_samples_leaf` evita folhas com poucas observações e reduz o custo de treinamento.

A árvore de referência apresentou:

- profundidade: **30**;
- folhas: **3.909**;
- `E_in = 0,2484`;
- `E_out = 0,3100`;
- gap: **0,0616**.

No teste, essa árvore apresentou:

- acurácia: 0,6900;
- precisão: 0,1351;
- recall: 0,5255;
- F1: 0,2149.

O gap maior motivou a aplicação de regularização por poda.

![Árvore antes da poda por custo-complexidade](figures/decision_tree_unpruned.png)

### 8.2 Minimal Cost-Complexity Pruning

A regularização utiliza o parâmetro `ccp_alpha`, que controla o compromisso entre ajuste aos dados e complexidade da árvore.

O caminho de poda apresentou **2.452 valores distintos de alpha**. Foram selecionados sete candidatos representativos.

A seleção foi realizada por **validação cruzada estratificada com 5 folds**.

Como:

\[
N = 215.257
\]

o fold de validação possui aproximadamente:

\[
\frac{215.257}{5} \approx 43.051
\]

exemplos.

Para reduzir o tempo de execução, cada candidato foi treinado em uma amostra estratificada de 60.000 registros do fold de treino e avaliado no fold de validação completo.

O melhor F1 médio obtido foi aproximadamente:

\[
0,2065
\]

com desvio-padrão aproximado de:

\[
0,0017
\]

O valor selecionado foi:

\[
ccp\_alpha = 3,6207798\times10^{-5}
\]

### 8.3 Threshold de decisão

Após o treinamento da árvore regularizada, o threshold foi otimizado na validação.

Com threshold 0,50:

- F1 de validação: 0,2157;
- precisão: 0,1321.

Com threshold 0,75:

- F1 de validação: **0,2580**;
- precisão: **0,1912**.

Portanto:

\[
threshold = 0,75
\]

foi escolhido.

### 8.4 Árvore final

A árvore regularizada final apresentou:

- profundidade: **30**;
- folhas: **2.322**;
- `E_in = 0,1535`;
- `E_out = 0,1826`;
- gap: **0,0292**.

Resultados no teste:

| Métrica | Resultado |
|---|---:|
| Acurácia | 0,817352 |
| Precisão | 0,193266 |
| Recall | 0,397691 |
| F1-score | 0,260121 |
| AUC | 0,6658 |

Matriz de confusão:

| | Predito 0 | Predito 1 |
|---|---:|---:|
| Real 0 | 36.221 | 6.182 |
| Real 1 | 2.243 | 1.481 |

![Árvore regularizada](figures/decision_tree_pruned.png)

### 8.5 Importância das variáveis

A variável mais importante da árvore final foi:

```text
EXT_SOURCE_MEAN
```

com importância aproximada de:

\[
0,3382
\]

Em seguida aparecem:

- `CREDIT_ANNUITY_RATIO`;
- `AGE_YEARS`;
- `EMPLOYMENT_YEARS`;
- `EXT_SOURCE_3`;
- `DAYS_ID_PUBLISH`;
- `DAYS_REGISTRATION`;
- `EXT_SOURCE_2`;
- `LTV`.

Esse resultado é coerente com a análise exploratória, na qual as pontuações externas já apareciam entre as variáveis com maior associação com `TARGET`.

---

## 9. Comparação dos modelos

A comparação no mesmo conjunto de teste é apresentada abaixo.

| Métrica | Rede Neural (thr 0,50) | Árvore (thr 0,75) |
|---|---:|---:|
| Acurácia | 0,716912 | **0,817352** |
| Precisão | 0,169242 | **0,193266** |
| Recall | **0,641246** | 0,397691 |
| F1 | **0,267803** | 0,260121 |
| AUC | **0,7495** | 0,6658 |

A Rede Neural apresenta maior capacidade de discriminação, maior recall e F1 ligeiramente superior. A Árvore obtém maior acurácia e precisão no threshold 0,75.

![Matrizes de confusão dos modelos](figures/comparison_confusion_matrices.png)

### 9.1 Curvas ROC

A curva ROC permite comparar os modelos ao longo de todos os thresholds possíveis.

Foram obtidos:

\[
AUC_{RN}=0,7495
\]

\[
AUC_{Árvore}=0,6658
\]

A maior AUC da Rede Neural mostra que ela possui melhor capacidade global de ordenar exemplos positivos acima dos negativos.

![Curvas ROC](figures/comparison_roc_curves.png)

### 9.2 Comparação de generalização

A Rede Neural utiliza Binary Cross-Entropy como medida de `E_in` e `E_out`, enquanto a Árvore utiliza erro de classificação (`1 - acurácia`) em sua análise própria. Portanto, esses valores não devem ser comparados diretamente entre modelos.

Para uma comparação comum, também foi calculado o log-loss da Árvore:

- Rede Neural:
  - treino: 0,5373;
  - teste: 0,5502.

- Árvore:
  - treino: 0,4635;
  - teste: 1,0163.

No teste, o log-loss da Rede Neural é substancialmente menor, indicando probabilidades mais adequadas sob essa métrica.

---

## 10. Escolha do melhor modelo

A **Rede Neural foi selecionada como o melhor modelo final**.

Os principais critérios foram:

1. maior AUC:
   - Rede Neural: 0,7495;
   - Árvore: 0,6658;

2. maior recall:
   - Rede Neural: 0,6412;
   - Árvore: 0,3977;

3. maior F1:
   - Rede Neural: 0,2678;
   - Árvore: 0,2601;

4. menor log-loss no teste:
   - Rede Neural: 0,5502;
   - Árvore: 1,0163;

5. pequeno gap de generalização da Rede Neural.

A Árvore, entretanto, possui uma vantagem relevante em **interpretabilidade**, pois suas decisões podem ser representadas como regras explícitas. Por isso, ela pode ser preferida em aplicações em que a explicabilidade tenha prioridade sobre a capacidade preditiva.

---

## 11. Limitações

O projeto possui algumas limitações importantes.

### 11.1 Desbalanceamento

A classe positiva representa apenas cerca de 8% da base. Apesar do uso de `class_weight='balanced'`, a precisão da classe positiva permanece baixa.

### 11.2 Dados utilizados

A primeira versão utiliza apenas `application_train.csv`. As tabelas históricas `bureau.csv` e `previous_application.csv` não foram incorporadas.

### 11.3 EDA realizada antes do split

Parte da análise exploratória foi realizada sobre a base completa. As estatísticas efetivamente utilizadas no pré-processamento e treinamento foram ajustadas sobre o treinamento, mas a EDA prévia deve ser interpretada como exploração descritiva.

### 11.4 Probabilidades da Árvore

O log-loss elevado da Árvore no teste indica que suas probabilidades são menos bem calibradas que as produzidas pela Rede Neural.

### 11.5 Acesso ao teste na árvore de referência

Na versão atualmente versionada do notebook, a árvore de referência sem poda também tem métricas calculadas em `X_test_tree` e `y_test` antes da seleção final do `ccp_alpha`. Esse cálculo não é utilizado para escolher `ccp_alpha` ou threshold, que são definidos com treino e validação, mas representa um acesso antecipado ao conjunto de teste.

Para aderência literal ao enunciado da disciplina, recomenda-se substituir, nessa etapa de diagnóstico da árvore de referência, as métricas de teste por métricas de validação e reservar o teste exclusivamente à avaliação final.

---

## 12. Trabalhos futuros

Como extensões do projeto, podem ser avaliadas:

- incorporação de `bureau.csv`;
- incorporação de `previous_application.csv`;
- calibração das probabilidades;
- regularização L2 ou Dropout na Rede Neural;
- busca mais ampla de hiperparâmetros;
- ensembles combinando Rede Neural e Árvore;
- análise explícita do custo de falsos positivos e falsos negativos;
- comparação com outros classificadores.

---

## 13. Conclusão

O projeto demonstrou a aplicação completa de um pipeline de Aprendizagem de Máquina para classificação de risco de crédito.

A análise exploratória revelou um conjunto fortemente desbalanceado e mostrou que as pontuações externas de risco possuem papel relevante na separação das classes. O pré-processamento resultou em 214 atributos finais e foi estruturado de forma diferente para a Rede Neural e para a Árvore de Decisão.

A arquitetura da Rede Neural foi definida utilizando a dimensão VC aproximada e a Regra de Ouro, resultando em 99 neurônios escondidos e 21.385 parâmetros. O EarlyStopping identificou a melhor época como a época 7 e limitou o efeito do overfitting.

Na Árvore de Decisão, a validação cruzada permitiu selecionar o `ccp_alpha` e reduzir a complexidade da árvore. A otimização do threshold aumentou precisão e F1, embora com redução do recall.

Na comparação final, a Rede Neural apresentou AUC, recall e F1 superiores, além de log-loss de teste significativamente menor. Por esses critérios, ela foi selecionada como **melhor modelo para o conjunto analisado**.

A Árvore de Decisão, por outro lado, mostrou maior interpretabilidade e maior precisão no threshold otimizado, oferecendo um perfil complementar.

Assim, os resultados mostram que a escolha do modelo depende não apenas da acurácia, mas do objetivo da aplicação e do custo associado aos diferentes tipos de erro.

---

## 14. Reprodutibilidade

O projeto está organizado com:

```text
notebooks/
    credit_risk_classification.ipynb

reports/
    figures/
    models/

data/
    raw/
```

Os dados brutos não são versionados no GitHub devido ao tamanho dos arquivos.

Os artefatos de treinamento incluem:

```text
reports/models/neural_network.keras
reports/models/neural_network_config.json
reports/models/neural_network_history.csv
reports/models/decision_tree_pruned.pkl
```

As figuras geradas pelo notebook são armazenadas em:

```text
reports/figures/
```

---

## Referência do conjunto de dados

**Home Credit Default Risk.** Kaggle Competition.  
Dataset utilizado para fins acadêmicos na disciplina de Aprendizagem de Máquina.
