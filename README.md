# Classificação de Risco de Crédito

![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Projeto de aprendizagem de máquina para identificação de solicitações de crédito associadas à dificuldade de pagamento, utilizando **Redes Neurais** e **Árvores de Decisão**.

O projeto utiliza dados da competição **Home Credit Default Risk**, disponibilizada no Kaggle, e está sendo desenvolvido como trabalho acadêmico da disciplina de Aprendizagem de Máquina.

---

## Sumário

- [Visão geral](#visão-geral)
- [Problema investigado](#problema-investigado)
- [Objetivos](#objetivos)
- [Conjunto de dados](#conjunto-de-dados)
- [Variável-alvo](#variável-alvo)
- [Metodologia](#metodologia)
- [Modelos](#modelos)
- [Métricas de avaliação](#métricas-de-avaliação)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Status do projeto](#status-do-projeto)
- [Limitações e uso responsável](#limitações-e-uso-responsável)
- [Tecnologias](#tecnologias)
- [Licença](#licença)
- [Referência do dataset](#referência-do-dataset)

---

## Visão geral

A análise de risco de crédito é uma aplicação de aprendizagem de máquina relacionada à identificação de solicitações que apresentam maior possibilidade de dificuldade de pagamento.

Neste projeto, informações disponíveis sobre o solicitante e sobre a solicitação de crédito serão utilizadas para treinar e comparar dois modelos supervisionados:

1. uma **Rede Neural**;
2. uma **Árvore de Decisão regularizada**.

O foco não é criar um sistema automático de aprovação ou rejeição de crédito. O objetivo é realizar uma comparação acadêmica entre os modelos, analisando desempenho, generalização, overfitting, regularização e interpretabilidade.

---

## Problema investigado

O projeto é formulado como um problema supervisionado de **classificação binária**.

A principal pergunta investigada é:

> É possível utilizar informações sobre o solicitante e sobre a solicitação de crédito para identificar casos associados à dificuldade de pagamento?

Em uma etapa posterior, também será investigada a seguinte questão:

> A inclusão de informações sobre créditos e solicitações anteriores melhora o desempenho dos modelos?

---

## Objetivos

### Objetivo geral

Desenvolver e comparar uma Rede Neural e uma Árvore de Decisão para classificação de solicitações de crédito associadas à dificuldade de pagamento.

### Objetivos específicos

- compreender e documentar o conjunto de dados;
- identificar o número de amostras e de atributos;
- analisar a distribuição da variável-alvo;
- investigar valores ausentes, duplicatas e possíveis inconsistências;
- tratar variáveis numéricas e categóricas;
- criar atributos com interpretação financeira;
- construir a matriz de características \(X\) e o vetor-alvo \(y\);
- separar os dados em treinamento, validação e teste;
- definir a arquitetura da Rede Neural;
- justificar a arquitetura utilizando a dimensão VC e a Regra de Ouro;
- calcular \(E_{in}\) e \(E_{out}\);
- identificar a ocorrência de overfitting;
- justificar o número de épocas e o batch size;
- construir uma Árvore de Decisão sem poda;
- regularizar a árvore utilizando Minimal Cost-Complexity Pruning;
- selecionar o valor de `ccp_alpha` com validação cruzada;
- avaliar os modelos com acurácia, precisão, recall e F1-score;
- comparar os modelos no mesmo conjunto de teste;
- selecionar o modelo com melhor capacidade de generalização.

---

## Conjunto de dados

O projeto utiliza o dataset **Home Credit Default Risk**, disponibilizado pelo Home Credit Group por meio de uma competição no Kaggle.

Página do dataset:

https://www.kaggle.com/competitions/home-credit-default-risk/data

### Arquivo principal

#### `application_train.csv`

É a tabela principal do projeto.

Cada linha representa uma solicitação de crédito e contém:

- informações do solicitante;
- características familiares;
- informações de renda e emprego;
- características da moradia;
- valor solicitado;
- valor da parcela;
- tipo de contrato;
- pontuações externas;
- variável-alvo `TARGET`.

A primeira versão completa do projeto será desenvolvida somente com esse arquivo.

### Arquivos auxiliares

#### `bureau.csv`

Contém informações sobre créditos anteriores dos clientes registrados por outras instituições financeiras.

Como um cliente pode possuir vários créditos anteriores, esse arquivo apresenta uma relação de um para muitos com a tabela principal.

Antes de sua integração, as informações serão agrupadas por `SK_ID_CURR`.

#### `previous_application.csv`

Contém solicitações anteriores de crédito realizadas pelos clientes na própria Home Credit.

Um mesmo cliente pode possuir várias solicitações anteriores. Por isso, essa tabela também será agregada por `SK_ID_CURR` antes da junção com a tabela principal.

### Ordem de utilização dos arquivos

O projeto será desenvolvido em duas etapas:

#### Etapa 1 — Modelo principal

```text
application_train.csv
```

Essa etapa será suficiente para desenvolver:

- análise exploratória;
- pré-processamento;
- Rede Neural;
- Árvore de Decisão;
- regularização;
- avaliação;
- comparação final.

#### Etapa 2 — Modelo ampliado

```text
application_train.csv
+ bureau.csv
+ previous_application.csv
```

Nessa etapa serão criadas características históricas, como:

- quantidade de créditos anteriores;
- quantidade de solicitações anteriores;
- proporção de solicitações aprovadas;
- proporção de solicitações recusadas;
- valor médio solicitado anteriormente;
- valor médio concedido;
- quantidade de créditos ativos;
- frequência de diferentes tipos de contrato.

O objetivo será verificar se a inclusão do histórico melhora o desempenho dos modelos.

### Armazenamento dos dados

Os arquivos CSV devem ser colocados localmente em:

```text
data/raw/
```

Os datasets não são enviados ao GitHub. A pasta está configurada no `.gitignore` para evitar o versionamento de arquivos grandes e manter apenas a estrutura do diretório.

---

## Variável-alvo

A variável-alvo \(y\) corresponde à coluna `TARGET`:

\[
y = \texttt{TARGET}
\]

Em que:

- `TARGET = 0`: solicitação não associada à classe de dificuldade de pagamento;
- `TARGET = 1`: solicitação associada à classe de dificuldade de pagamento.

A classe positiva do projeto será:

```text
TARGET = 1
```

A coluna `TARGET` será retirada da matriz de entrada antes do treinamento.

A coluna `SK_ID_CURR` também não será usada diretamente como característica preditiva, pois representa apenas o identificador do cliente. Ela será mantida somente quando necessária para relacionar as diferentes tabelas.

---

## Metodologia

O projeto seguirá as etapas abaixo.

### 1. Carregamento e inspeção

- carregamento do `application_train.csv`;
- identificação do número de linhas e colunas;
- inspeção dos tipos das variáveis;
- verificação de duplicatas;
- análise dos valores ausentes;
- verificação da distribuição da variável `TARGET`.

### 2. Análise exploratória

Serão analisados:

- balanceamento das classes;
- distribuição da renda;
- valor do crédito;
- valor das parcelas;
- idade;
- tempo de emprego;
- composição familiar;
- variáveis categóricas mais frequentes;
- possíveis valores extremos;
- possíveis relações entre os atributos e a variável-alvo.

Os principais gráficos serão salvos em:

```text
reports/figures/
```

### 3. Limpeza dos dados

A etapa de limpeza poderá incluir:

- correção de tipos;
- identificação de códigos utilizados para representar valores ausentes;
- tratamento de valores impossíveis;
- remoção de colunas constantes;
- remoção de identificadores;
- tratamento de categorias raras;
- análise de colunas com percentual elevado de dados ausentes.

Nenhuma informação do conjunto de teste será utilizada para definir regras de limpeza ou transformação.

### 4. Engenharia de atributos

Serão criados atributos com interpretação financeira, como:

\[
\text{Crédito/Renda}
=
\frac{\text{valor do crédito}}{\text{renda anual}}
\]

\[
\text{Parcela/Renda}
=
\frac{\text{valor da parcela}}{\text{renda anual}}
\]

Outras possibilidades incluem:

- idade em anos;
- tempo de emprego em anos;
- valor do crédito por integrante da família;
- quantidade de filhos por integrante da família;
- relação entre valor do crédito e valor da parcela;
- indicadores de presença ou ausência de determinadas informações.

Todos os atributos criados serão documentados no notebook.

### 5. Construção de \(X\) e \(y\)

A matriz de características será definida como:

\[
X = \text{informações do solicitante e da solicitação}
\]

O vetor-alvo será:

\[
y = \texttt{TARGET}
\]

As colunas `TARGET` e `SK_ID_CURR` não serão utilizadas como características preditivas.

### 6. Separação dos dados

Os dados serão divididos em:

- treinamento;
- validação;
- teste.

A divisão será estratificada para preservar aproximadamente a proporção das classes.

O conjunto de teste permanecerá isolado durante:

- seleção de atributos;
- definição da arquitetura;
- escolha do número de épocas;
- seleção do batch size;
- escolha do `ccp_alpha`;
- ajuste de hiperparâmetros.

O teste será utilizado somente na avaliação final.

### 7. Pré-processamento

#### Variáveis numéricas

- imputação de valores ausentes;
- análise de valores extremos;
- padronização para a Rede Neural.

#### Variáveis categóricas

- imputação de valores ausentes;
- transformação por One-Hot Encoding;
- tratamento de categorias desconhecidas.

O pré-processamento será ajustado somente com os dados de treinamento.

A Rede Neural e a Árvore de Decisão poderão utilizar pipelines diferentes, pois a árvore não exige a padronização das variáveis numéricas.

---

## Modelos

### Rede Neural

A Rede Neural será construída utilizando TensorFlow e Keras.

A arquitetura será definida considerando:

- número de amostras de treinamento;
- número de características após o pré-processamento;
- quantidade de parâmetros treináveis;
- dimensão VC;
- Regra de Ouro;
- presença de bias em cada neurônio;
- Teorema da Aproximação Universal.

A configuração inicial deverá conter:

- camada de entrada;
- uma ou mais camadas ocultas;
- ativação ReLU nas camadas ocultas;
- um neurônio na saída;
- ativação sigmoide;
- função de perda `binary_crossentropy`;
- otimizador Adam.

Durante o treinamento serão acompanhadas:

- perda de treinamento;
- perda de validação;
- acurácia de treinamento;
- acurácia de validação.

O overfitting será analisado observando o momento em que o erro de treinamento continua diminuindo enquanto o erro de validação começa a aumentar.

Também serão justificados:

- número máximo de épocas;
- batch size;
- critério de parada;
- configuração escolhida.

### Árvore de Decisão

Primeiramente será treinada uma Árvore de Decisão sem poda.

Serão analisados:

- profundidade;
- número de folhas;
- erro de treinamento;
- erro de validação;
- diferença entre \(E_{in}\) e \(E_{out}\);
- possível ocorrência de overfitting.

Depois será aplicado o algoritmo de **Minimal Cost-Complexity Pruning**.

O procedimento será:

1. obter os valores candidatos de `ccp_alpha`;
2. treinar árvores com diferentes valores;
3. avaliar os candidatos por validação cruzada;
4. selecionar o valor com melhor desempenho de validação;
5. preferir a árvore mais simples em caso de desempenho semelhante;
6. treinar a árvore regularizada;
7. calcular as métricas;
8. visualizar a árvore final.

---

## Métricas de avaliação

Os modelos serão avaliados com:

- acurácia;
- precisão;
- recall;
- F1-score;
- matriz de confusão.

Como a classe positiva pode ser menos frequente, a acurácia não será analisada isoladamente.

### Interpretação dos erros

#### Falso negativo

O modelo classifica como baixo risco um caso que pertence à classe de dificuldade de pagamento.

#### Falso positivo

O modelo classifica como arriscado um caso que não pertence à classe de dificuldade de pagamento.

O recall da classe positiva receberá atenção especial, pois mede a proporção dos casos de dificuldade de pagamento corretamente identificados.

A comparação final considerará:

- desempenho preditivo;
- capacidade de generalização;
- diferença entre \(E_{in}\) e \(E_{out}\);
- estabilidade;
- complexidade;
- interpretabilidade.

---

## Estrutura do repositório

```text
credit-risk-classification/
├── data/
│   ├── raw/
│   │   ├── application_train.csv
│   │   ├── bureau.csv
│   │   └── previous_application.csv
│   └── processed/
├── notebooks/
│   └── credit_risk_classification.ipynb
├── reports/
│   └── figures/
├── src/
│   └── __init__.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Descrição das pastas

| Caminho | Finalidade |
|---|---|
| `data/raw/` | Arquivos brutos, sem alterações |
| `data/processed/` | Dados resultantes do pré-processamento |
| `notebooks/` | Notebook principal do projeto |
| `reports/figures/` | Gráficos utilizados no relatório |
| `src/` | Código modular que poderá ser extraído do notebook |
| `requirements.txt` | Dependências do ambiente Python |

---

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/viniciusrodriguesai/credit-risk-classification.git
cd credit-risk-classification
```

### 2. Criar um ambiente virtual

```bash
python -m venv .venv
```

### 3. Ativar o ambiente

No Git Bash do Windows:

```bash
source .venv/Scripts/activate
```

No Linux ou macOS:

```bash
source .venv/bin/activate
```

### 4. Instalar as dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Baixar o dataset

Acesse:

https://www.kaggle.com/competitions/home-credit-default-risk/data

Baixe inicialmente:

```text
application_train.csv
bureau.csv
previous_application.csv
```

Coloque os arquivos em:

```text
data/raw/
```

O arquivo obrigatório para iniciar o projeto é:

```text
application_train.csv
```

Os outros dois serão utilizados somente após a conclusão da primeira versão.

### 6. Abrir o notebook

```bash
jupyter lab notebooks/credit_risk_classification.ipynb
```

Também é possível abrir o repositório no VS Code:

```bash
code .
```

Em seguida, abra:

```text
notebooks/credit_risk_classification.ipynb
```

---

## Status do projeto

O projeto está em desenvolvimento.

### Concluído

- [x] definição do tema;
- [x] escolha do dataset;
- [x] criação do repositório;
- [x] organização inicial das pastas;
- [x] configuração do `.gitignore`;
- [x] criação do notebook;
- [x] descrição do problema;
- [x] descrição inicial do dataset.

### Em desenvolvimento

- [ ] carregamento completo dos dados;
- [ ] análise exploratória;
- [ ] tratamento de valores ausentes;
- [ ] engenharia de atributos;
- [ ] construção de \(X\) e \(y\);
- [ ] separação de treinamento, validação e teste;
- [ ] pré-processamento;
- [ ] definição da arquitetura da Rede Neural;
- [ ] treinamento da Rede Neural;
- [ ] cálculo de \(E_{in}\) e \(E_{out}\);
- [ ] análise de overfitting;
- [ ] treinamento da Árvore de Decisão;
- [ ] seleção do `ccp_alpha`;
- [ ] validação cruzada;
- [ ] comparação dos modelos;
- [ ] avaliação no conjunto de teste;
- [ ] integração de `bureau.csv`;
- [ ] integração de `previous_application.csv`;
- [ ] relatório final;
- [ ] apresentação.

---

## Resultados

Os resultados ainda não estão disponíveis.

Esta seção será atualizada somente após:

1. conclusão do pré-processamento;
2. definição dos modelos;
3. seleção dos hiperparâmetros;
4. avaliação final no conjunto de teste.

Serão apresentados:

- tabela comparativa dos modelos;
- matriz de confusão;
- acurácia;
- precisão;
- recall;
- F1-score;
- \(E_{in}\);
- \(E_{out}\);
- gráficos de treinamento;
- análise de overfitting;
- profundidade e quantidade de folhas da árvore;
- valor final de `ccp_alpha`.

---

## Limitações e uso responsável

Este projeto possui finalidade acadêmica e experimental.

Os resultados não devem ser utilizados diretamente para:

- aprovar ou rejeitar solicitações de crédito;
- determinar limites de crédito;
- substituir a análise humana;
- tomar decisões financeiras reais;
- avaliar pessoas fora do contexto e da população representada pelo dataset.

Entre as limitações esperadas estão:

- dados anonimizados;
- possíveis valores ausentes;
- desbalanceamento entre as classes;
- ausência de determinadas informações de contexto;
- possíveis mudanças entre o cenário original da base e outros mercados;
- risco de vieses presentes nos dados históricos;
- dificuldade de interpretar completamente modelos neurais.

Uma boa métrica de classificação não significa que o modelo esteja pronto para uso real. Sistemas de crédito exigem validação adicional, análise jurídica, governança, monitoramento e avaliação de equidade.

---

## Tecnologias

- Python;
- Pandas;
- NumPy;
- Matplotlib;
- Scikit-learn;
- TensorFlow;
- Keras;
- JupyterLab;
- Git;
- GitHub.

---

## Licença

Este projeto está disponibilizado sob a licença MIT.

Consulte o arquivo:

```text
LICENSE
```

---

## Referência do dataset

**Home Credit Default Risk**

Kaggle Competition:

https://www.kaggle.com/competitions/home-credit-default-risk

Dados:

https://www.kaggle.com/competitions/home-credit-default-risk/data

O dataset pertence aos seus respectivos autores e está sujeito às regras e condições da competição do Kaggle.
