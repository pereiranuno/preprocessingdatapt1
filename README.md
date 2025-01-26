
# Projeto Pre Processamento de Dados

## Descrição
Este projeto tem como objetivo prever os preços de casas com base em um conjunto de variáveis relacionadas a características dos imóveis. Foi desenvolvido um notebook que envolve o pré-processamento dos dados, a aplicação de técnicas de transformação e o uso de diferentes abordagens de feature engineering para otimizar o desempenho do modelo preditivo.

## Estrutura do Projeto

### 1. Carregamento e Análise Inicial dos Dados
- Importação e inspeção inicial dos dados.
- Identificação de variáveis relevantes para o modelo.

### 2. Baseline Inicial
- Definição de um baseline utilizando variáveis-chave:
  - **YearBuilt**
  - **LotFrontage**
  - **MasVnrType**
  - **SaleCondition**
  - **GarageArea**
  - **GarageCars**
- Avaliação inicial do desempenho do modelo.

### 3. Transformações Aplicadas

#### a) Ordinal Encoding
- Aplicado à variável **MasVnrType**.
- Impacto limitado no desempenho do modelo.

#### b) One-Hot Encoding
- Aplicado à variável **SalesCondition**.
- Pequeno aumento no desempenho do modelo ao capturar relações entre as categorias e o preço.

#### c) Bins Quantile Strategy
- Divisão da variável **LotFrontage** em categorias baseadas nos quantis.
- Redução do impacto de outliers e melhoria significativa no desempenho.

#### d) Binning Uniform com e sem Imputação
- A variável **GarageArea** foi categorizada com bins uniformes.
- Imputação pela mediana resultou em perda de informação relevante. Removida em iterações seguintes.

#### e) Feature Scaling
- Escalonamento de **GarageAreaUniform** e **GarageCars** com **MinMaxScaler** para o intervalo [-1, 1].
- Melhorias marginais no desempenho ao uniformizar escalas.

#### f) Feature Engineering
- Transformou a variável **YearBuilt** em **Idade do Edifício** (**Ano Atual - Ano de Construção**).
- Maior impacto no desempenho, com aumento significativo no score final.

### 4. Resultados e Conclusões
- O modelo final apresentou uma melhoria significativa no score, de **0.741** (baseline inicial) para **0.793**.
- Abordagens como **Feature Engineering** e **Bins Quantile Strategy** foram fundamentais para o aumento de desempenho.

## Tecnologias Utilizadas
- **Python**
- Bibliotecas:
  - **pandas**: Manipulação de dados.
  - **numpy**: Operações numéricas.
  - **scikit-learn**: Transformações e modelagem.
  - **seaborn** e **matplotlib**: Visualizações.



