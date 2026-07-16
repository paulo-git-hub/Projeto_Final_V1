# Inteligência Artificial para Análise Preditiva Imobiliária (King County, EUA)

## 📋 Declaração do Problema de Negócio

No mercado imobiliário moderno, a precificação incorreta de ativos pode acarretar sérios prejuízos financeiros para imobiliárias, investidores e instituições financeiras. Estimar o valor de venda de um imóvel de forma empírica ou com base em suposições lineares simples frequentemente resulta em modelos enviesados que falham significativamente quando expostos a cenários reais de mercado.

**Objetivo do Projeto:** Desenvolver um pipeline de Machine Learning escalável e robusto capaz de prever o valor numérico contínuo (variável-alvo `price` em dólares) de imóveis localizados no condado de King County (EUA), com base em suas características físicas e geográficas.

**Impacto no Negócio:** Mitigar estimativas equivocadas, otimizar as margens do portfólio da imobiliária e apoiar decisões sequer de compra, venda ou concessão de crédito imobiliário.

---

## 🛠️ Estrutura do Projeto

O projeto foi estruturado seguindo as melhores práticas de desenvolvimento de software e ciência de dados, separando a lógica de execução (Notebook) dos submódulos reutilizáveis:

```text
├── data/
│   ├── raw/         # Datasets brutos originais
│   └── processed/   # Datasets limpos e tratados após o Data Prep
├── notebook/
│   └── Projeto_Final_V1.ipynb   # Notebook principal de execução do pipeline
├── outputs/
│   └── figures/     # Gráficos e mapas de calor exportados automaticamente
├── src/             # Código-fonte do projeto (submódulos)
│   ├── __init__.py
│   ├── dataset.py   # Funções de carga e higienização estrutural
│   ├── features.py  # Engenharia de recursos e preparação para modelagem
│   ├── modeling/
│   │   └── train.py # Pipeline de treinamento e validação cruzada
│   └── plots.py     # Lógica de geração de gráficos (EDA e Avaliação)
└── .gitignore       # Arquivo de configuração para omissão de dados e binários
```

---

## 🔬 Pipeline de Machine Learning

### 1. Análise Exploratória de Dados (EDA)
Foi realizada uma análise estatística descritiva para compreender as dimensões, tipos primitivos e distribuições do dataset original (21.613 linhas e 21 colunas).

* **Distribuição da Variável-Alvo:** O histograma de `price` revelou uma assimetria positiva acentuada (cauda longa à direita), indicando a necessidade futura de transformações logarítmicas para estabilizar a variância dos erros em modelos lineares.
* **Análise de Relações:** Variáveis como área construída (`sqft_living`) e qualidade da construção (`grade`) demonstraram forte correlação positiva com o preço.

### 2. Data Prep & Tratamento de Dados Ruidosos
Para evitar o fenômeno *Garbage In, Garbage Out*, os dados passaram por um rigoroso tratamento estatístico:

* **Valores Ausentes:** Identificados nulos na coluna `sqft_above`, tratados por meio de imputação pela mediana (1560.0), evitando as distorções que a média causaria devido à assimetria dos dados.
* **Tratamento de Outliers:** Identificação e remoção de registros inconsistentes (ex: um imóvel com 33 quartos), que afetariam severamente os coeficientes da regressão linear.
* **Multicolinearidade (VIF):** Foi calculado o Fator de Inflação da Variância (VIF). Variáveis com alta colinearidade (como `sqft_above` com VIF de 5.10, `sqft_living15` e `sqft_lot15`) e identificadores redundantes (`id`) foram removidos para garantir a estabilidade do modelo.

### 3. Feature Engineering
Foram concebidas novas variáveis para extrair maior valor do ativo:

* **`idade_imovel`:** Calculado a partir da diferença entre o ano de venda e o ano de construção.
* **`foi_reformado`:** Variável binária (0 ou 1) derivada do ano de renovação.
* **Prevenção de *Data Leakage*:** Variáveis como preço por metro quadrado foram utilizadas apenas na EDA e omitidas da modelagem para evitar vazamento de dados.

### 4. Validação Cruzada e Treinamento
Os dados foram divididos em 80% para treinamento e 20% para teste. O escalonamento dos dados foi realizado com o `StandardScaler`, aplicando o método `fit_transform` estritamente nos dados de treino e apenas `transform` nos dados de teste. Foi aplicada a metodologia de Validação Cruzada (*K-Fold* com 5 partições) para garantir a estabilidade das métricas.

---

## 📊 Desempenho e Diagnóstico dos Modelos

O algoritmo base de Regressão Linear foi confrontado com um modelo não-linear de Árvore de Decisão (`DecisionTreeRegressor`):

| Métrica | Regressão Linear (Treino) | Regressão Linear (Teste) | Árvore de Decisão (Treino) | Árvore de Decisão (Teste) |
| :--- | :---: | :---: | :---: | :---: |
| **MAE** (Erro Médio Absoluto) | $125.216,26 | $126.898,22 | $110.677,60 | $121.132,92 |
| **RMSE** (Erro Quadrático Médio) | $199.626,15 | $213.457,97 | $182.420,46 | $219.292,13 |
| **$R^2$** (Coeficiente de Determinação) | — | **0.6964** | — | 0.6796 |

### 🔍 Diagnóstico Técnico:

* **Modelo Campeão:** A Regressão Linear foi selecionada como o modelo campeão por apresentar maior generalização no conjunto de teste ($R^2$ de 69.64% e menor RMSE).
* **Overfitting Detectado:** O modelo de Árvore de Decisão apresentou forte indício de *overfitting* (sobreajuste). Embora tenha obtido o menor MAE no conjunto de treino ($110k), seu desempenho decaiu drasticamente no teste, com o RMSE disparando para $219k. Isso ocorre porque árvores não podadas tendem a decorar o ruído dos dados de treino em vez de aprender o padrão geral.

---

## 💼 Interpretação e Veredito de Negócio

Apesar do rigor estatístico aplicado no desenvolvimento, o diagnóstico financeiro indica que nenhum dos modelos está pronto para produção comercial:

* **Margem de Erro Inaceitável:** Um MAE de $126.898,22 significa que o modelo erra, em média, essa quantia por imóvel. Em uma propriedade padrão de $500.000,00, o erro representa mais de 25% do valor total do ativo.
* **Risco de Prejuízo:** Precificar um imóvel com essa margem de erro faria com que a imobiliária comprasse ativos supervalorizados ou vendesse seu próprio inventário muito abaixo do preço de mercado, gerando quebra de caixa.
* **Inviabilidade Bancária:** Para a concessão de crédito ou avaliação de garantias imobiliárias, um erro dessa magnitude é perigoso, pois expõe a instituição a um risco de inadimplência desalinhado com o valor real do colateral.

### 🚀 Próximos Passos:
Para elevar o $R^2$ acima de 85% e reduzir o MAE a níveis comercialmente aceitáveis, as próximas iterações do projeto devem focar em:

1. Aplicação de transformação logarítmica na variável-alvo para corrigir a assimetria positiva.
2. Utilização de algoritmos de *Ensemble* (como *Random Forest* ou *XGBoost*) para mitigar o *overfitting* da árvore isolada e capturar padrões não-lineares.
3. **Aperfeiçoamento do impacto geográfico:** Refinar o agrupamento por classes da variável `zipcode` realizado no Data Prep ou utilizar algoritmos de clusterização espacial (como o K-Means) diretamente nas variáveis `lat` e `long` para capturar com maior precisão as nuances de micro-localização e vizinhança do mercado imobiliário.

---
> *Projeto desenvolvido como trabalho de conclusão do Módulo 1 do Curso Desenvolvimento de IA para Análise Preditiva - Carga Horária:150 horas