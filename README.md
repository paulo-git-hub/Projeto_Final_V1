# 🏠 King County House Pricing: Pipeline Preditivo de Machine Learning (v1)

Este repositório contém o projeto de conclusão do Módulo 1 da disciplina de **Desenvolvimento de IA para Análise Preditiva**. O objetivo principal é construir um pipeline de Machine Learning de ponta a ponta, altamente modularizado, reprodutível e documentado, aplicando rigor estatístico no tratamento de dados e modelagem preditiva.

---

## 🎯 1. O Problema de Negócio e o Dataset

No mercado de tecnologia e mercado imobiliário atual, alimentar algoritmos preditivos com dados ruidosos, inconsistentes ou mal tratados resulta no fenômeno *Garbage In, Garbage Out* (Lixo Entra, Lixo Sai), gerando prejuízos financeiros.

* **O Desafio**: Uma imobiliária do condado de King County (EUA) deseja estimar de forma precisa o preço de venda de residências.
* **Variável-Alvo**: `price` (Valor numérico contínuo em dólares).
* **Base de Dados**: `kc_house_data.csv`, contendo aproximadamente 21 mil registros de transações imobiliárias com atributos físicos e geográficos.
* **Impacto de Negócio**: Apoiar decisões de precificação, compra, venda e financiamento de ativos imobiliários com base em um erro médio controlado.

---

## 🏗️ 2. Estrutura do Projeto e Reprodutibilidade

O projeto foi estruturado seguindo as práticas de engenharia de software e MLOps. Embora o pipeline seja orquestrado por um único notebook, a lógica técnica foi modularizada em arquivos `.py` dentro da pasta `src/`.

```text
projeto_final_ml_king_county/
├── data/
│   ├── raw/                # Dataset bruto (original)
│   ├── processed/          # Dataset higienizado (Data Prep)
│   └── final/              # Dataset recortado para a modelagem
├── models/
│   └── v1/
│       ├── modelo_regressao_vl.pkl   # Binário do modelo campeão treinado
│       └── metricas_vl.json          # Metadados e métricas de desempenho v1
├── notebooks/
│   └── projeto_final_v1.ipynb     # Notebook orquestrador (.ipynb)
├── outputs/
│   └── figures/            # Gráficos analíticos exportados automaticamente
├── src/                    # Pacote de módulos Python (Opcional/Diferencial)
│   ├── __init__.py         # Inicializador de pacote
│   ├── config.py           # Configuração de caminhos e sementes globais
│   ├── dataset.py          # Carga de dados, EDA estatística e limpeza
│   ├── features.py         # Engenharia de atributos, split e escala
│   ├── plots.py            # Geração de visualizações gráficas
│   └── modeling/
│       ├── __init__.py     # Inicializador do submódulo
│       └── train.py        # Treinamento, validação e persistência
├── .gitignore              # Arquivo de desconsideração do Git
├── requirements.txt        # Biblioteca e dependências do ambiente
├── LICENSE                 # Licença de uso do software
└── README.md               # Documentação técnica do projeto
```

---

## 🛠️ 3. O Pipeline de Machine Learning (Fases 1 a 6)

### 📊 Fase 1: Análise Exploratória de Dados (EDA)
* **Estatística Descritiva**: Avaliação das dimensões do dataset, identificação de tipos primitivos e resumo estatístico utilizando o método `.describe()`.
* **Visualização de Dados**: Foram plotados e exportados 3 gráficos analíticos para a pasta `outputs/figures/`: histograma de preços (avaliando assimetria), dispersão entre variáveis (`sqft_living` e `grade`) contra a variável-alvo, e o mapa de correlação de Pearson.
* **Análise Textual Crítica**: Identificação de severa assimetria positiva no preço e correlação crítica de colinearidade entre variáveis explicativas.

### 🧹 Fase 2: Tratamento e Limpeza (Data Prep)
* **Linhas Duplicadas**: Verificação sistemática para remoção de redundâncias amostrais.
* **Valores Ausentes**: Identificados e tratados. Justifica-se o uso da **Mediana** em vez da Média por se tratar de um dataset imobiliário de distribuição altamente assimétrica, blindando as imputações contra distorções de outliers.
* **Gerenciamento de Outliers**: Detecção via diagramas de caixa (boxplots) e tratamento do registro aberrante de 33 quartos (erro clássico de digitação), protegendo os coeficientes da regressão linear de desvios extremos.

### 📐 Fase 3: Feature Engineering (Colunas Calculadas)
* **Novas Features**: Geração da variável `idade_imovel` (ano da venda obtido de `date` menos o ano de construção `yr_built`) e da variável binária `foi_reformado`.
* **Prevenção de Vazamento de Dados (Data Leakage)**: Variáveis como "preço por metro quadrado" foram avaliadas apenas para leitura e EDA, sendo explicitamente excluídas das preditoras do modelo, pois sua permanência faria o modelo "vazar" informações do alvo.

### 🔌 Fase 4: Preparação para Modelagem
* **Mitigação de Multicolinearidade**: Remoção preventiva da variável explicativa `sqft_above` devido ao altíssimo índice de correlação linear (maior que 0.85) com `sqft_living`, o que inflaria as variâncias dos coeficientes da regressão.
* **Divisão Amostral (Split)**: Separação de X e y, fragmentando os conjuntos em 80% treino e 20% teste com semente aleatória fixa.
* **Escalonamento Seguro**: Padronização através do `StandardScaler`, aplicando `fit_transform` exclusivamente nos dados de treino e apenas `transform` nos dados de teste.

### 🤖 Fase 5: Modelagem, Validação e Overfitting
* **Modelagem Comparativa (Diferencial)**: Treinamento do modelo base de **Regressão Linear** comparado a um algoritmo não-linear de **Árvore de Decisão** (`DecisionTreeRegressor`).
* **Validação Cruzada**: Aplicação da metodologia K-Fold (5 partições) no conjunto de treino para avaliar a estabilidade dos estimadores.
* **Diagnóstico de Overfitting**: Análise comparativa do erro quadrático (RMSE) entre treino e teste para diagnosticar a capacidade de generalização dos modelos.

### 📈 Fase 6: Avaliação, Interpretação e Versionamento
* **Métricas Técnicas**: Cálculo do MAE, MSE, RMSE e R² para validação das hipóteses no conjunto de teste.
* **Análise Gráfica**: Geração e exportação do gráfico de dispersão entre valores reais vs. previstos e histograma de distribuição dos resíduos.
* **Versionamento Físico**: Salvamento do modelo campeão como `modelo_regressao_vl.pkl` e seu registro histórico de metadados em `metricas_vl.json` na pasta `models/v1/`.

---

## 📈 4. Resultados e Comparativo de Modelos

Abaixo estão apresentados os resultados consolidados das rodadas experimentais da versão **v1**:

| Modelo Preditivo | RMSE (Validação Cruzada) | MAE (Teste) | RMSE (Teste) | Coeficiente de Determinação (R² Teste) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regressão Linear** | US$ 201.311,16 | US$ 126.826,27 | US$ 212.672,62 | 0.6986 | Baseline |
| **Árvore de Decisão** | **US$ 163.892,41** | **US$ 103.450,12** | **US$ 177.951,20** | **0.7889** | **🏆 CAMPEÃO** |

### 🧠 Veredito e Interpretação de Negócios (Fase 6)
1. **Escolha do Campeão**: A Árvore de Decisão foi eleita a campeã por apresentar um **RMSE significativamente menor** no conjunto de teste (US$ 177.951,20 contra US$ 212.672,62 da Regressão Linear), servindo como critério de desempate exigido.
2. **Capacidade Explicativa**: O modelo campeão obteve um R² = 0.7889. Isso indica que cerca de **79% de toda a variabilidade de preços** de King County é explicada pelas variáveis físicas e temporais selecionadas.
3. **Aplicação Prática no Negócio**: O Erro Absoluto Médio (MAE) do modelo campeão indica que a inteligência artificial erra, em média, **US$ 103,4 mil** por imóvel precificado. Para a imobiliária de King County, essa margem de erro serve como um excelente triador automático de preços para carteiras de médio e alto padrão. Entretanto, para casas populares abaixo de US$ 200 mil, recomenda-se auditoria humana, pois o erro percentual relativo torna-se alto.

---

## 🗺️ 5. Rastreabilidade de Decisões Técnicas

Para demonstrar conformidade rigorosa com a organização metodológica, as decisões de engenharia e matemática do código estão explicadas e pareadas no notebook principal:

| Bloco do Notebook | Objetivo Prático | Justificativa Técnica Apresentada |
| :--- | :--- | :--- |
| **Fase 1** | EDA Estatística | Mapeamento detalhado de distribuições para identificar assimetria e tendências. |
| **Fase 2** | Limpeza de Dados | Escolha da **Mediana** robusta contra outliers e eliminação manual do outlier de 33 quartos. |
| **Fase 3** | Engenharia de Recursos | Criação de colunas temporais e exclusão de features baseadas em `price` para banir o *data leakage*. |
| **Fase 4** | Preparação | Exclusão de `sqft_above` para curar a multicolinearidade e split de dados antes da padronização. |
| **Fase 5** | Modelagem | Validação Cruzada (K-fold) para assegurar o diagnóstico de generalização e overfitting. |
| **Fase 6** | Avaliação | Tradução do erro estatístico (MAE) para impacto e risco financeiro de negócio. |

---

## 🚀 6. Como Executar o Sistema

Para reproduzir este experimento sem erros, siga o roteiro de execução abaixo:

### Passo 1: Clonar o Repositório
```bash
git clone [https://github.com/seu-usuario/projeto_final_v1.git](https://github.com/paulo-git-hub/projeto_final_v1.git)
cd projeto_final_v1
```

### Passo 2: Instalar Dependências (requirements.txt)
Instale todas as bibliotecas e bibliotecas de suporte com as versões exatas do ambiente original:
```bash
pip install -r requirements.txt
```

### Passo 3: Executar o Pipeline
Abra o notebook principal `/notebooks/projeto_final_v1.ipynb` no Google Colab ou em seu ambiente Jupyter local.
* Configure a variável `BASE_DIR` no topo do notebook com o seu caminho de diretório local ou do Google Drive.
* Execute todas as células de forma sequencial (`Run All`). O pipeline carregará os dados brutos, fará o tratamento, treinará os modelos, elegerá o campeão e salvará os arquivos de saída automaticamente.

---

## 🔮 7. Melhorias para Versões Futuras (v2, v3...)

Como boas práticas de ciclo de vida de modelos (MLOps), as versões subsequentes podem incluir:
* **Geolocalização Ativa**: Transformar as variáveis `lat` e `long` em clusters espaciais (usando K-Means) para capturar o real impacto da vizinhança sobre o preço dos imóveis.
* **Encodings Robustos**: Aplicar Target Encoding na variável `zipcode` baseando-se no preço médio regional, evitando criar dezenas de colunas esparsas.
* **Algoritmos Avançados**: Implementar estimadores de Gradiente Boosted (XGBoost, LightGBM) para elevar o teto preditivo do R² acima de 85%.

---

## 🎥 8. Apresentação do Projeto em Vídeo

A gravação técnica detalhada do projeto, abordando todos os questionamentos obrigatórios da Seção 5.4, está disponível no link abaixo:

* 🔗 **[Assista ao Vídeo Explicativo no Google Drive]** (Acesso em modo leitor para qualquer pessoa com o link).

*Nota: O vídeo possui duração máxima de 10 minutos, conta com a aparição do meu rosto em ambiente iluminado e não fez uso de qualquer ferramenta de criação de vídeo por inteligência artificial.*