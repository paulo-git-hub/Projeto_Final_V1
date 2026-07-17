# Inteligência Artificial para Análise Preditiva Imobiliária (King County, EUA)

## 📋 Declaração do Problema de Negócio

No mercado imobiliário moderno, a precificação incorreta de ativos pode acarretar sérios prejuízos financeiros para imobiliárias, investidores e instituições financeiras. Estimar o valor de venda de um imóvel de forma empírica ou com base em suposições lineares simples frequentemente resulta em modelos enviesados que falham significativamente quando expostos a cenários reais de mercado.

**Objetivo do Projeto:** Desenvolver um pipeline de Machine Learning escalável e robusto capaz de prever o valor numérico contínuo (variável-alvo `price` em dólares) de imóveis localizados no condado de King County (EUA), com base em suas características físicas e geográficas.

**Impacto no Negócio:** Mitigar estimativas equivocadas, otimizar as margens do portfólio da imobiliária e apoiar decisões sequer de compra, venda ou concessão de crédito imobiliário.

---

## 🛠️ Estrutura do Projeto

O projeto foi estruturado seguindo as melhores práticas de desenvolvimento de software e ciência de dados, separando a lógica de execução (Notebook) dos submódulos reutilizáveis:

```text
projeto/
├── data/
│ ├── raw/ # Datasets brutos originais
│ ├── processed/ # Datasets limpos e tratados após o Data Prep
│ └── final/ # Recorte usado na modelagem
├── models/
│ └── v1/
│ ├── Projeto_Final_v1.pkl
│ └── metricas_v1.json
├── notebooks/ # notebook principal (.ipynb)
├── outputs/
│ └── figures/
├── src/ # modularização em .py
│ ├── __init__.py # torna src/ um pacote importável
│ ├── config.py # caminhos e parâmetros
│ ├── dataset.py # carga/salvamento dos dados
│ ├── features.py # limpeza + colunas derivadas
│ ├── plots.py # funções de visualização
│ └── modeling/
│ ├── __init__.py # torna modeling/ um subpacote
│ └── train.py # treino e avaliação
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

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

### Diagnóstico de Overfitting (Treino vs Teste)

| Modelo | MAE (Treino) | MAE (Teste) | RMSE (Teste) | R² (Teste) | Diagnóstico Final |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Regressão Linear | $ 125,444.52 | $ 127,456.21 | $ 212,493.32 | 0.6992 | Ajuste Estável / Boa generalização |
| Árvore de Decisão | $ 100,490.68 | $ 112,298.57 | $ 214,569.68 | 0.6932 | Overfitting (Erro sobe no teste) |

### 🔍 Diagnóstico Técnico:

* **1. Regressão Linear (Ajuste Estável)
Consistência do Erro (MAE):** O modelo apresenta uma variação muito pequena entre o erro no treinamento ($125,444.52) e o erro no teste ($127,456.21). A diferença é de menos de 2%.

* **Interpretação:** Essa proximidade indica que o modelo conseguiu aprender os padrões gerais (a tendência) sem memorizar os dados. Ele tem alta capacidade de generalização e estabilidade, mantendo o mesmo nível de precisão quando apresentado a imóveis que nunca viu antes.

* **Vantagem:** Possui baixa variância estatística, tornando-o um modelo mais seguro e previsível.

* **2. Árvore de Decisão (Overfitting)
Descolamento das Métricas:** Aqui, o erro no treinamento ($100,490.68) é ilusoriamente bom. Quando o modelo é testado com dados novos, o erro sobe mais de 11%, chegando a $112,298.57.

* **Interpretação:** Este é o sintoma clássico do overfitting (sobreajuste). A árvore criou "regras de divisão" tão específicas e profundas que acabou decorando as peculiaridades da base de treino em vez de aprender as regras reais do mercado. Quando recebe dados novos, ela se confunde e a margem de erro dispara.

* **Alerta de RMSE:** Apesar do MAE (erro médio) da Árvore ser menor que o da Regressão Linear, o RMSE dela no teste é o maior de todos ($214,569.68). Como a métrica RMSE penaliza fortemente erros grandes, isso prova que a Árvore de Decisão comete desvios individuais muito mais drásticos do que a Regressão Linear quando erra o preço de uma casa.

* **3. Métricas Globais e Limitações
Poder de Explicação (R²):** O R² está empatado na casa dos 0.69 para ambos. Isso significa que cerca de 69% da variação nos preços dos imóveis pode ser explicada pelas colunas numéricas selecionadas. É um bom valor de partida, mas insuficiente para precisão de mercado.

* **Teto de Desempenho (Underfitting):** Embora a Regressão Linear não sofra de overfitting, ambos os modelos esbarram em um alto viés (underfitting). Erros médios que ultrapassam a casa dos 120 mil dólares e picos de erro (RMSE) acima de 212 mil mostram que algoritmos simples não conseguem capturar toda a complexidade e não-linearidade do mercado imobiliário.

---

## 💼 Interpretação e Veredito de Negócio

Apesar do rigor estatístico aplicado no desenvolvimento, o diagnóstico financeiro indica que nenhum dos modelos está pronto para produção comercial:

* **Margem de Erro Inaceitável:** Um MAE de $127.456,21 significa que o modelo erra, em média, essa quantia por imóvel. Em uma propriedade padrão de $500.000,00, o erro representa mais de 25% do valor total do ativo.
* **Risco de Prejuízo:** Precificar um imóvel com essa margem de erro faria com que a imobiliária comprasse ativos supervalorizados ou vendesse seu próprio inventário muito abaixo do preço de mercado, gerando quebra de caixa.
* **Inviabilidade Bancária:** Para a concessão de crédito ou avaliação de garantias imobiliárias, um erro dessa magnitude é perigoso, pois expõe a instituição a um risco de inadimplência desalinhado com o valor real do colateral.

### 🚀 Próximos Passos:
Para elevar o $R^2$ acima de 85% e reduzir o MAE a níveis comercialmente aceitáveis, as próximas iterações do projeto devem focar em:

1. Aplicação de transformação logarítmica na variável-alvo para corrigir a assimetria positiva.
2. Utilização de algoritmos de *Ensemble* (como *Random Forest* ou *XGBoost*) para mitigar o *overfitting* da árvore isolada e capturar padrões não-lineares.
3. **Aperfeiçoamento do impacto geográfico:** Refinar o agrupamento por classes da variável `zipcode` realizado no Data Prep ou utilizar algoritmos de clusterização espacial (como o K-Means) diretamente nas variáveis `lat` e `long` para capturar com maior precisão as nuances de micro-localização e vizinhança do mercado imobiliário.

---

Instruções detalhadas para a execução do projeto.

Eu dividi as instruções em duas partes: Execução Local (ideal para quem clonar o GitHub) e Execução via Google Colab.

🚀 Como Executar o Projeto
Você pode reproduzir este pipeline de dados em sua máquina local ou executá-lo diretamente na nuvem. Siga o passo a passo correspondente ao seu ambiente de preferência:

📋 Pré-requisitos (Execução Local)
Certifique-se de ter instalado em sua máquina:

Python 3.8+

Git

Um ambiente de desenvolvimento com suporte a notebooks (VS Code, JupyterLab ou Jupyter Notebook).

💻 Opção 1: Execução em Máquina Local
1. Clone o repositório:
Abra o seu terminal e baixe o projeto para a sua máquina.

Bash
git clone https://github.com/paulo-git-hub/Projeto_Final_V1.git
cd Projeto_Final_V1

2. Crie e ative um ambiente virtual:
É uma boa prática isolar as bibliotecas do projeto do restante do seu sistema operacional.

Bash
# No Windows:
python -m venv venv
venv\Scripts\activate

# No Linux/Mac:
python3 -m venv venv
source venv/bin/activate
3. Instale as dependências:
O arquivo requirements.txt mapeia todas as bibliotecas e versões exatas utilizadas neste projeto (como Pandas, Scikit-Learn e Seaborn).

Bash
pip install -r requirements.txt
4. Execute o Notebook:
Abra a sua interface preferida ou inicie o Jupyter pelo terminal:

Bash
jupyter notebook notebooks/Projeto_Final_V1.ipynb
Dica de Execução: Rode as células sequencialmente, de cima para baixo. O pipeline foi arquitetado para fluir logicamente desde a extração dos dados brutos até a persistência do modelo em disco na pasta models/v1/.

☁️ Opção 2: Execução via Google Colab
Caso prefira rodar o sistema diretamente pelo navegador, sem instalar bibliotecas localmente:

Faça o download do repositório inteiro e faça o upload para uma pasta no seu Google Drive.

Abra o arquivo Projeto_Final_V1.ipynb utilizando o Google Colab.

Na primeira célula do notebook, autorize a montagem do disco (drive.mount) para que o Colab consiga enxergar os arquivos do projeto.

Ajuste a variável BASE_DIR na célula de configuração inicial para apontar para o caminho exato onde você salvou a pasta no seu Drive. 
Sugestão: BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/Projeto Final/Projeto_Final_V1"

Execute as células de forma sequencial, aguardando a geração automática dos gráficos na tela e o salvamento dos arquivos nas pastas locais virtuais do Drive.

> *Projeto desenvolvido como trabalho de conclusão do Módulo 1 do Curso Desenvolvimento de IA para Análise Preditiva - Carga Horária:150 horas