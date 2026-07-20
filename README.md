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
Execução de estatística descritiva estruturada para compreender as dimensões, tipos primitivos e a distribuição geral do dataset original (21.613 linhas e 21 colunas).

* **Distribuição da Variável-Alvo:** O histograma de `price` revelou uma assimetria positiva acentuada (cauda longa à direita). Essa não-normalidade viola a suposição de normalidade dos resíduos em regressões lineares clássicas, justificando a aplicação futura da transformação logarítmica na variável-alvo para estabilizar a variância dos erros.
* **Comportamento das Variáveis Explicativas:** A área construída (`sqft_living`) confirmou uma forte relação linear positiva com o preço. Por outro lado, a qualidade da construção (`grade`) demonstrou uma relação não-linear exponencial (crescimento acentuado após a nota 8) e alta heterocedasticidade nos níveis mais elevados (imóveis de luxo).
* **Identificação de Multicolinearidade:** O mapa de calor de correlação de Pearson detectou uma colinearidade crítica entre as variáveis preditoras, destacando a forte correlação entre `sqft_living` e `sqft_above`.

### 2. Data Prep & Tratamento de Dados Ruidosos
Etapa destinada à higienização amostral para eliminar redundâncias e ruídos, mitigando o efeito *Garbage In, Garbage Out*:

* **Valores Ausentes:** Tratados via estratégia de imputação baseada na Mediana[cite: 1]. Como o mercado imobiliário apresenta forte assimetria, a mediana foi adotada como a escolha estatística correta por ser robusta a outliers.
* **Gerenciamento de Outliers:** Inspeções via boxplots identificaram e removeram anomalias críticas (como um registro inconsistente de um imóvel com 33 quartos), impedindo a distorção dos coeficientes dos modelos.
* **Eliminação de Multicolinearidade:** Foi calculado o Fator de Inflação da Variância (VIF). Para estabilizar os coeficientes e conter a alta dimensionalidade, foram eliminadas as colunas correlacionadas ou redundantes (`yr_built`, `yr_renovated`, `sqft_above`, `id`, `date`, `sqft_living15`, `sqft_lot15` e `sqft_basement`)[cite: 1]. A variável preditora principal (`sqft_living`) foi mantida com um VIF seguro e controlado de 5.10.

### 3. Feature Engineering
Concepção de novas colunas preditoras a partir de operações lógicas e matemáticas sobre os dados existentes:

* **`idade_imovel`:** Derivada da subtração entre o ano da venda e o ano de construção do ativo.
* **`foi_reformado`:** Variável binária (0 ou 1) que indica se o imóvel recebeu modificações ao longo dos anos.
* **Mitigação de *Data Leakage*:** Recursos calculados diretamente sobre o preço (como valor por pé quadrado) foram restritos à leitura de negócios na EDA e totalmente omitidos das matrizes preditoras para blindar o sistema contra o vazamento de dados.

### 4. Preparação para Modelagem e Validação
* **Split Amostral Segurado:** Divisão estável na proporção de 80% para treinamento (17.289 linhas) e 20% para teste (4.323 linhas).
* **Escalonamento Seguro:** Aplicação do `StandardScaler` utilizando `fit_transform` exclusivamente na matriz de treino e apenas `transform` na matriz de teste, evitando que estatísticas do ambiente de teste contaminem o aprendizado do modelo.
* **Estratégia de Validação:** Implementação de Validação Cruzada (*K-Fold* com 5 partições) confrontando os modelos com a aplicação da Transformação Logarítmica diretamente na variável-alvo (`price`) para garantir resíduos mais gaussianos.

---

## 📊 Desempenho e Diagnóstico dos Modelos

O pipeline confrontou o algoritmo base de Regressão Linear com o modelo não-linear de Árvore de Decisão:

### Diagnóstico de Generalização (Treino vs Teste)

| Modelo | MAE (Treino) | MAE (Teste) | RMSE (Teste) | R² (Teste) | Variação do Erro (MAE) | Diagnóstico Final |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Regressão Linear | US$ 111.440,72 | US$ 118.646,84 | US$ 320.097,49 | 0.3173 | +6,47% | Underfitting Crítico |
| **Árvore de Decisão** | US$ 98.255,44 | US$ 105.638,81 | US$ 187.092,03 | **0.7668** | +7,51% | **Modelo Campeão (Estável)** |

### 🔍 Diagnóstico Técnico:

* **1. Regressão Linear (Subajuste Crítico):**
  Apesar de apresentar uma variação de erro baixa entre treino e teste (+6,47%), o modelo falhou gravemente na capacidade preditiva, explicando apenas **31,73%** da variação dos preços no teste (`R² = 0.3173`)[cite: 1]. O modelo linear simples não foi capaz de processar as complexidades heterocedásticas e não-lineares capturadas na EDA (como a natureza da variável `grade`).
* **2. Árvore de Decisão (Modelo Campeão e Consistente):**
  Apresentou desempenho amplamente superior, alcançando um poder de explicação de **76,68%** sobre a variação dos preços dos imóveis no ambiente de teste (`R² = 0.7668`)[cite: 1]. Além disso, reduziu drasticamente o erro quadrático médio (RMSE) para US$ 187.092,03.
* **3. Controle de Overfitting:**
  Diferente de árvores de decisão convencionais que sofrem com sobreajuste, o pipeline controlado registrou uma variação no erro MAE de apenas **+7,51%** na transição do treino para o teste[cite: 1]. Essa proximidade milimétrica comprova que o modelo reteve um excelente poder de generalização para dados inéditos do mercado, mantendo-se perfeitamente estável.

---

## 💼 Interpretação e Veredito de Negócio

O desenvolvimento técnico atingiu maturidade estatística ao reverter a assimetria dos dados e neutralizar a multicolinearidade[cite: 1]. A validação prática do modelo aponta os seguintes impactos para a operação imobiliária:

* **Poder de Decisão:** O modelo campeão (Árvore de Decisão) consegue mapear e explicar mais de 76% do comportamento de preços de King County, gerando estimativas muito mais assertivas que os moldes lineares tradicionais.
* **Mitigação de Prejuízos:** O erro médio absoluto (MAE) de US$ 105.638,81 no teste atua como uma barreira de segurança importante para balizar propostas de compra, venda e análise de portfólio imobiliário, diminuindo drasticamente o risco de avaliações severamente equivocadas.
* **Persistência do Sistema:** O pipeline concluiu a automação salvando com sucesso o modelo vencedor e suas métricas diretamente em disco, tornando a inteligência artificial ativa e pronta para consumo imediato em produção ou deploy.

### 🚀 Próximos Passos:
Para expandir o poder de explicação (`R²`) acima de 85% e mitigar ainda mais o erro geral (RMSE), as próximas evoluções do repositório devem focar em:
1. Testar algoritmos de *Ensemble* mais robustos, como *Random Forest*, *Gradient Boosting* ou *XGBoost*.
2. Executar uma etapa focada de Otimização de Hiperparâmetros (*Hyperparameter Tuning* via GridSearch ou Optuna) sobre o modelo vencedor.

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