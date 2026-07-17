import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import OUTPUT_FIGURES_DIR
import numpy as np

def plotar_histograma_preco(df, coluna_preco='price'):
    """
    Gera, exibe no notebook e salva o histograma de distribuição de preços.
    
    Parâmetros:
        df (pd.DataFrame): O dataset contendo os dados.
        coluna_preco (str): O nome da coluna alvo. Padrão é 'price'.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Avaliando a assimetria da distribuição
    sns.histplot(df[coluna_preco], kde=True, ax=ax, bins=50, color='skyblue')
    
    ax.set_title('Distribuição do Preço dos Imóveis', fontsize=16)
    ax.set_xlabel('Preço ($)', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    # 1. SALVAR A FIGURA PRIMEIRO
    hist_price_path = os.path.join(OUTPUT_FIGURES_DIR, 'price_distribution.png')
    fig.savefig(hist_price_path)
    
    # 2. MOSTRAR O GRÁFICO NO NOTEBOOK
    plt.show()
    
    # 3. LIBERAR MEMÓRIA
    plt.close(fig)

def plotar_analise_exploratoria(df, coluna_alvo='price'):
    """
    Gera, exibe e salva os gráficos da Análise Exploratória Inicial.
    Atualmente inclui: Histograma de distribuição da variável-alvo.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plotagem
    sns.histplot(df[coluna_alvo], kde=True, ax=ax, bins=50, color='skyblue')
    
    # 2. Estilização
    ax.set_title('Distribuição do Preço dos Imóveis', fontsize=16)
    ax.set_xlabel('Preço ($)', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    
    # 3. Salvamento Automático
    caminho_arquivo = os.path.join(OUTPUT_FIGURES_DIR, 'price_distribution.png')
    fig.savefig(caminho_arquivo)
    
    # 4. Exibição e Limpeza de Memória
    plt.show()
    plt.close(fig)

def plotar_dispersao_area_preco(df):
    """
    Gera, exibe e salva o gráfico de dispersão avaliando a relação 
    linear entre a área construída (sqft_living) e o preço (price).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plotagem
    sns.scatterplot(x='sqft_living', y='price', data=df, ax=ax, alpha=0.6, color='coral')
    
    # 2. Estilização
    ax.set_title('Relação entre Área Construída (sqft_living) e Preço (price)', fontsize=16)
    ax.set_xlabel('Área Construída (sqft_living)', fontsize=12)
    ax.set_ylabel('Preço ($)', fontsize=12)
    plt.grid(alpha=0.75)
    plt.tight_layout()
    
    # 3. Salvamento Automático (SEMPRE ANTES DO SHOW)
    caminho_arquivo = os.path.join(OUTPUT_FIGURES_DIR, 'sqft_living_vs_price.png')
    fig.savefig(caminho_arquivo)
    
    # 4. Exibição e Limpeza de Memória
    plt.show()
    plt.close(fig)

def plotar_dispersao_qualidade_preco(df):
    """
    Gera, exibe e salva o gráfico de dispersão avaliando a relação 
    entre a qualidade da construção (grade) e o preço (price).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plotagem
    sns.scatterplot(x='grade', y='price', data=df, ax=ax, alpha=0.6, color='lightgreen')
    
    # 2. Estilização
    ax.set_title('Relação entre Qualidade da Construção (grade) e Preço (Price)', fontsize=16)
    ax.set_xlabel('Qualidade (grade)', fontsize=12)
    ax.set_ylabel('Preço ($)', fontsize=12)
    plt.grid(alpha=0.75)
    plt.tight_layout()
    
    # 3. Salvamento Automático (SEMPRE ANTES DO SHOW)
    caminho_arquivo = os.path.join(OUTPUT_FIGURES_DIR, 'grade_vs_price.png')
    fig.savefig(caminho_arquivo)
    
    # 4. Exibição e Limpeza de Memória
    plt.show()
    plt.close(fig)

def plotar_mapa_correlacao(df):
    """
    Gera, exibe e salva o mapa de calor da correlação de Pearson 
    para identificar multicolinearidade entre as variáveis numéricas.
    """
    # 1. Preparação dos dados
    colunas_numericas = df.select_dtypes(include=['number']).columns
    matriz_correlacao = df[colunas_numericas].corr()

    # 2. Plotagem e Estilização
    fig, ax = plt.subplots(figsize=(16, 14)) 
    sns.heatmap(
        matriz_correlacao, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=.5, 
        ax=ax, 
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_title('Mapa de Calor da Correlação de Pearson entre Variáveis Numéricas', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 3. Salvamento Automático (ANTES DO SHOW)
    caminho_arquivo = os.path.join(OUTPUT_FIGURES_DIR, 'correlation_heatmap.png')
    fig.savefig(caminho_arquivo)

    # 4. Exibição e Limpeza de Memória
    plt.show()
    plt.close(fig)

def plotar_avaliacao_modelo(y_true, y_pred):
    """Esqueleto para a futura função."""
    pass

def plotar_boxplots_outliers(df):
    """
    Gera, exibe e salva os boxplots para justificar o tratamento de outliers 
    nas variáveis de quartos e área construída.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Plotagem dos Boxplots
    sns.boxplot(data=df, x='bedrooms', color='coral', ax=axes[0])
    axes[0].set_title("Outliers: Número de Quartos")
    
    sns.boxplot(data=df, x='sqft_living', color='skyblue', ax=axes[1])
    axes[1].set_title("Outliers: Área Construída")
    
    plt.tight_layout()
    
    # 2. Salvamento Automático
    caminho_arquivo = os.path.join(OUTPUT_FIGURES_DIR, "fase2_outliers_boxplots.png")
    fig.savefig(caminho_arquivo, dpi=150, bbox_inches='tight')
    
    # 3. Exibição e Limpeza de Memória
    print("📈 Boxplots de Outliers salvos com sucesso!")
    plt.show()
    plt.close(fig)

def plotar_avaliacao_modelo(y_true, y_pred):
    """
    Gera, exibe e salva os plots de avaliação do modelo campeão:
    1. Gráfico de dispersão preditiva (Valores Reais vs. Preditos)
    2. Gráfico de análise residual (Resíduos vs. Valores Preditos)
    """
    print("📈 Gerando gráficos de avaliação de performance do modelo...")
    
    # Cálculo dos resíduos (Erro = Valor Real - Valor Predito)
    residuos = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- 1. DISPERSÃO PREDITIVA ---
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[0], alpha=0.5, color='purple')
    
    # Linha diagonal de referência (Y = X) para indicar previsão perfeita
    limite_min = min(y_true.min(), y_pred.min())
    limite_max = max(y_true.max(), y_pred.max())
    axes[0].plot([limite_min, limite_max], [limite_min, limite_max], color='red', linestyle='--', lw=2)
    
    axes[0].set_title('Dispersão Preditiva (Real vs. Predito)', fontsize=14)
    axes[0].set_xlabel('Preço Real ($)', fontsize=11)
    axes[0].set_ylabel('Preço Predito ($)', fontsize=11)
    axes[0].grid(True, alpha=0.5)
    
    # --- 2. ANÁLISE RESIDUAL ---
    sns.scatterplot(x=y_pred, y=residuos, ax=axes[1], alpha=0.5, color='teal')
    
    # Linha horizontal de referência em zero (erro zero)
    axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
    
    axes[1].set_title('Análise Residual (Predito vs. Erro)', fontsize=14)
    axes[1].set_xlabel('Preço Predito ($)', fontsize=11)
    axes[1].set_ylabel('Resíduos / Erro ($)', fontsize=11)
    axes[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    
    # Salvamento Automático (ANTES DO SHOW)
    caminho_arquivo = os.path.join(OUTPUT_FIGURES_DIR, "performance_modelo_campeao.png")
    fig.savefig(caminho_arquivo, dpi=150, bbox_inches='tight')
    
    # Exibição e Limpeza de Memória
    plt.show()
    plt.close(fig)
    print("💾 Gráficos de validação salvos em outputs/figures!")
