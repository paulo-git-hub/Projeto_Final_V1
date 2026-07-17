dataset.py

import pandas as pd
import numpy as np
from src.config import DATA_RAW, DATA_PROCESSED

def carregar_dados():
    return pd.read_csv(DATA_RAW)

def obter_estatisticas_basicas(df):
    return df.shape, df.dtypes, df.describe()

def limpar_dados(df):
    print("🧹 Iniciando limpeza de dados (Fase 2)...")
    df_clean = df.copy()
    
    # 1. Tratamento de Duplicados
    duplicados = df_clean.duplicated().sum()
    if duplicados > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"   -> Remoção de {duplicados} registros duplicados.")
    else:
        print("   -> Nenhum registro duplicado detectado.")
        
    # 2. Tratamento de Nulos por Mediana (Apenas colunas numéricas)
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isnull().any():
            mediana = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(mediana)
            print(f"   -> Imputação em '{col}' com mediana: {mediana}")
            
    # 3. Remoção de Outliers Críticos (Ex: imóvel com 33 quartos)
    if 'bedrooms' in df_clean.columns:
        linhas_antes = df_clean.shape[0]
        df_clean = df_clean[df_clean['bedrooms'] < 20]
        linhas_removidas = linhas_antes - df_clean.shape[0]
        print(f"   -> Outlier de quartos tratado ({linhas_removidas} registros removidos).")
    
    # 4. Tratamento de Multicolinearidade e IDs Irrelevantes
    colunas_remover = ['id', 'sqft_above', 'sqft_living15', 'sqft_lot15']
    # Mantém apenas as colunas que realmente existem no DataFrame para evitar erros
    colunas_presentes = [col for col in colunas_remover if col in df_clean.columns]
    
    if colunas_presentes:
        df_clean = df_clean.drop(columns=colunas_presentes)
        print(f"   -> Remoção de colunas multicolinares/irrelevantes: {colunas_presentes}")

    # Salvar dados processados
    df_clean.to_csv(DATA_PROCESSED, index=False)
    print(f"💾 Dados limpos salvos em: {DATA_PROCESSED}")
    return df_clean
-------------
plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from src.config import OUTPUT_FIGURES_DIR

def plotar_analise_exploratoria(df):
    """
    Fase 1: Gera e salva as visualizações obrigatórias.
    Garante a criação automática das pastas para evitar FileNotFoundError.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histograma do Preço (Variável-Alvo)
    sns.histplot(df['price'], kde=True, color='teal', ax=axes[0, 0])
    axes[0, 0].set_title("Distribuição de Preço (Assimetria Positiva)")
    axes[0, 0].set_xlabel("Preço (USD)")
    
    # 2. Dispersão 1: sqft_living vs Preço
    sns.scatterplot(data=df, x='sqft_living', y='price', alpha=0.3, color='orange', ax=axes[0, 1])
    axes[0, 1].set_title("Área Construída (sqft_living) vs Preço")
    axes[0, 1].set_xlabel("Área (sqft)")
    
    # 3. Dispersão 2: grade vs Preço (Segunda explicativa)
    sns.scatterplot(data=df, x='grade', y='price', alpha=0.3, color='green', ax=axes[1, 0])
    axes[1, 0].set_title("Avaliação do Imóvel (grade) vs Preço")
    axes[1, 0].set_xlabel("Classe de Construção (Grade)")
    
    # 4. Heatmap de Pearson
    cols_num = df.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore')
    sns.heatmap(cols_num.corr(), cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1, 1], cbar=True)
    axes[1, 1].set_title("Matriz de Correlação de Pearson")
    
    plt.tight_layout()
    
    # Garante que a pasta de destino exista antes de tentar salvar
    os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
    
    # Salvando em ambos os locais por segurança (caso o seu código orquestrador chame caminhos diferentes)
    caminho_eda = os.path.join(OUTPUT_FIGURES_DIR, "fase1_eda.png")
    plt.savefig(caminho_eda, dpi=150, bbox_inches='tight')
    
    # Backup para caminhos antigos/referência
    backup_dir = os.path.join(os.path.dirname(OUTPUT_FIGURES_DIR), "../reports/figures")
    os.makedirs(backup_dir, exist_ok=True)
    plt.savefig(os.path.join(backup_dir, "distribuicao_precos.png"), dpi=150, bbox_inches='tight')
    
    plt.close()
    print("📈 Gráficos de EDA gerados e salvos com sucesso!")

def plotar_boxplots_outliers(df):
    """
    Fase 2: Gera os boxplots para justificar o tratamento de outliers.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.boxplot(data=df, x='bedrooms', color='coral', ax=axes[0])
    axes[0].set_title("Outliers: Número de Quartos")
    
    sns.boxplot(data=df, x='sqft_living', color='skyblue', ax=axes[1])
    axes[1].set_title("Outliers: Área Construída")
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, "fase2_outliers_boxplots.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Boxplots de Outliers salvos!")

def plotar_avaliacao_modelo(y_real, y_pred):
    """
    Fase 6: Gráficos de análise residual e dispersão real vs previsto.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.scatterplot(x=y_real, y=y_pred, alpha=0.4, color='b', ax=axes[0])
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], '--r', lw=2)
    axes[0].set_title("Valores Reais vs. Previstos")
    
    residuos = y_real - y_pred
    sns.histplot(residuos, kde=True, color='purple', ax=axes[1])
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_title("Distribuição dos Resíduos (Erros)")
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, "fase6_avaliacao.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Gráficos de Avaliação da Fase 6 salvos!")

