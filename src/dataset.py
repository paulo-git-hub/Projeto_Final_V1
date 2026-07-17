import pandas as pd
from src.config import DATA_RAW
import numpy as np
from src.config import DATA_PROCESSED

def carregar_dados():
    """
    Carrega o dataset bruto do projeto usando o caminho definido no config.py.
    """
    return pd.read_csv(DATA_RAW)

def obter_estatisticas_basicas(df):
    """
    Extrai as dimensões, os tipos de dados e o resumo estatístico descritivo.
    
    Retorno:
        tuple: (shape, dtypes, describe)
    """
    dim = df.shape
    tipos = df.dtypes
    # include='all' garante que variáveis categóricas também apareçam no resumo, se existirem
    resumo = df.describe(include='all') 
    
    return dim, tipos, resumo

def limpar_dados(df):
    """Executa a rotina de limpeza, imputação e remoção de redundâncias."""
    print("🧹 Iniciando limpeza de dados (Fase 2)...")
    df_clean = df.drop_duplicates()
    
    colunas_numericas = df_clean.select_dtypes(include=[np.number]).columns
    for col in colunas_numericas:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
    if 'bedrooms' in df_clean.columns:
        df_clean = df_clean[df_clean['bedrooms'] < 20]
        
    colunas_remover = [c for c in ['id', 'date','sqft_above', 'sqft_living15', 'sqft_lot15'] if c in df_clean.columns]
    df_clean = df_clean.drop(columns=colunas_remover)
    
    df_clean.to_csv(DATA_PROCESSED, index=False)
    print(f"💾 Dados limpos salvos em: {DATA_PROCESSED}")
    return df_clean