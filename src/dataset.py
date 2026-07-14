import pandas as pd
import numpy as np
from src.config import DATA_RAW, DATA_PROCESSED

def carregar_dados():
    print("📥 Carregando dados brutos de King County...")
    return pd.read_csv(DATA_RAW)

def obter_estatisticas_basicas(df):
    return df.shape, df.dtypes, df.describe()

def limpar_dados(df):
    print("🧹 Iniciando limpeza de dados (Fase 2)...")
    df_clean = df.copy()
    
    # Tratamento de Duplicados
    duplicados = df_clean.duplicated().sum()
    if duplicados > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"   -> Remoção de {duplicados} registros duplicados.")
    else:
        print("   -> Nenhum registro duplicado detectado.")
        
    # Tratamento de Nulos por Mediana
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            mediana = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(mediana)
            print(f"   -> Imputação em '{col}' com mediana: {mediana}")
            
    # Remoção de Outliers Críticos (Ex: imóvel com 33 quartos)
    df_clean = df_clean[df_clean['bedrooms'] < 20]
    print("   -> Outlier de 33 quartos tratado.")
    
    df_clean.to_csv(DATA_PROCESSED, index=False)
    print(f"💾 Dados limpos salvos em: {DATA_PROCESSED}")
    return df_clean
