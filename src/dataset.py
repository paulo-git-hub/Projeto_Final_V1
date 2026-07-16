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

