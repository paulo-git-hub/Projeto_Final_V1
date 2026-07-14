import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DATA_FINAL, RANDOM_STATE, TEST_SIZE

def criar_novas_features(df):
    print("📐 Executando Feature Engineering (Fase 3)...")
    df_feat = df.copy()
    df_feat['ano_venda'] = pd.to_datetime(df_feat['date']).dt.year
    df_feat['idade_imovel'] = (df_feat['ano_venda'] - df_feat['yr_built']).clip(lower=0)
    df_feat['foi_reformado'] = (df_feat['yr_renovated'] > 0).astype(int)
    return df_feat.drop(columns=['ano_venda'])

def preparar_dados_modelagem(df):
    print("⚙ Preparando dados para a modelagem (Fase 4)...")
    df_prep = df.copy()
    colunas_remover = ['id', 'date', 'sqft_above', 'zipcode']
    df_prep = df_prep.drop(columns=colunas_remover, errors='ignore')
    
    X = df_prep.drop(columns=['price'])
    y = df_prep['price']
    df_prep.to_csv(DATA_FINAL, index=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test
