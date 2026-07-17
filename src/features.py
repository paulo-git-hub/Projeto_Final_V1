import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

def criar_novas_features(df):
    """
    Cria novas variáveis (Feature Engineering) para melhorar a performance do modelo.
    """
    print("💡 Iniciando a criação de novas features (Engenharia de Recursos)...")
    df_features = df.copy()
    
    # 1. Criação da 'idade_imovel'
    # Utilizando 2015 (ano final do dataset) como referência para evitar idades negativas
    if 'yr_built' in df_features.columns:
        df_features['idade_imovel'] = 2015 - df_features['yr_built']
        
    # 2. Criação da variável binária 'foi_reformado'
    # Se o ano de renovação for maior que 0, recebe 1 (Sim), caso contrário, 0 (Não)
    if 'yr_renovated' in df_features.columns:
        df_features['foi_reformado'] = df_features['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
        
    print("  -> Features 'idade_imovel' e 'foi_reformado' criadas com sucesso!")
    
    return df_features

def calcular_vif(df, colunas=None):
    """
    Calcula o Fator de Inflação da Variância (VIF) para as variáveis selecionadas.
    Variáveis com VIF > 5 (ou 10, a depender do critério) devem ser analisadas
    para remoção ou combinação.
    """
    print("📊 Calculando o Fator de Inflação da Variância (VIF)...")
    
    # Se nenhuma lista de colunas for fornecida, analisa todas as numéricas (exceto o preço)
    if colunas is None:
        colunas = df.select_dtypes(include=['number']).columns.tolist()
        if 'price' in colunas:
            colunas.remove('price')
            
    # Garante que usaremos apenas colunas que realmente existem no DataFrame
    colunas_presentes = [col for col in colunas if col in df.columns]
    
    X = df[colunas_presentes].copy()
    
    # Adiciona a constante (intercepto), obrigatória para o cálculo correto do VIF
    X['intercept'] = 1
    
    # Executa o cálculo para cada coluna
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Remove o intercepto do resultado visual e ordena de forma decrescente
    vif_data = vif_data[vif_data['Variável'] != 'intercept'].sort_values(by="VIF", ascending=False)
    
    return vif_data.round(2)

def preparar_dados_modelagem(df, target_col='price'):
    """
    Realiza a remoção de colunas multicolineares, separa variáveis previsoras e alvo,
    divide em treino/teste e aplica o escalonamento seguro.
    """
    print("⚙️ Preparando dados para modelagem...")
    df_model = df.copy()

    # 1. Eliminação de multicolinearidade
    colunas_remover = ['yr_built', 'yr_renovated']
    colunas_presentes = [c for c in colunas_remover if c in df_model.columns]
    if colunas_presentes:
        df_model = df_model.drop(columns=colunas_presentes)
        print(f"  -> Colunas removidas (multicolinearidade): {colunas_presentes}")

    # 2. SELEÇÃO SEGURA: Seleciona apenas colunas numéricas para o X
    # Isso impede que colunas de texto (como datas) quebrem o StandardScaler
    df_numerico = df_model.select_dtypes(include=['number'])
    
    if target_col not in df_numerico.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada ou não é numérica.")
        
    X = df_numerico.drop(columns=[target_col])
    y = df_numerico[target_col]

    # 3. Split Amostral (80% Treino / 20% Teste)
    # random_state=42 garante que a divisão será sempre a mesma toda vez que você rodar o código
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  -> Split Amostral: 80% Treino ({X_train.shape[0]} linhas), 20% Teste ({X_test.shape[0]} linhas).")

    # 4. Escalonamento Seguro
    scaler = StandardScaler()
    
    # IMPORTANTE: O 'fit' (aprender a escala) é feito APENAS nos dados de treino!
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    
    # Nos dados de teste, usamos apenas o 'transform', aplicando a escala aprendida do treino
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    print("  -> Escalonamento seguro (StandardScaler) aplicado com sucesso.")

    return X_train_scaled, X_test_scaled, y_train, y_test
