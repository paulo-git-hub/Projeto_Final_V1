
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def criar_novas_features(df):
    """
    Cria novas features a partir das colunas existentes do DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame com as novas features.
    """
    df_features = df.copy()

    # Assegura que 'date' esteja no formato datetime para cálculos
    if 'date' in df_features.columns:
        df_features['date'] = pd.to_datetime(df_features['date'])

    # Idade do imóvel
    if 'date' in df_features.columns and 'yr_built' in df_features.columns:
        df_features['idade_imovel'] = df_features['date'].dt.year - df_features['yr_built']
        # Trata idades negativas para casos onde o ano de construção é posterior ao da venda
        df_features['idade_imovel'] = df_features['idade_imovel'].apply(lambda x: max(x, 0)) # Garante que a idade não seja negativa

    # Se foi reformado (binário)
    if 'yr_renovated' in df_features.columns:
        df_features['foi_reformado'] = (df_features['yr_renovated'] > 0).astype(int)

    print("📐 Executando Feature Engineering (Fase 3)...")
    return df_features

def preparar_dados_modelagem(df):
    """
    Prepara o DataFrame para a modelagem:
    - Agrupa 'zipcode' em categorias mais amplas e aplica One-Hot Encoding.
    - Remove variáveis desnecessárias ou que causam multicolinearidade.
    - Divide os dados em conjuntos de treino e teste.
    - Aplica escalonamento nas features numéricas.

    Args:
        df (pd.DataFrame): DataFrame com as features criadas.

    Returns:
        tuple: X_train, X_test, y_train, y_test (DataFrames e Series).
    """
    print("⚙ Preparando dados para a modelagem (Fase 4 - Com agrupamento de zipcode)...")
    df_processed = df.copy()

    # Remover variáveis não necessárias ou que causam multicolinearidade
    # 'id' e 'date' são identificadores/temporais que não serão usados diretamente no modelo linear
    # 'sqft_above' é removido devido à alta multicolinearidade com 'sqft_living'
    columns_to_drop_initial = ['id', 'date', 'sqft_above']
    for col in columns_to_drop_initial:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])

    # Feature Engineering para zipcode: agrupamento
    if 'zipcode' in df_processed.columns:
        # Cria uma nova feature de grupo baseada nos 3 primeiros dígitos do zipcode
        # Isso reduz a cardinalidade e agrupa regiões próximas
        df_processed['zipcode_group'] = df_processed['zipcode'].astype(str).str[:3]
        # Remove a coluna zipcode original
        df_processed = df_processed.drop(columns=['zipcode'])
    else:
        print("Aviso: 'zipcode' não encontrado no DataFrame para agrupamento. Ignorando...")

    # Separar features (X) e target (y)
    X = df_processed.drop(columns=['price'], errors='ignore')
    y = df_processed['price']

    # Identificar colunas numéricas e categóricas para pré-processamento
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Criação do pré-processador: escalonamento para numéricas e One-Hot Encoding para categóricas
    # handle_unknown='ignore' para evitar erros se novas categorias aparecerem no teste
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough' # Mantém outras colunas (se houver, como booleanas que não foram incluídas em num/cat)
    )

    # Divisão em treino e teste
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplica o pré-processamento (escalonamento + one-hot encoding)
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    # Recupera os nomes das colunas após One-Hot Encoding
    ohe_feature_names = []
    # Itera sobre os transformadores do ColumnTransformer
    for name, transformer, features in preprocessor.transformers_:
        if name == 'cat' and transformer is not None: # Verifica se é o transformador 'cat' e se não é None (ou seja, foi aplicado)
            if hasattr(transformer, 'get_feature_names_out'):
                # Usar get_feature_names_out que é mais robusto
                ohe_feature_names = list(transformer.get_feature_names_out(categorical_cols))
            else:
                # Fallback para get_feature_names em versões mais antigas
                ohe_feature_names = list(transformer.get_feature_names(categorical_cols))
            break # Sai do loop depois de encontrar o OneHotEncoder

    # Concatena os nomes das colunas numéricas e as novas colunas one-hot encoded
    all_feature_names = list(numerical_cols) + ohe_feature_names

    X_train = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test_raw.index)

    return X_train, X_test, y_train, y_test
