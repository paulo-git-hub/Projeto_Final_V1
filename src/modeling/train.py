import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def treinar_e_avaliar_modelo(X_train, X_test, y_train, y_test):
    """
    Treina os modelos aplicando transformação logarítmica no preço (Target), 
    extrai as métricas e elege o campeão.
    """
    print("🤖 Iniciando treinamento (com Transformação Logarítmica no Preço)...")
    
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Árvore de Decisão": DecisionTreeRegressor(max_depth=6, random_state=42)
    }
    
    resultados = {}
    metricas_lista = []
    
    # 🌟 NOVIDADE: Transforma o y_train (preço) para a escala logarítmica
    y_train_log = np.log1p(y_train)
    
    for nome, modelo in modelos.items():
        print(f"  -> Treinando {nome}...")
        
        # 1. Treinamento na escala log
        modelo.fit(X_train, y_train_log)
        
        # 2. Predições (o modelo vai prever em escala log)
        y_pred_train_log = modelo.predict(X_train)
        y_pred_test_log = modelo.predict(X_test)
        
        # 🌟 NOVIDADE: Reverte a previsão (Exponencial) para voltar a ser Dólares reais
        y_pred_train = np.expm1(y_pred_train_log)
        y_pred_test = np.expm1(y_pred_test_log)
        
        # 3. Cálculo das Métricas (usando o valor em dólares reais contra a previsão em dólares)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)
        
	# 4. Cálculo da Variação Percentual do Erro (Teste vs Treino)
        variacao_pct = ((mae_test - mae_train) / mae_train) * 100
        diagnostico = f"{variacao_pct:+.2f}%"
        
        # Armazena os dados para a tabela
        metricas_lista.append({
            "Modelo": nome,
            "MAE (Treino)": f"$ {mae_train:,.2f}",
            "MAE (Teste)": f"$ {mae_test:,.2f}",
            "RMSE (Teste)": f"$ {rmse_test:,.2f}",
            "R² (Teste)": f"{r2_test:.4f}",
            "Variação do Erro": diagnostico  # ✔ Nome alterado para combinar com o percentual
        })
        
        resultados[nome] = {
            "modelo": modelo,
            "rmse": rmse_test,
            "pred": y_pred_test
        }
        
    df_metricas = pd.DataFrame(metricas_lista)
    nome_campeao = min(resultados, key=lambda k: resultados[k]["rmse"])
    campeao_info = resultados[nome_campeao]
    
    return campeao_info["pred"], nome_campeao, campeao_info["modelo"], df_metricas
        
    # 5. Criação do DataFrame dinâmico com os resultados
    df_metricas = pd.DataFrame(metricas_lista)
    
    # 6. Seleção Automática do Campeão
    nome_campeao = min(resultados, key=lambda k: resultados[k]["rmse"])
    campeao_info = resultados[nome_campeao]
    
    return campeao_info["pred"], nome_campeao, campeao_info["modelo"], df_metricas


def salvar_modelo(modelo, nome_modelo):
    """
    Salva o modelo campeão com o nome fixado pela arquitetura (.pkl).
    """
    from src.config import MODEL_DIR
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    caminho_completo = os.path.join(MODEL_DIR, "Projeto_Final_v1.pkl")
    
    joblib.dump(modelo, caminho_completo)
    print(f"💾 Modelo campeão ({nome_modelo}) salvo em: {caminho_completo}")
    
    return caminho_completo


def salvar_metricas(df_metricas):
    """
    Salva o dataframe de métricas no formato .json
    """
    from src.config import MODEL_DIR
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    caminho_completo = os.path.join(MODEL_DIR, "metricas_v1.json")
    
    # orient='records' salva como uma lista de objetos JSON bem formatada
    df_metricas.to_json(caminho_completo, orient='records', force_ascii=False, indent=4)
    print(f"📊 Métricas salvas com sucesso em: {caminho_completo}")
    
    return caminho_completo