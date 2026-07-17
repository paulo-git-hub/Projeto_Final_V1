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
    Treina os modelos, extrai as métricas de treino e teste, 
    gera uma tabela de diagnóstico de overfitting e elege o campeão.
    """
    print("🤖 Iniciando treinamento comparativo de modelos...")
    
    modelos = {
        "Regressão Linear": LinearRegression(),
        "Árvore de Decisão": DecisionTreeRegressor(max_depth=6, random_state=42)
    }
    
    resultados = {}
    metricas_lista = []
    
    for nome, modelo in modelos.items():
        print(f"  -> Treinando {nome}...")
        
        # 1. Treinamento
        modelo.fit(X_train, y_train)
        
        # 2. Predições (Treino e Teste)
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        
        # 3. Cálculo das Métricas
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)
        
        # 4. Lógica de Diagnóstico Automático
        if mae_test > (mae_train * 1.05):
            diagnostico = "Overfitting (Erro sobe no teste)"
        else:
            diagnostico = "Ajuste Estável / Boa generalização"
            
        metricas_lista.append({
            "Modelo": nome,
            "MAE (Treino)": f"$ {mae_train:,.2f}",
            "MAE (Teste)": f"$ {mae_test:,.2f}",
            "RMSE (Teste)": f"$ {rmse_test:,.2f}",
            "R² (Teste)": f"{r2_test:.4f}",
            "Diagnóstico Final": diagnostico
        })
        
        resultados[nome] = {
            "modelo": modelo,
            "rmse": rmse_test,
            "pred": y_pred_test
        }
        
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