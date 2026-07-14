import os, json, pickle, numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import MODEL_DIR

def treinar_e_avaliar_modelo(X_train, X_test, y_train, y_test):
    print("🤖 Iniciando Modelagem e Diagnósticos (Fases 5 e 6)...")
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    y_pred_tr = modelo.predict(X_train)
    y_pred_te = modelo.predict(X_test)
    
    mae_tr = mean_absolute_error(y_train, y_pred_tr)
    rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
    
    mae_te = mean_absolute_error(y_test, y_pred_te)
    mse_te = mean_squared_error(y_test, y_pred_te)
    rmse_te = np.sqrt(mse_te)
    r2_te = r2_score(y_test, y_pred_te)
    
    print("\n=== COMPARATIVO DE DESEMPENHO ===")
    print(f"Treino - MAE: ${mae_tr:,.2f} | RMSE: ${rmse_tr:,.2f}")
    print(f"Teste  - MAE: ${mae_te:,.2f} | RMSE: ${rmse_te:,.2f} | R²: {r2_te:.4f}")
    print("===================================\n")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "modelo_regressao_vl.pkl"), "wb") as f:
        pickle.dump(modelo, f)
        
    metadados = {
        "modelo_id": "modelo_regressao_v1",
        "data_treinamento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "variaveis_preditoras": list(X_train.columns),
        "metricas_validacao": {
            "MAE": round(mae_te, 2),
            "MSE": round(mse_te, 2),
            "RMSE": round(rmse_te, 2),
            "R2": round(r2_te, 4)
        }
    }
    with open(os.path.join(MODEL_DIR, "metricas_vl.json"), "w", encoding="utf-8") as f:
        json.dump(metadados, f, indent=4, ensure_ascii=False)
    return y_pred_te
