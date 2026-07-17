import os
import json
import pickle
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from src.config import MODEL_DIR

def treinar_e_avaliar_modelo(X_train, X_test, y_train, y_test, Versao='V1'):
    """
    Treina, valida e salva os modelos de regressão, aplicando a reversão da escala logarítmica
    para as métricas de negócio e isolando os artefatos de saída de acordo com a versão informada.
    """
    print(f"🚀 Iniciando Modelagem e Diagnósticos (Fases 5 e 6 - Versão {Versao.upper()})...")
    
    # Inicialização dos modelos para comparação (V2)
    modelos = {
        'LinearRegression': LinearRegression(),
        'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    
    resultados = {}
    modelo_campeao = None
    melhor_rmse = float('inf')
    nome_campeao = ""
    y_pred_campeao = None
    
    for nome, modelo in modelos.items():
        print("\n Treinando {}".format(nome))
        
        # Validação Cruzada (K-Fold)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_cv_scores = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            modelo.fit(X_train_fold, y_train_fold)
            y_pred_val = modelo.predict(X_val_fold)
            
            # REVERSÃO DO LOG NO FOLD DA CV: Transforma predições de volta para a escala original (dólares)
            y_val_fold_real = np.expm1(y_val_fold)
            y_pred_val_real = np.expm1(y_pred_val)
            
            rmse_cv_scores.append(np.sqrt(mean_squared_error(y_val_fold_real, y_pred_val_real)))
            
        rmse_cv_medio = np.mean(rmse_cv_scores)
        print("RMSE Médio CV ({:}): ${:,.2f}".format(nome, rmse_cv_medio))
        
        # Treinar o modelo final com todo o conjunto de treino
        modelo.fit(X_train, y_train)
        
        # Predições (ainda na escala log)
        y_pred_tr = modelo.predict(X_train)
        y_pred_te = modelo.predict(X_test)
        
        # REVERSÃO DO LOG DO MODELO FINAL: Traz dados reais e predições de volta para dólares
        y_train_real = np.expm1(y_train)
        y_pred_tr_real = np.expm1(y_pred_tr)
        y_test_real = np.expm1(y_test)
        y_pred_te_real = np.expm1(y_pred_te)
        
        # Métricas de avaliação calculadas na escala real em dólares
        mae_tr = mean_absolute_error(y_train_real, y_pred_tr_real)
        rmse_tr = np.sqrt(mean_squared_error(y_train_real, y_pred_tr_real))
        
        mae_te = mean_absolute_error(y_test_real, y_pred_te_real)
        mse_te = mean_squared_error(y_test_real, y_pred_te_real)
        rmse_te = np.sqrt(mse_te)
        r2_te = r2_score(y_test_real, y_pred_te_real)
        
        resultados[nome] = {
            'MAE_treino': mae_tr,
            'RMSE_treino': rmse_tr,
            'MAE_teste': mae_te,
            'RMSE_teste': rmse_te,
            'R2_teste': r2_te,
            'RMSE_CV_Medio': rmse_cv_medio,
            'modelo': modelo
        }
        
        print("Treino - MAE: ${:,.2f} | RMSE: ${:,.2f}".format(mae_tr, rmse_tr))
        print("Teste - MAE: ${:,.2f} | RMSE: ${:,.2f} | R²: {:.4f}".format(mae_te, rmse_te, r2_te))
        
        # Seleção do modelo campeão pelo RMSE no conjunto de teste (em dólares)
        if rmse_te < melhor_rmse:
            melhor_rmse = rmse_te
            modelo_campeao = modelo
            nome_campeao = nome
            y_pred_campeao = y_pred_te_real  # Retorna as predições na escala original de mercado
            
    print("\n=== COMPARATIVO DE DESEMPENHO FINAL ===")
    for nome, res in resultados.items():
        print("Modelo: {} RMSE Teste: ${:,.2f} | R² Teste: {:.4f}".format(nome, res['RMSE_teste'], res['R2_teste']))
        
    # === Salvando o modelo de forma inteligente e dinâmica ===
    # Cria uma pasta dedicada (ex: outputs/models/V1 ou outputs/models/V2) com base no notebook
    pasta_salvamento = os.path.join(MODEL_DIR, Versao)
    os.makedirs(pasta_salvamento, exist_ok=True)
    
    nome_arquivos = nome_campeao.lower().replace(' ', '_')
    model_filename = f"modelo_regressao_{nome_arquivos}_{Versao}.pkl"
    metadata_filename = f"metricas_{nome_arquivos}_{Versao}.json"
    
    with open(os.path.join(pasta_salvamento, model_filename), "wb") as f:
        pickle.dump(modelo_campeao, f)
        
    metadados = {
        "modelo_id": nome_campeao,
        "data_treinamento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "variaveis_preditoras": list(X_train.columns),
    }
    
    # Criar uma cópia das métricas para evitar incluir o objeto modelo no JSON
    metricas_para_json = resultados[nome_campeao].copy()
    metricas_para_json.pop('modelo', None)
    metadados["metricas_validacao"] = metricas_para_json
    
    with open(os.path.join(pasta_salvamento, metadata_filename), "w", encoding="utf-8") as f:
        json.dump(metadados, f, indent=4, ensure_ascii=False)
        
    return y_pred_campeao, nome_campeao, modelo_campeao
