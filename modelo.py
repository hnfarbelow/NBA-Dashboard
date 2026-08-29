import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def treinar_modelo():
    df = pd.read_csv('dataset_ml_com_posse.csv')
    
    #dividir por jogo
    game_ids = df['game_id_str'].unique()

    np.random.seed(42)
    np.random.shuffle(game_ids)
    
    #treino/teste
    split_idx = int(len(game_ids) * 0.8)
    train_ids = game_ids[:split_idx]
    test_ids = game_ids[split_idx:]
    
    train_df = df[df['game_id_str'].isin(train_ids)]
    test_df = df[df['game_id_str'].isin(test_ids)]
    
    #Variáveis
    features = ['seconds_remaining', 'score_margin_home', 'home_has_possession']
    target = 'home_team_won'
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"Treino: {len(train_df)} linhas | Teste: {len(test_df)} linhas")


    #modelo
    modelo = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,      
        max_depth=5,           
        learning_rate=0.05,    
        subsample=0.8,         
        colsample_bytree=0.8, 
        random_state=42
    )

    modelo.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

   
    #avaliando
    probs = modelo.predict_proba(X_test)[:, 1]
    preds = modelo.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)
    
    print(f"✅ Acurácia: {acc * 100:.2f}%")
    print(f"✅ Brier Score: {brier:.4f} (Quanto menor melhor)")
    
    #variaveis mais explicativas
    importancia = modelo.feature_importances_
    for f, imp in zip(features, importancia):
        print(f"Importância da variável '{f}': {imp:.4f}")

    #salvar em .json
    modelo.save_model('nba_wp_model_v2.json')

if __name__ == "__main__":
    treinar_modelo()

