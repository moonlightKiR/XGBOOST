# XGBoost Model
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import load_iris

def split_df(df):
    X = df.filter(like="px_").values
    y = df["category"].values
    
    #separar en train y test con estratificación para mantener la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #codificar las etiquetas de las clases
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    return X_train, X_test, y_train, y_test, le

# Entrenamiento con early stopping manual
def train_model(X_train, y_train, X_test, y_test,
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.5,
                reg_alpha=0.5,
                reg_lambda=1,
                early_stopping_rounds=30):

    num_class = len(np.unique(y_train))
    
    # Configurar el modelo con los hiperparámetros especificados
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method="hist",
        random_state=42,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=num_class,
        early_stopping_rounds=early_stopping_rounds
    )

    # Entrenar el modelo con early stopping 
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Obtener la mejor iteración según el early stopping
    best_iteration = model.best_iteration
    print(f"Early stopping alcanzado en iteración: {best_iteration}")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("\nReporte (Test):")
    print(classification_report(y_test, y_pred_test))

    return model, y_pred_train, y_pred_test, best_iteration


def run_grid_search(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [200, 250, 300, 350],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Inicializar modelo base
    xgb_model = XGBClassifier(
        tree_method="hist",
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    # Configurar Grid Search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    print("Iniciando Grid Search...")
    grid_search.fit(X_train, y_train)

    # Resultados
    print(f"\nMejores parámetros encontrados: {grid_search.best_params_}")
    print(f"Mejor Accuracy en CV: {grid_search.best_score_:.4f}")

    # Evaluación final con el mejor modelo
    best_model = grid_search.best_estimator_
    best_model.set_params(early_stopping_rounds=30)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = best_model.predict(X_test)
    print("\nReporte de clasificación con el mejor modelo (Test Set):")
    print(classification_report(y_test, y_pred))

    return best_model


    #optuna

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "objective": "multi:softprob",
        "num_class": len(set(y_train)),
        "tree_method": "hist",
        "random_state": 42,
        "eval_metric": "mlogloss"
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False, early_stopping_rounds=30)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
