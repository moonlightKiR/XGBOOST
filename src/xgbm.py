# XGBoost Model
iimport numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def split_df(df):
    X = df.filter(like="px_").values  # todas las columnas que contienen "px_"
    y = df["category"].values
    
    # Separar datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Codificar etiquetas
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    return X_train, X_test, y_train, y_test


    # Modelo
def train_model(X_train, y_train, X_test, y_test,
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                early_stopping_rounds=30):
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    # Entrenar con early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False
    )

    # Mejor número de iteraciones
    best_iteration = model.best_iteration
    print(f"Early stopping alcanzado en iteración: {best_iteration}")

    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("\nReporte (Test):")
    print(classification_report(y_test, y_pred_test))


    return model

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
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                   early_stopping_rounds=30, verbose=False)

    y_pred = best_model.predict(X_test)
    print("\nReporte de clasificación con el mejor modelo (Test Set):")
    print(classification_report(y_test, y_pred))

    return best_model
