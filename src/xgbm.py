# XGBoost Model
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
        X, y, test_size=0.2, random_state=42
    )

    # Codificar etiquetas
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    return X_train, X_test, y_train, y_test


    # Modelo
def train_model(X_train, y_train, X_test, y_test, n_estimators=200, max_depth=6, learning_rate=0.1):
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def run_grid_search(X_train, y_train, X_test, y_test):
    # Definir el rango de búsqueda
    param_grid = {
        'n_estimators': list(range(50, 350, 50)) # 50, 100, 150, 200, 250, 300
    }
    
    # Inicializar el modelo base
    xgb_model = XGBClassifier(
        tree_method="hist",
        random_state=42
    )

    # Configurar Grid Search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1
    )

    print("Iniciando Grid Search...")
    grid_search.fit(X_train, y_train)

    # Resultados
    print(f"\nMejores parámetros encontrados: {grid_search.best_params_}")
    print(f"Mejor Accuracy en CV: {grid_search.best_score_:.4f}")

    # Evaluación final con el mejor modelo
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nReporte de clasificación con el mejor modelo (Test Set):")
    print(classification_report(y_test, y_pred))
    
    return best_model