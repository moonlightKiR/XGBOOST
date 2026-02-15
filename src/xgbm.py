# XGBoost Model
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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
def train_model(X_train, y_train, X_test, y_test):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))