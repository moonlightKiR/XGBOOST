# Proyecto Quick, Draw! Dataset con SQLite y XGBoost

Este proyecto automatiza la descarga del dataset [Quick, Draw!](https://github.com/googlecreativelab/quickdraw_dataset) de Google, su almacenamiento en una base de datos local SQLite y su posterior analisis y modelado.

## Estructura del Proyecto

```text
XGBOOST/
├── database/               # Contiene la base de datos local quickdraw.db
├── src/
│   ├── data_quickdraw/     # Carpeta local para los archivos .npy descargados
│   ├── database.py         # Modulo para gestion de base de datos (descarga, creacion e insercion)
│   ├── eda.py              # Modulo con funciones comunes para Analisis Exploratorio de Datos
│   └── main.ipynb          # Notebook principal donde se orquestan los procesos
├── LICENSE
└── README.md
```

## Flujo de Trabajo

### 1. Preparacion del Entorno
Es necesario instalar las dependencias principales:
```bash
pip install numpy pandas requests tqdm matplotlib seaborn scipy
```

### 2. Gestion de Datos (Modulo database.py)
El proceso de datos esta modularizado para permitir una gestion limpia:
* **Inicializacion**: Crea la carpeta `database/` y la estructura de tablas en `quickdraw.db`.
* **Descarga**: Obtiene los archivos `.npy` directamente desde Google Cloud Storage y los guarda en `src/data_quickdraw/`.
* **Carga**: Inserta los datos en la base de datos SQLite como objetos BLOB, limitando por defecto a 5.000 imagenes por categoria para optimizar el rendimiento.

### 3. Analisis Exploratorio (Modulo eda.py)
Contiene funciones reutilizables para el analisis de datos:
* `data_resume_info`: Resumen general, nulos y duplicados.
* `outlier_impact_test`: Analisis del impacto de los valores atipicos.
* `graf_box_hist`: Visualizacion dinamica de histogramas y boxplots.
* `plot_binary_bars`: Analisis de variables binarias.
* `analyze_shapiro_qq`: Pruebas de normalidad (Shapiro-Wilk) y graficos Q-Q.

## Uso del Notebook

El notebook `main.ipynb` actua como el orquestador central del proyecto. Sus funciones principales son:

* Coordinar la descarga de las imagenes originales y la poblacion automatica de la base de datos local utilizando los modulos internos.
* Realizar la carga masiva de los datos desde SQLite hacia la memoria de trabajo para ejecutar el analisis exploratorio de datos (EDA).
* Implementar las transformaciones necesarias para preparar las variables de entrada y etiquetas para el modelo de clasificacion.

## Categorias Incluidas
Por defecto, el proyecto trabaja con las siguientes clases:
* apple, banana, cactus, broom, bicycle.
Cada dibujo es una matriz de 28x28 pixeles.

## Tecnologias
* Python 3.10+
* SQLite3
* Pandas, NumPy, Scikit-learn, XGBoost
* Matplotlib & Seaborn
