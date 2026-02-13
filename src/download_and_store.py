import os
import numpy as np
import sqlite3
import requests
from tqdm import tqdm

# Definición de rutas: la base de datos estará en XGBOOST/database/
# (al mismo nivel que la carpeta src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "quickdraw.db")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "src", "data_quickdraw")

def initialize_database():
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS drawings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        data BLOB
    )
    ''')
    conn.commit()
    conn.close()

def download_data():
    categories = ['apple', 'banana', 'cactus', 'broom', 'bicycle']
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
    
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    for category in categories:
        filename = f"{category}.npy"
        url = f"{base_url}{category.replace(' ', '%20')}.npy"
        path = os.path.join(DOWNLOAD_DIR, filename)
        
        if not os.path.exists(path):
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(path, "wb") as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
    return categories

def upload_to_db(categories, limit=5000):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    for category in categories:
        cursor.execute("SELECT COUNT(*) FROM drawings WHERE category = ?", (category,))
        if cursor.fetchone()[0] > 0:
            continue

        path = os.path.join(DOWNLOAD_DIR, f"{category}.npy")
        if os.path.exists(path):
            data = np.load(path)
            data = data[:limit]
            for img in tqdm(data, desc=f"Subiendo {category}"):
                cursor.execute("INSERT INTO drawings (category, data) VALUES (?, ?)", (category, img.tobytes()))
    
    conn.commit()
    conn.close()
    print(f"Todo listo. Base de datos en: {DATABASE_PATH}")
