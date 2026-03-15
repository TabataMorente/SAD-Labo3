# -*- coding: utf-8 -*-
import sys
import json
import pickle
import joblib  # Usamos joblib por eficiencia con arrays grandes
import pandas as pd
import numpy as np

# Herramientas de Scikit-Learn para dividir datos, limpiar y preprocesar
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# Librerías para el balanceo
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Función para abrir y leer el archivo de configuración JSON
def load_config(json_path):
    with open(json_path, 'r') as f: # Abre el archivo en modo lectura
        return json.load(f) # Convierte el contenido del JSON en un diccionario de Python

# Función para cargar el CSV y mover la columna objetivo al final del DataFrame
def load_data(csv_file, config):
    df = pd.read_csv(csv_file) # Carga el archivo CSV en un DataFrame
    target = config['target']
    # Crea una lista de columnas sin el target y lo añade al final
    cols = [c for c in df.columns if c != target] + [target]
    return df[cols]


def apply_preprocessing(X_train, X_dev, X_test, config):
    """
    FLUJO DE PREPROCESADO COMPLETO:
    1. Texto -> 2. Categorías -> 3. Booleanos -> 4. Imputación -> 5. Outliers -> 6. Escalado
    """
    prep_cfg = config['preprocessing']

    # IMPORTANTE: Convertimos a DataFrame para poder usar nombres de columnas y tipos
    # Usamos los nombres originales de las columnas del CSV

    train_df = pd.DataFrame(X_train)
    dev_df = pd.DataFrame(X_dev)
    test_df = pd.DataFrame(X_test)

    # --- 1. PREPROCESADO DE TEXTO (TF-IDF / BoW / One-Hot) ---
    text_cols = prep_cfg.get('text_features', [])
    if text_cols:
        method = prep_cfg.get('text_process', 'tf-idf')

        # Elegimos la técnica de vectorización según indique el JSON
        if method == 'tf-idf':
            vec = TfidfVectorizer()
        elif method == 'bow':
            vec = CountVectorizer()
        elif method == 'one-hot':
            vec = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        for col in text_cols:
            # fit_transform aprende el vocabulario del TRAIN; transform lo aplica a DEV/TEST
            t_train = vec.fit_transform(train_df[col].astype(str).values.reshape(-1, 1))
            t_dev = vec.transform(dev_df[col].astype(str).values.reshape(-1, 1))
            t_test = vec.transform(test_df[col].astype(str).values.reshape(-1, 1))

            # Si es OHE, convertimos a array si no lo es ya
            if hasattr(t_train, "toarray"):
                t_train = t_train.toarray()
                t_dev = t_dev.toarray()
                t_test = t_test.toarray()

            # Reemplazamos la columna original por las nuevas columnas numéricas
            train_df = pd.concat([train_df.drop(columns=[col]), pd.DataFrame(t_train)], axis=1)
            dev_df = pd.concat([dev_df.drop(columns=[col]), pd.DataFrame(t_dev)], axis=1)
            test_df = pd.concat([test_df.drop(columns=[col]), pd.DataFrame(t_test)], axis=1)

    # --- 2. CATEGORIALES (Reemplazo por número/Ordinal) ---
    cat_cols = prep_cfg.get('categorical_features', [])
    for col in cat_cols:
        # Crea un mapa: cada categoría única recibe un número (0, 1, 2...)
        categorias = train_df[col].unique()
        mapeo_cat = {val: i for i, val in enumerate(categorias)}
        # Reemplazamos cada categoría por un número
        train_df[col] = train_df[col].map(mapeo_cat)
        dev_df[col] = dev_df[col].map(mapeo_cat)
        test_df[col] = test_df[col].map(mapeo_cat)

    # --- 3. BOOLEANOS (Conversión de Texto a 0/1) ---
    # Solo se procesan los que vienen en el JSON (formato texto)
    bool_cols = prep_cfg.get('boolean_features', [])
    mapeo_bool = {'true': 1, 'false': 0, 'sí': 1, 'no': 0, 'yes': 1, 'si': 1, '1': 1, '0': 0}
    for col in bool_cols:
        train_df[col] = train_df[col].astype(str).str.lower().map(mapeo_bool)
        dev_df[col] = dev_df[col].astype(str).str.lower().map(mapeo_bool)
        test_df[col] = test_df[col].astype(str).str.lower().map(mapeo_bool)

    # --- 4. GESTIÓN DE MISSING VALUES ---
    # Ahora que todas es número, imputamos
    if prep_cfg.get('missing_values') == 'impute':
        strategy = prep_cfg.get('impute_strategy', 'mean')
        imputer = SimpleImputer(strategy=strategy)
        cols_nombres = train_df.columns
        # Rellenamos huecos (NaN) usando la estrategia (media, moda...) calculada en TRAIN
        train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=cols_nombres)
        dev_df = pd.DataFrame(imputer.transform(dev_df), columns=cols_nombres)
        test_df = pd.DataFrame(imputer.transform(test_df), columns=cols_nombres)
    else:
        # Si la opción es "eliminar", borramos filas con NaNs
        train_df = train_df.dropna()
        dev_df = dev_df.dropna()
        test_df = test_df.dropna()

    # --- 5. GESTIÓN DE OUTLIERS (IQR Clipping) ---
    # Solo actuamos sobre las columnas que originalmente eran numéricas
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        # Calculamos los límites estadísticos (Cuartiles)
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        # Recortamos los valores en los 3 conjuntos usando los límites de TRAIN
        train_df[col] = np.clip(train_df[col], lower_limit, upper_limit)
        dev_df[col] = np.clip(dev_df[col], lower_limit, upper_limit)
        test_df[col] = np.clip(test_df[col], lower_limit, upper_limit)

    # --- 6. ESCALADO FINAL  ---
    # Normalizamos los rangos de los números para que el KNN funcione bien
    if prep_cfg.get('scaling') == 'max-min':
        scaler = MinMaxScaler()
    elif prep_cfg.get('scaling') == 'z-score':
        scaler = StandardScaler()
    elif prep_cfg.get('scaling') == 'max':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()  # Escala robusta frente a outliers residuales

    # Ajustamos el escalador con TRAIN y transformamos los tres conjuntos
    X_train_final = scaler.fit_transform(train_df)
    X_dev_final = scaler.transform(dev_df)
    X_test_final = scaler.transform(test_df)

    return X_train_final, X_dev_final, X_test_final


def train():
    # Validación de que el usuario ha pasado los archivos por consola
    if len(sys.argv) < 3:
        print("Uso: python knn_train.py <datos.csv> <config.json>")
        sys.exit(1)

    # Carga de configuración y datos iniciales
    config = json.load(open(sys.argv[2], 'r'))
    df = pd.read_csv(sys.argv[1])
    target = config['target']

    # Separamos características (X) de la etiqueta a predecir (y)
    X = df.drop(columns=[target])
    y = df[target]

    # Primera división: 70% entrenamiento, 30% para un conjunto temporal
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    # Segunda división: repartimos el 30% temporal en 15% validación (Dev) y 15% Test
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # 1. Aplicamos el preprocesado a todos
    X_train_p, X_dev_p, X_test_p = apply_preprocessing(X_train.copy(), X_dev.copy(), X_test.copy(), config)

    # --- 2. BALANCEO (Sampling) - EXCLUSIVO PARA TRAIN ---
    sampling_type = config['preprocessing'].get('sampling')
    if sampling_type == "undersampling":
        sampler = RandomUnderSampler(random_state=42)
        X_train_p, y_train = sampler.fit_resample(X_train_p, y_train)
    elif sampling_type == "oversampling":
        sampler = RandomOverSampler(random_state=42)
        X_train_p, y_train = sampler.fit_resample(X_train_p, y_train)

    # BARRIDO DE HIPERPARÁMETROS: Probamos combinaciones de k, p y pesos
    best_f1 = -1
    best_model = None
    best_params = ""

    for k in [1, 3, 5]: # Numero de vecinos
        for p in [1, 2]: # Distancia (1: Manhattan, 2: Euclídea)
            for w in ['uniform', 'distance']: # Peso de los vecinos
                knn = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
                knn.fit(X_train_p, y_train) # Entrenamos con los datos balanceados
                f1 = f1_score(y_dev, knn.predict(X_dev_p), average='macro') # Evaluamos con DEV

                # Si este modelo es mejor que el anterior, lo guardamos como el "Mejor"
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = knn
                    best_params = f"k={k}_p={p}_w={w}"

    # Guardamos el modelo ganador en un archivo físico para usarlo después
    model_name = f"knn_{best_params}.sav"
    pickle.dump(best_model, open(model_name, 'wb'))
    print(f"✅ Balanceo '{sampling_type}' aplicado. Modelo guardado: {model_name}")


if __name__ == "__main__":
    train()