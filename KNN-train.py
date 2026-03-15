# -*- coding: utf-8 -*-
import sys
import json
import joblib  # Usamos joblib por eficiencia con arrays grandes
import pandas as pd
import numpy as np

# Herramientas de Scikit-Learn para dividir datos, limpiar y preprocesar
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

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
    target = config['target'] # Extrae el nombre de la columna objetivo del JSON
    # Crea una lista de columnas sin el target y lo añade al final
    cols = [c for c in df.columns if c != target] + [target]
    return df[cols] # Devuelve el DataFrame ordenado


def apply_preprocessing(X_train, X_dev, X_test, config):
    """
    FLUJO DE PREPROCESADO COMPLETO:
    1. Texto -> 2. Categorías -> 3. Booleanos -> 4. Imputación -> 5. Outliers -> 6. Escalado
    """
    prep_cfg = config['preprocessing'] # Leer la configuración de preprocesado del JSON

    # IMPORTANTE: Convertimos a DataFrame para poder usar nombres de columnas y tipos
    # Usamos los nombres originales de las columnas del CSV

    train_df = pd.DataFrame(X_train)
    dev_df = pd.DataFrame(X_dev)
    test_df = pd.DataFrame(X_test)

    # --- 1. PREPROCESADO DE TEXTO (TF-IDF / BoW / One-Hot) ---
    text_cols = prep_cfg.get('text_features', []) # Busca si hay columnas de texto definidas en el JSON
    if text_cols:
        method = prep_cfg.get('text_process', 'tf-idf') # Obtiene el metodo de procesado

        # Elegimos la técnica de vectorización según indique el JSON
        if method == 'tf-idf':
            vec = TfidfVectorizer()
        elif method == 'bow':
            vec = CountVectorizer()
        else: #Si es OneHot encoding
            vec = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Texto como categoria

        for col in text_cols:
            # fit_transform aprende el vocabulario del TRAIN; transform lo aplica a DEV/TEST
            t_train = vec.fit_transform(train_df[col].astype(str).values.reshape(-1, 1))
            t_dev = vec.transform(dev_df[col].astype(str).values.reshape(-1, 1))
            t_test = vec.transform(test_df[col].astype(str).values.reshape(-1, 1))

            # Si el resultado es una matriz dispersa (sparse), la convertimos a densa (array)
            if hasattr(t_train, "toarray"):
                t_train = t_train.toarray()
                t_dev = t_dev.toarray()
                t_test = t_test.toarray()

            # Borramos la columna de texto original y concatenamos las nuevas columnas numéricas
            train_df = pd.concat([train_df.drop(columns=[col]), pd.DataFrame(t_train)], axis=1)
            dev_df = pd.concat([dev_df.drop(columns=[col]), pd.DataFrame(t_dev)], axis=1)
            test_df = pd.concat([test_df.drop(columns=[col]), pd.DataFrame(t_test)], axis=1)

    # --- 2. CATEGORIALES (Reemplazo por número/Ordinal) ---
    cat_cols = prep_cfg.get('categorical_features', [])
    for col in cat_cols:
        # Crea un mapa: cada categoría única recibe un número (0, 1, 2...)
        categorias = train_df[col].unique() # Obtiene los valores únicos (ej: 'rojo', 'azul')
        mapeo_cat = {val: i for i, val in enumerate(categorias)} # Crea mapa {'rojo': 0, 'azul': 1}
        # Transforma las palabras en números usando el mapa anterior
        train_df[col] = train_df[col].map(mapeo_cat)
        dev_df[col] = dev_df[col].map(mapeo_cat)
        test_df[col] = test_df[col].map(mapeo_cat)

    # --- 3. BOOLEANOS (Conversión de Texto a 0/1) ---
    # Solo se procesan los que vienen en el JSON (formato texto)
    bool_cols = prep_cfg.get('boolean_features', [])
    # Diccionario de traducción para normalizar diferentes formas de escribir booleanos
    mapeo_bool = {'true': 1, 'false': 0, 'sí': 1, 'no': 0, 'yes': 1, 'si': 1, '1': 1, '0': 0}
    for col in bool_cols:
        # Convierte a minúsculas, traduce según el mapa y guarda como número
        train_df[col] = train_df[col].astype(str).str.lower().map(mapeo_bool)
        dev_df[col] = dev_df[col].astype(str).str.lower().map(mapeo_bool)
        test_df[col] = test_df[col].astype(str).str.lower().map(mapeo_bool)

    # --- 4. GESTIÓN DE MISSING VALUES ---
    # Ahora que todas es número, imputamos
    if prep_cfg.get('missing_values') == 'impute':
        strategy = prep_cfg.get('impute_strategy', 'mean') # 'mean' rellenará con la media
        imputer = SimpleImputer(strategy=strategy) # Configura el imputador
        cols_nombres = train_df.columns # Guarda los nombres de las columnas
        # Rellenamos huecos (NaN) usando la estrategia (media, moda...) calculada en TRAIN
        # fit aprende las medias de TRAIN; transform las aplica para rellenar huecos en todos
        train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=cols_nombres)
        dev_df = pd.DataFrame(imputer.transform(dev_df), columns=cols_nombres)
        test_df = pd.DataFrame(imputer.transform(test_df), columns=cols_nombres)
    else:
        # Si la opción es "eliminar", borramos filas con NaNs
        train_df = train_df.dropna()
        dev_df = dev_df.dropna()
        test_df = test_df.dropna()

    # --- 5. GESTIÓN DE OUTLIERS (IQR Clipping) ---
    # Solo actúa en columnas numéricas
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        # Calculamos los límites estadísticos (Cuartiles)
        Q1 = train_df[col].quantile(0.25) # Primer cuartil (percentil 25)
        Q3 = train_df[col].quantile(0.75) # Tercer cuartil (percentil 75)
        IQR = Q3 - Q1 # Rango intercuartílico (la "anchura" de la caja)
        lower_limit = Q1 - 1.5 * IQR # Límite inferior
        upper_limit = Q3 + 1.5 * IQR # Límite superior

        # Recortamos los valores en los 3 conjuntos usando los límites de TRAIN
        train_df[col] = np.clip(train_df[col], lower_limit, upper_limit)
        dev_df[col] = np.clip(dev_df[col], lower_limit, upper_limit)
        test_df[col] = np.clip(test_df[col], lower_limit, upper_limit)

    # --- 6. ESCALADO FINAL  ---
    # Normalizamos los rangos de los números para que el KNN funcione bien
    if prep_cfg.get('scaling') == 'max-min':
        scaler = MinMaxScaler() # Escala al rango entre 0 y 1
    elif prep_cfg.get('scaling') == 'z-score':
        scaler = StandardScaler() # Centra los datos (media 0, desviación 1)
    elif prep_cfg.get('scaling') == 'max':
        scaler = StandardScaler() # Centra los datos (media 0, desviación 1)
    else:
        scaler = RobustScaler()  # Escala robusta frente a outliers residuales

    # Ajustamos el escalador con TRAIN y transformamos los tres conjuntos
    X_train_final = scaler.fit_transform(train_df)
    X_dev_final = scaler.transform(dev_df)
    X_test_final = scaler.transform(test_df)

    return X_train_final, X_dev_final, X_test_final


def train():
    # 1. Validación de que el usuario ha pasado los archivos por consola
    if len(sys.argv) < 3:
        print("Uso: python knn_train.py <datos.csv> <config.json>")
        sys.exit(1)

    # 2. Carga de configuración y datos iniciales
    config = load_config(sys.argv[2])  # Usamos el metodo load_config
    df = load_data(sys.argv[1], config)  # Usamos el metodo load_data (que ya ordena el target al final)

    # Separamos características (X) de la etiqueta a predecir (y)
    # Como load_data garantizó que el target es la última columna:
    X = df.iloc[:, :-1]  # "Coge todas las columnas menos la última"
    y = df.iloc[:, -1]  # "Coge solo la última columna"

    # DIVISIÓN 70/15/15:

    # 1. Separa el 70% para Train y deja el 30% en un grupo temporal (X_temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    # 2. Divide ese 30% a la mitad: 15% para Desarrollo (Dev) y 15% para Test final
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # Aplica todas las transformaciones de limpieza
    X_train_p, X_dev_p, X_test_p = apply_preprocessing(X_train.copy(), X_dev.copy(), X_test.copy(), config)

    # --- 2. BALANCEO (Sampling) - (Solo se aplica al conjunto de TRAIN para que el modelo no tenga sesgos) ---
    sampling_type = config['preprocessing'].get('sampling')
    if sampling_type == "undersampling":
        sampler = RandomUnderSampler(random_state=42) # Borra filas de la clase mayoritaria
        X_train_p, y_train = sampler.fit_resample(X_train_p, y_train)
    elif sampling_type == "oversampling":
        sampler = RandomOverSampler(random_state=42) # Inventa filas de la clase minoritaria
        X_train_p, y_train = sampler.fit_resample(X_train_p, y_train)

    # --- DETECCIÓN DE TAREA Y CONFIGURACIÓN ---
    task = config.get('task', 'classification')  # Por defecto clasificación si no existe la clave

    best_score = -1 # Inicializamos la mejor métrica en el peor valor posible
    best_model = None # Aquí guardaremos el objeto del modelo ganador
    best_params = "" # Aquí guardaremos el nombre de los parámetros ganadores

    # BARRIDO DE HIPERPARÁMETROS: Probamos combinaciones de k, p y pesos
    for k in [1, 3, 5]: # k: número de vecinos a consultar
        for p in [1, 2]: # p=1 es distancia Manhattan, p=2 es distancia Euclídea
            for w in ['uniform', 'distance']: # w: peso de la distancia (uniforme o ponderado)

                # 1. ELEGIMOS EL ALGORITMO SEGÚN LA TAREA
                if task == 'regression':
                    model = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
                else:
                    model = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)

                # 2. ENTRENAMIENTO
                model.fit(X_train_p, y_train) # El modelo aprende con los datos de TRAIN

                # 3. EVALUACIÓN SEGÚN LA TAREA
                if task == 'regression':
                    # Para regresión usamos R² (Coeficiente de determinación)
                    # 1.0 es perfecto, valores bajos son malos.
                    score = r2_score(y_dev, model.predict(X_dev_p))
                else:
                    # Para clasificación seguimos con F1-Score
                    score = f1_score(y_dev, model.predict(X_dev_p), average='macro')

                # 4. GUARDAR EL MEJOR (Funciona igual para ambos porque cuanto más alto el score, mejor)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = f"k={k}_p={p}_w={w}"

    # GUARDADO FINAL: Creamos el nombre del archivo con los parámetros ganadores
    model_name = f"{task}_knn_{best_params}.sav"
    # Guardamos el objeto del mejor modelo en un archivo para usarlo en el futuro
    joblib.dump(best_model, model_name)
    print(f"✅ Tarea de {task} completada. Modelo guardado: {model_name}")


if __name__ == "__main__":
    train()