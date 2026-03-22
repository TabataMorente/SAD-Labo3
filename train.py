# -*- coding: utf-8 -*-
import sys
import os  # Para manejar nombres de archivos y rutas
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
from sklearn.naive_bayes import GaussianNB  # --- Algoritmo Naive Bayes ---
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
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        print("INFO: Columna 'ID' eliminada para mejorar el entrenamiento.")
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

    train_df = pd.DataFrame(X_train).reset_index(drop=True)
    dev_df = pd.DataFrame(X_dev).reset_index(drop=True)
    test_df = pd.DataFrame(X_test).reset_index(drop=True)

    # --- 1. PREPROCESADO DE TEXTO (TF-IDF / BoW / One-Hot) ---
    text_cols = prep_cfg.get('text_features', [])  # Busca si hay columnas de texto definidas en el JSON
    if text_cols:
        method = prep_cfg.get('text_process', 'tf-idf')  # Obtiene el metodo de procesado

        # Elegimos la técnica de vectorización según indique el JSON
        if method == 'tf-idf':
            vec = TfidfVectorizer()
        elif method == 'bow':
            vec = CountVectorizer()
        else:  # Si es OneHot encoding
            vec = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Texto como categoria

        for col in text_cols:
            # 2. TRANSFORMACIÓN: Aquí está el truco de las dimensiones
            if method == 'one-hot':
                # One-Hot NECESITA 2D (DataFrame) -> [[col]]
                t_train = vec.fit_transform(train_df[[col]])
                t_dev = vec.transform(dev_df[[col]])
                t_test = vec.transform(test_df[[col]])
            else:
                # TF-IDF/BoW NECESITAN 1D (Series/Texto) -> [col]
                t_train = vec.fit_transform(train_df[col].astype(str))
                t_dev = vec.transform(dev_df[col].astype(str))
                t_test = vec.transform(test_df[col].astype(str))

            # Si el resultado es una matriz dispersa (sparse), la convertimos a densa (array)
            if hasattr(t_train, "toarray"):
                t_train = t_train.toarray()
                t_dev = t_dev.toarray()
                t_test = t_test.toarray()

            # Definimos nombres de columnas (ej: message_0, message_1...) para que Pandas no se pierda
            col_names = [f"{col}_{i}" for i in range(t_train.shape[1])]

            # Convertimos a DataFrame especificando el tipo de dato (float) desde el principio
            t_train_df = pd.DataFrame(t_train, columns=col_names, dtype=float).reset_index(drop=True)
            t_dev_df = pd.DataFrame(t_dev, columns=col_names, dtype=float).reset_index(drop=True)
            t_test_df = pd.DataFrame(t_test, columns=col_names, dtype=float).reset_index(drop=True)

            # Borramos la columna de texto original y concatenamos las nuevas columnas numéricas
            train_df = pd.concat([train_df.drop(columns=[col]).reset_index(drop=True), t_train_df], axis=1)
            dev_df = pd.concat([dev_df.drop(columns=[col]).reset_index(drop=True), t_dev_df], axis=1)
            test_df = pd.concat([test_df.drop(columns=[col]).reset_index(drop=True), t_test_df], axis=1)

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
    if len(sys.argv) < 4:
        print("Uso: python train.py <train.csv> <test.csv> <config_file.json>")
        sys.exit(1)

    # 2. Carga de configuración y datos por separado
    config = load_config(sys.argv[3])
    df_train = load_data(sys.argv[1], config)
    df_test = load_data(sys.argv[2], config)  # Cargamos el test externo directamente

    # Si en el JSON pusiste "drop", limpiamos el DataFrame entero AQUÍ
    if config['preprocessing'].get('missing_values') == 'drop':
        df_train = df_train.dropna().reset_index(drop=True)
        df_test = df_test.dropna().reset_index(drop=True)

    target = config['target']

    # Separamos características (X) de la etiqueta a predecir (y)
    # Como load_data garantizó que el target es la última columna:
    X_train_full = df_train.iloc[:, :-1]  # "Coge todas las columnas menos la última"
    y_train_full = df_train.iloc[:, -1]  # "Coge solo la última columna"

    # Separación Test (el examen final que ya nos dan)
    X_test_final = df_test.iloc[:, :-1]
    y_test_final = df_test.iloc[:, -1]

    # Codificación del target si es clasificación
    if config.get('task') == 'classification' and y_train_full.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_full = pd.Series(le.fit_transform(y_train_full))
        y_test_final = pd.Series(le.transform(y_test_final))
        print(f"INFO: Target codificado: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # DIVISION PARA VALIDACIÓN (Dev):
    # Como ya tenemos el Test aparte, dividimos el Train para sacar un 20% para elegir parámetros (Dev)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=42)

    # Aplicamos preprocesado a los tres bloques
    X_train_p, X_dev_p, X_test_p = apply_preprocessing(X_train.copy(), X_dev.copy(), X_test_final.copy(), config)

    # --- 2. BALANCEO (Sampling) - (Solo se aplica al conjunto de TRAIN para que el modelo no tenga sesgos) ---
    sampling_type = config['preprocessing'].get('sampling')
    if sampling_type == "undersampling":
        sampler = RandomUnderSampler(random_state=42) # Borra filas de la clase mayoritaria
        X_train_p, y_train = sampler.fit_resample(X_train_p, y_train)
    elif sampling_type == "oversampling":
        sampler = RandomOverSampler(random_state=42) # Inventa filas de la clase minoritaria
        X_train_p, y_train = sampler.fit_resample(X_train_p, y_train)


    # --- INICIO DEL ENTRENAMIENTO ---
    method = config.get('method', 'knn')
    task = config.get('task', 'classification')
    csv_id = os.path.basename(sys.argv[1]).split('.')[0]

    print(f"🚀 Iniciando entrenamiento con método: {method}...")

    # --- CASO NAIVE BAYES ---
    if method == 'bayes':
        params_cfg = config.get('hyperparameters_bayes', {"var_smoothing": [1e-9]})

        # Naive Bayes suele probarse con diferentes suavizados (smoothing)
        for sm in params_cfg.get('var_smoothing', [1e-9]):
            model = GaussianNB(var_smoothing=sm)
            model.fit(X_train_p, y_train)

            # Evaluación en Dev (para elegir) y en Test (para saber la verdad)
            score_dev = f1_score(y_dev, model.predict(X_dev_p), average='macro')
            score_test = f1_score(y_test_final, model.predict(X_test_p), average='macro')

            # GUARDAMOS TODOS LOS MODELOS GENERADOS
            folder_path = os.path.join("modelos", csv_id, method)
            os.makedirs(folder_path, exist_ok=True)

            params_str = f"sm={sm}"
            model_name = f"{csv_id}_bayes_{params_str}.sav"
            full_save_path = os.path.join(folder_path, model_name)

            joblib.dump(model, full_save_path)
            print(f"✅ Guardado: {full_save_path} | F1-Dev: {score_dev:.4f} | F1-Test: {score_test:.4f}")

    elif method == 'knn':
        # Sacamos las listas del JSON. Si no existen, ponemos unas por defecto []
        params_cfg = config.get('hyperparameters_knn', {})

        # .get(clave, valor_por_defecto)
        lista_k = params_cfg.get('n_neighbors', [1, 3, 5])
        lista_p = params_cfg.get('p', [1, 2])
        lista_w = params_cfg.get('weights', ['uniform', 'distance'])

        # BARRIDO DE HIPERPARÁMETROS: Probamos combinaciones de k, p y pesos
        for k in lista_k: # k: número de vecinos a consultar
            for p in lista_p: # p=1 es distancia Manhattan, p=2 es distancia Euclídea
                for w in lista_w: # w: peso de la distancia (uniforme o ponderado)

                    # 1. ELEGIMOS EL ALGORITMO SEGÚN LA TAREA
                    if task == 'regression':
                        model = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
                        model.fit(X_train_p, y_train)
                        score_dev = r2_score(y_dev, model.predict(X_dev_p))
                        score_test = r2_score(y_test_final, model.predict(X_test_p))
                        metric = "R2"
                    else:
                        model = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
                        model.fit(X_train_p, y_train)
                        score_dev = f1_score(y_dev, model.predict(X_dev_p), average='macro')
                        score_test = f1_score(y_test_final, model.predict(X_test_p), average='macro')
                        metric = "F1"

                    # GUARDAMOS TODOS LOS MODELOS GENERADOS
                    folder_path = os.path.join("modelos", csv_id, method)
                    os.makedirs(folder_path, exist_ok=True)

                    params_str = f"k={k}_p={p}_w={w}"
                    model_name = f"{csv_id}_{task}_knn_{params_str}.sav"
                    full_save_path = os.path.join(folder_path, model_name)

                    joblib.dump(model, full_save_path)
                    print(f"✅ Guardado: {full_save_path} | F1-Dev: {score_dev:.4f} | F1-Test: {score_test:.4f}")

if __name__ == "__main__":
    train()