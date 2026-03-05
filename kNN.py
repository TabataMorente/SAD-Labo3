# -*- coding: utf-8 -*-
"""
Autor: Xabier Gabiña Barañano
Script para la implementación del algoritmo kNN
Recoge los datos de un fichero csv y los clasifica en función de los k vecinos más cercanos
"""

import sys
import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


def load_data(csv_file, config_file):
    """
    Carga los datos y asegura que la clase (target) sea siempre la última columna.
    Esto permite que el resto del script funcione sea cual sea el CSV.
    """
    # Leemos el archivo CSV crudo
    df = pd.read_csv(csv_file)

    # Obtenemos el nombre de la columna objetivo desde el JSON de configuración [cite: 75]
    target = config_file['target']

    # Identificamos todas las columnas que NO son el objetivo usando una lista de comprensión
    # Esto responde a la pregunta del guion sobre cómo acceder a atributos de forma general [cite: 57, 59]
    columnas_features = [c for c in df.columns if c != target]

    # Reorganizamos el DataFrame: primero las características y al final la clase objetivo
    # Así, cuando hagamos .iloc[:, -1] siempre obtendremos la etiqueta correcta
    df = df[columnas_features + [target]]

    return df


def apply_preprocessing(X_train, X_test, config):
    """
    Realiza la limpieza y normalización de datos basándose en el JSON.
    Usa el conjunto de entrenamiento para aprender los parámetros y los aplica al de test.
    """
    # Extraemos la sección de preprocesamiento del diccionario de configuración [cite: 74]
    prep_cfg = config['preprocessing']

    # --- 1. GESTIÓN DE VALORES FALTANTES (IMPUTACIÓN) ---
    # Leemos del JSON si queremos usar la media, mediana o moda [cite: 64, 77]
    strategy = prep_cfg.get('impute_strategy', 'mean')
    imputer = SimpleImputer(strategy=strategy)

    # IMPORTANTE: Aprendemos la media/mediana SOLO del entrenamiento (fit)
    # Luego aplicamos ese valor aprendido tanto a Train como a Test (transform) [cite: 65, 67]
    # Esto evita que los datos del test "contaminen" el entrenamiento
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # --- 2. TRANSFORMACIÓN DE ESCALA (NORMALIZACIÓN) ---
    # Mapeamos las opciones de texto del JSON a las clases reales de Scikit-Learn [cite: 78]
    scalers = {
        "z-score": StandardScaler(),  # Ajusta los datos para tener media 0 y desviación 1
        "min-max": MinMaxScaler(),  # Escala los datos al rango [0, 1] [cite: 78]
        "max-abs": MaxAbsScaler()  # Útil si los datos ya están centrados o son dispersos
    }

    # Seleccionamos el escalador indicado en el JSON; si no hay, usamos z-score por defecto
    scaling_type = prep_cfg.get('scaling', 'z-score')
    scaler = scalers.get(scaling_type, StandardScaler())

    # De nuevo: fit en entrenamiento y transform en ambos conjuntos [cite: 65, 67]
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def calculate_fscore(y_test, y_pred):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    from sklearn.metrics import f1_score
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    return fscore_micro, fscore_macro

def calculate_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm

def kNN(data, k, weights, p):
    """
    Función para implementar el algoritmo kNN
    
    :param data: Datos a clasificar
    :type data: pandas.DataFrame
    :param k: Número de vecinos más cercanos
    :type k: int
    :param weights: Pesos utilizados en la predicción ('uniform' o 'distance')
    :type weights: str
    :param p: Parámetro para la distancia métrica (1 para Manhattan, 2 para Euclídea)
    :type p: int
    :return: Clasificación de los datos
    :rtype: tuple
    """
    # Seleccionamos las características y la clase
    X = data.iloc[:, :-1].values # Todas las columnas menos la última
    y = data.iloc[:, -1].values # Última columna
    
    # Dividimos los datos en entrenamiento y test
    from sklearn.model_selection import train_test_split
    np.random.seed(42)  # Set a random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Escalamos los datos
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Entrenamos el modelo
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = k, weights = weights, p = p)
    classifier.fit(X_train, y_train)
    
    # Predecimos los resultados
    y_pred = classifier.predict(X_test)
    
    return y_test, y_pred

if __name__ == "__main__":
    # Comprobamos que se han introducido los parámetros correctos
    if len(sys.argv) < 3:
        print("Error en los parámetros de entrada")
        print("Uso: kNN.py <fichero*> <k*> <weights> <p>")
        sys.exit(1)
    
    # Cargamos los datos
    data = load_data(sys.argv[1])
    
    # Implementamos el algoritmo kNN
    y_test, y_pred = kNN(data, int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'uniform', int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    
    # Mostramos la matriz de confusión
    print("\nMatriz de confusión:")
    print(calculate_confusion_matrix(y_test, y_pred))

    # Mostramos el F-score
    print("\nF-score:")
    print(calculate_fscore(y_test, y_pred))
