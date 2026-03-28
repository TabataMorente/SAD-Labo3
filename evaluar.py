# -*- coding: utf-8 -*-
import sys
import os
import joblib  # Librería para cargar los modelos guardados (.sav) de forma eficiente
import pandas as pd
# Importamos las métricas necesarias: F1 para clasificación, R2 para regresión y la Matriz de Confusión
from sklearn.metrics import f1_score, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Importamos las funciones de utilidad desde tu script train.py
# Esto asegura que la carga y el preprocesamiento sean idénticos en ambos procesos
from train import load_data, load_config, apply_preprocessing
from sklearn.preprocessing import LabelEncoder


def evaluar_mejor_modelo():
    """
    Función maestra para evaluar el rendimiento del mejor modelo obtenido
    frente a un conjunto de datos de test (examen final).
    """

    # --- 1. CONFIGURACIÓN INICIAL Y CARGA DE RUTAS ---
    # Comprobamos si el usuario ha pasado los 3 archivos por consola. Si no, usamos valores por defecto.
    if len(sys.argv) < 4:
        train_path, test_path, config_path = "ventas_train.csv", "ventas_test.csv", "config_file.json"
        print(f"⚠️ INFO: Usando rutas por defecto: {train_path}, {test_path}, {config_path}")
    else:
        train_path, test_path, config_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Leemos el archivo JSON para conocer la configuración del experimento
    config = load_config(config_path)
    method = config.get('method', 'knn')  # Metodo de ML (knn, bayes, tree, forest)
    task = config.get('task', 'classification')  # Tipo de tarea (clasificación o regresión)
    target = config['target']  # Nombre de la columna que queremos predecir
    csv_id = os.path.basename(train_path).split('.')[0]  # Nombre del archivo sin extensión (ej: ventas_train)

    # --- 2. RECONSTRUCCIÓN DEL ESTADO DE ENTRENAMIENTO ---
    # Cargamos el dataset completo de entrenamiento y el de test
    df_full_train = load_data(train_path, config)
    df_test_final = load_data(test_path, config)

    # REPETIMOS EL SPLIT EXACTO: Es vital usar el mismo random_state y stratify que en train.py
    # Necesitamos X_train_80 para que el preprocesado 'encaje' con lo que el modelo aprendió (las 215 columnas)
    X_train_80, _, _, _ = train_test_split(
        df_full_train.drop(columns=[target]),
        df_full_train[target],
        test_size=0.2,
        random_state=42,
        stratify=df_full_train[target] if task == 'classification' else None
    )

    # Separamos las características (X) y la realidad (y) del conjunto de Test externo
    X_test_final = df_test_final.drop(columns=[target])
    y_test_final = df_test_final[target]

    # PASO CRÍTICO: Preprocesamos el test usando X_train_80 como base para el 'fit'
    # Esto garantiza que si hubo TF-IDF o escalado, las dimensiones finales sean las correctas.
    _, _, X_test_p = apply_preprocessing(X_train_80, X_train_80, X_test_final, config)

    # --- 3. LOCALIZACIÓN Y CARGA DEL MODELO GANADOR ---
    # Definimos la carpeta donde el train.py guardó al 'campeón'
    final_folder = "modelos_finales"
    model_name = f"MEJOR_{method}_{csv_id}.sav"
    ruta_modelo = os.path.join(final_folder, model_name)

    # Verificamos que el archivo existe antes de intentar cargarlo
    if not os.path.exists(ruta_modelo):
        print(f"❌ ERROR: No se encuentra el archivo {ruta_modelo}. ¿Has ejecutado train.py primero?")
        return

    print("\n" + "=" * 60)
    print(f" 🏆 EVALUANDO EL MEJOR MODELO: {model_name}")
    print("=" * 60)

    # Cargamos el modelo con joblib y realizamos las predicciones sobre el test preprocesado
    modelo = joblib.load(ruta_modelo)
    y_pred = modelo.predict(X_test_p)

    # --- 4. MÉTRICAS Y MAPEOS UNIVERSALES (CON LABEL ENCODER) ---
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # 1. Convertimos la REALIDAD a números (ajustándose a lo que haya en el test)
    y_test_num = le.fit_transform(y_test_final)

    # 2. Convertimos la PREDICCIÓN
    # Si el modelo ya devuelve números, lo dejamos. Si devuelve texto, lo mapeamos.
    if y_pred.dtype == 'object' or isinstance(y_pred[0], str):
        y_pred_num = le.transform(y_pred)
    else:
        y_pred_num = y_pred

    # Leemos la estrategia de evaluación (macro/micro)
    eval_strat = config.get('evaluation', 'macro')

    if task == 'regression':
        score = r2_score(y_test_num, y_pred_num)
        print(f" ✅ RESULTADO FINAL (R2-Score): {score:.4f}")
    else:
        # Ahora ambos son arrays de números (int), no habrá Mix of label types
        score = f1_score(y_test_num, y_pred_num, average=eval_strat)
        print(f" ✅ RESULTADO FINAL (F1-Score {eval_strat}): {score:.4f}")

        # --- 🧩 CÁLCULO DE LA MATRIZ DE CONFUSIÓN ---
        # Nos permite ver cuántos aciertos y fallos ha tenido el modelo por cada clase
        print("\n 📊 MATRIZ DE CONFUSIÓN (Fila: Real | Columna: Predicho):")
        matrix = confusion_matrix(y_test_num, y_pred_num)
        print(matrix)
        print(f"\n Desglose de resultados:")
        print(f" - [0,0] Verdaderos Negativos (Dijo No y era No): {matrix[0][0]}")
        print(f" - [0,1] Falsos Positivos (Dijo Si y era No): {matrix[0][1]}")
        print(f" - [1,0] Falsos Negativos (Dijo No y era Si): {matrix[1][0]}")
        print(f" - [1,1] Verdaderos Positivos (Dijo Si y era Si): {matrix[1][1]}")

    # --- 5. EXPORTACIÓN DE RESULTADOS A CSV ---
    # Creamos la carpeta de salida (ej: csv/ventas_train/) si no existe todavía
    salida_dir = os.path.join("csv", csv_id)
    os.makedirs(salida_dir, exist_ok=True)

    # Creamos un DataFrame para comparar fila a fila la realidad vs la predicción
    df_resultados = pd.DataFrame({
        'Valor_Real': y_test_num,
        'Prediccion_Campeon': y_pred_num
    })

    # Guardamos el archivo CSV final para análisis externo (ej: en Excel)
    ruta_unificada = os.path.join(salida_dir, f"evaluacion_final_{method}.csv")
    df_resultados.to_csv(ruta_unificada, index=False)

    print("\n" + "📂 " * 10)
    print(f"ARCHIVO DE RESULTADOS GENERADO EN: {ruta_unificada}")
    print("📂 " * 10 + "\n")


if __name__ == "__main__":
    # Punto de entrada del script
    evaluar_mejor_modelo()