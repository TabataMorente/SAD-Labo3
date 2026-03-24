# -*- coding: utf-8 -*-
import sys
import os
import joblib
import pandas as pd
import shutil  # Para copiar archivos
from sklearn.metrics import f1_score

# Importamos tus funciones
from train import load_data, load_config, apply_preprocessing


def evaluar_y_elegir_mejor():
    # Rutas por defecto (para que le des al botón y listo)
    if len(sys.argv) < 4:
        train_path = "ventas_train.csv"
        test_path = "ventas_test.csv"
        config_path = "config_file.json"
    else:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        config_path = sys.argv[3]

    # 2. El programa extrae los datos automáticamente
    config = load_config(config_path)
    method = config.get('method', 'tree') # Lee si es "knn" o "bayes" o "tree" o el metodo que sea

    # Extrae "ventas_train" del nombre del archivo
    csv_id = os.path.basename(train_path).split('.')[0]
    target = config['target']

    # 1. Cargar datos originales
    df_full_train = load_data(train_path, config)  # Las 100 filas originales
    df_test_final = load_data(test_path, config)  # Tu CSV de test externo (el examen)

    # 2. REPETIMOS EL SPLIT (Asegúrate de que random_state sea el mismo que en train.py)
    from sklearn.model_selection import train_test_split
    df_train_80, _ = train_test_split(df_full_train, test_size=0.2, random_state=42)

    # 3. Limpieza de nulos (si está en el JSON)
    if config['preprocessing'].get('missing_values') == 'drop':
        df_train_80 = df_train_80.dropna().reset_index(drop=True)
        df_test_final = df_test_final.dropna().reset_index(drop=True)

    # 4. Preparamos X e y finales
    X_train_final = df_train_80.drop(columns=[target])
    y_train_final = df_train_80[target]

    X_test_final = df_test_final.drop(columns=[target])
    y_test_final = df_test_final[target]

    # 5. Codificación del target
    mapeo_aux = {'no': 0, 'si': 1, '0': 0, '1': 1}

    if config.get('task') == 'classification':
        # Convertimos la realidad a números
        if y_test_final.dtype == 'object':
            y_test_final = y_test_final.str.lower().map(mapeo_aux)
        else:
            y_test_final = y_test_final.astype(int)

    # 6. PASO MAESTRO: Preprocesamos usando las 80 filas para el "fit"
    # Esto garantiza que salgan las 215 columnas que espera Bayes
    _, _, X_test_p = apply_preprocessing(X_train_final, X_train_final, X_test_final, config)

    # Construye la ruta igual que el train.py
    # Carpeta donde están los modelos que acabamos de entrenar
    folder_path = os.path.join("modelos", csv_id, method)

    if not os.path.exists(folder_path):
        print(f"❌ No encuentro la carpeta de modelos: {folder_path}")
        return

    # 2. BUCLE PARA PROBAR TODOS LOS MODELOS
    best_score = -1
    best_model_name = ""

    print("\n" + "=" * 60)
    print(f" INICIANDO EVALUACIÓN DE MODELOS ({method.upper()})")
    print("=" * 60)
    print(f" Cargando modelos desde: {folder_path}\n")

    # Listamos los archivos y los probamos uno a uno
    for file in sorted(os.listdir(folder_path)):  # sorted para que salgan en orden
        if file.endswith(".sav"):
            ruta_completa = os.path.join(folder_path, file)
            modelo = joblib.load(ruta_completa)

            # Predicción con los datos de test preprocesados (las 215 columnas)
            y_pred = modelo.predict(X_test_p)

            # --- PARCHE DE SEGURIDAD PARA TIPOS ---
            # Forzamos a que sea string primero y luego mapeamos a número
            mapeo_aux = {'no': 0, 'si': 1, '0': 0, '1': 1}

            # Convertimos realidad y predicción a Series de Pandas para usar .map()
            y_test_num = pd.Series(y_test_final).astype(str).str.lower().map(mapeo_aux).values
            y_pred_num = pd.Series(y_pred).astype(str).str.lower().map(mapeo_aux).values

            # Calculamos la métrica
            score = f1_score(y_test_num, y_pred_num, average='macro')

            # Imprimimos el resultado de este modelo concreto
            print(f" Probando: {file:50} | F1-Score: {score:.4f}")

            # Lógica para encontrar al máximo ganador
            if score > best_score:
                best_score = score
                best_model_name = file

    # 3. VEREDICTO FINAL Y GUARDADO
    if best_model_name:
        dest_folder = "modelos_finales"
        os.makedirs(dest_folder, exist_ok=True)

        origen = os.path.join(folder_path, best_model_name)
        destino = os.path.join(dest_folder, f"MEJOR_{method}_{best_model_name}")

        shutil.copy2(origen, destino)

        print("\n" + "⭐ " * 20)
        print(f" 🏆 EL GANADOR ES: {best_model_name}")
        print(f" PUNTUACIÓN MÁXIMA EN TEST: {best_score:.4f}")
        print(f" MODELO ASEGURADO EN: {destino}")
        print("⭐ " * 20 + "\n")
    else:
        print("⚠️ No se encontraron modelos válidos para evaluar.")


if __name__ == "__main__":
    evaluar_y_elegir_mejor()