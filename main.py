import sys
import json
import subprocess


def main():
    if len(sys.argv) < 3:
        print("Uso: python main.py <datos.csv> <config_file.json>")
        sys.exit(1)

    csv_file = sys.argv[1]
    json_file = sys.argv[2]

    # Leemos el JSON para saber qué método usar
    with open(json_file, 'r') as f:
        config = json.load(f)

    metodo = config.get("method", "knn")  # Por defecto knn

    if metodo == "knn":
        print("🚀 Lanzando experimento kNN...")
        # Llamamos al script de knn pasando los archivos
        subprocess.run(["python", "knn.py", csv_file, json_file])

    elif metodo == "arbol":
        print("🌳 Lanzando experimento Árbol de Decisión...")
        subprocess.run(["python", "arbol.py", csv_file, json_file])

    else:
        print(f"❌ Método {metodo} no reconocido.")


if __name__ == "__main__":
    main()