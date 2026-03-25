import sys
import subprocess


def main():
    if len(sys.argv) < 4:
        print("Uso: python main.py <train.csv> <test.csv> <config_file.json>")
        sys.exit(1)

    csv_train = sys.argv[1]
    csv_test = sys.argv[2]
    json_file = sys.argv[3]

    ## FASE DE ENTRENAMIENTO
    # Llamamos a train.py con sus 3 argumentos correspondientes
    subprocess.run(["python", "train.py", csv_train, csv_test, json_file])

    ## FASE DE EVALUACION
    # Llamamos a evaluar.py
    subprocess.run(["python", "evaluar.py", csv_train, csv_test, json_file])

if __name__ == "__main__":
    main()