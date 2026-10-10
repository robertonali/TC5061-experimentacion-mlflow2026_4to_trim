"""Lanza la serie de corridas del experimento, cada una como un proceso independiente de train.py.

Cada configuración cambia un solo hiperparámetro respecto a la de referencia, para poder
atribuir la diferencia en las métricas a ese hiperparámetro. La última corrida repite la
de referencia para comprobar que la semilla hace la corrida reproducible.

Ejecutar desde la raíz del proyecto:

    python -m semana_3_refactored_code.modeling.run_experiments
"""

import subprocess
import sys

from loguru import logger

# Referencia: valores por defecto de train.py
CORRIDAS = [
    ("referencia", []),
    ("n_estimators_50", ["--n-estimators", "50"]),
    ("n_estimators_600", ["--n-estimators", "600"]),
    ("max_depth_8", ["--max-depth", "8"]),
    ("max_depth_15", ["--max-depth", "15"]),
    ("min_samples_leaf_5", ["--min-samples-leaf", "5"]),
    ("min_samples_leaf_20", ["--min-samples-leaf", "20"]),
    ("max_features_sqrt", ["--max-features", "sqrt"]),
    ("max_features_0.5", ["--max-features", "0.5"]),
    ("referencia_repeticion", []),
]


def main() -> None:
    for i, (nombre, argumentos) in enumerate(CORRIDAS, start=1):
        comando = [sys.executable, "-m", "semana_3_refactored_code.modeling.train",
                   *argumentos, "--run-name", nombre]
        logger.info(f"[{i}/{len(CORRIDAS)}] python -m semana_3_refactored_code.modeling.train "
                    + " ".join(comando[3:]))
        subprocess.run(comando, check=True)
    logger.success(f"{len(CORRIDAS)} corridas registradas. Abre la interfaz con: "
                   "mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
