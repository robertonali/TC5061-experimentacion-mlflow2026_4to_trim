"""Entrena un bosque aleatorio para el precio de autos usados y registra la corrida en MLflow.

Ejecutar desde la raíz del proyecto (ahí se crea mlflow.db):

    python -m semana_3_refactored_code.modeling.train
    python -m semana_3_refactored_code.modeling.train --n-estimators 500 --max-depth 20 --seed 7
"""

import argparse
import json
from pathlib import Path
import random
import tempfile

from loguru import logger
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from semana_3_refactored_code import plots
from semana_3_refactored_code.dataset import RUTA_LIMPIOS, cargar_datos_limpios
from semana_3_refactored_code.features import (
    COLS_NUMERICAS,
    construir_preprocesador,
    separar_x_y,
)

EXPERIMENTO = "TC5061_precio_autos_random_forest"
TRACKING_URI = "sqlite:///mlflow.db"  # ruta relativa: evita los espacios de la ruta absoluta


# --------------------------------------------------------------------------------------
# Argumentos de línea de comandos
# --------------------------------------------------------------------------------------
def entero(valor: str, minimo: int) -> int:
    """Convierte a entero y exige un valor mínimo; los errores se muestran en español."""
    try:
        numero = int(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{valor}' no es un número entero") from None
    if numero < minimo:
        raise argparse.ArgumentTypeError(
            f"debe ser un entero mayor o igual a {minimo}, no {numero}")
    return numero


def entero_positivo(valor: str) -> int:
    return entero(valor, minimo=1)


def semilla_valida(valor: str) -> int:
    """sklearn solo acepta semillas no negativas."""
    return entero(valor, minimo=0)


def entero_o_none(valor: str) -> int | None:
    """'none' -> None (árboles sin límite de profundidad); otro valor -> entero positivo."""
    return None if valor.lower() == "none" else entero_positivo(valor)


def max_features_valido(valor: str) -> float | int | str:
    """Acepta 'sqrt', 'log2', una fracción en (0, 1] (0.5) o un número de columnas (10)."""
    if valor in ("sqrt", "log2"):
        return valor
    if "." not in valor:
        return entero_positivo(valor)
    try:
        fraccion = float(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{valor}' no es válido: usa una fracción, un entero, sqrt o log2") from None
    if not 0 < fraccion <= 1:
        raise argparse.ArgumentTypeError(f"la fracción debe estar en (0, 1], no {fraccion}")
    return fraccion


def proporcion(valor: str) -> float:
    """Fracción estrictamente entre 0 y 1."""
    try:
        numero = float(valor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{valor}' no es un número") from None
    if not 0 < numero < 1:
        raise argparse.ArgumentTypeError(f"debe estar entre 0 y 1, no {numero}")
    return numero


class AyudaConDefaults(argparse.ArgumentDefaultsHelpFormatter):
    """Agrega '(default: ...)' a la ayuda, salvo cuando el default es None."""

    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def leer_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m semana_3_refactored_code.modeling.train",
        description="Entrena un RandomForestRegressor sobre log(precio) y lo registra en MLflow.",
        formatter_class=AyudaConDefaults,
    )
    modelo = parser.add_argument_group("hiperparámetros del modelo")
    modelo.add_argument("--n-estimators", metavar="N", type=entero_positivo, default=300,
                        help="número de árboles del bosque")
    # Default como texto: argparse lo pasa por entero_o_none y la ayuda muestra 'none'
    modelo.add_argument("--max-depth", metavar="PROF", type=entero_o_none, default="none",
                        help="profundidad máxima de cada árbol ('none' = sin límite)")
    modelo.add_argument("--min-samples-leaf", metavar="N", type=entero_positivo, default=1,
                        help="mínimo de autos en cada hoja")
    modelo.add_argument("--max-features", metavar="VALOR", type=max_features_valido, default=1.0,
                        help="variables candidatas en cada división: "
                             "fracción, entero, sqrt o log2")

    corrida = parser.add_argument_group("datos y corrida")
    corrida.add_argument("--seed", metavar="N", type=semilla_valida, default=42,
                         help="semilla de la división y del bosque")
    corrida.add_argument("--test-size", metavar="FRAC", type=proporcion, default=0.2,
                         help="proporción de los datos reservada para evaluar")
    corrida.add_argument("--data-path", metavar="RUTA", type=Path, default=None,
                         help="CSV limpio generado por semana_3_refactored_code.dataset "
                              "(default: data/processed/autos_limpios.csv)")
    corrida.add_argument("--experiment-name", metavar="NOMBRE", default=EXPERIMENTO,
                         help="experimento de MLflow donde se agrupan las corridas")
    corrida.add_argument("--run-name", metavar="NOMBRE", default=None,
                         help="nombre de la corrida (por defecto se arma con los hiperparámetros)")
    corrida.add_argument("--tracking-uri", metavar="URI", default=TRACKING_URI,
                         help="almacén de MLflow")
    args = parser.parse_args(argv)
    # La ruta por defecto se resuelve aquí para que la ayuda no muestre la ruta absoluta
    args.data_path = args.data_path or RUTA_LIMPIOS
    return args


# --------------------------------------------------------------------------------------
# Datos, modelo y evaluación
# --------------------------------------------------------------------------------------
def fijar_semilla(seed: int) -> None:
    """Fija las fuentes de aleatoriedad globales; sklearn recibe la semilla explícitamente."""
    random.seed(seed)
    np.random.seed(seed)


def dividir_datos(df: pd.DataFrame, test_size: float, seed: int):
    """División entrenamiento/prueba con semilla fija."""
    X, y = separar_x_y(df)
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def construir_modelo(args: argparse.Namespace) -> TransformedTargetRegressor:
    """Preprocesamiento + bosque aleatorio, entrenado sobre log(precio).

    TransformedTargetRegressor ajusta con log(precio) y devuelve la predicción en rupias,
    igual que el modelo base de la semana anterior.
    """
    bosque = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed,
        n_jobs=-1,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", construir_preprocesador()), ("rf", bosque)]),
        func=np.log,
        inverse_func=np.exp,
    )


def evaluar(y: pd.Series, pred: np.ndarray) -> dict[str, float]:
    """Métricas de regresión sobre el precio en rupias."""
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAPE_pct": mean_absolute_percentage_error(y, pred) * 100,
        "R2": r2_score(y, pred),
    }


def importancias(modelo: TransformedTargetRegressor) -> pd.Series:
    """Importancia de cada columna después de codificar."""
    pipeline = modelo.regressor_
    nombres = pipeline.named_steps["prep"].get_feature_names_out()
    return pd.Series(pipeline.named_steps["rf"].feature_importances_, index=nombres)


# --------------------------------------------------------------------------------------
# Registro en MLflow
# --------------------------------------------------------------------------------------
def nombre_corrida(args: argparse.Namespace) -> str:
    return (f"rf_n{args.n_estimators}_depth{args.max_depth}_leaf{args.min_samples_leaf}"
            f"_mf{args.max_features}_seed{args.seed}")


def registrar_en_mlflow(args, modelo, df, X_train, X_test, y_test, pred_test, metricas) -> str:
    """Registra parámetros, métricas, artefactos y modelo en una corrida de MLflow."""
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name or nombre_corrida(args)) as corrida:
        # Parámetros: hiperparámetros y lo necesario para repetir la corrida
        parametros = {
            "modelo": "RandomForestRegressor",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "seed": args.seed,
            "test_size": args.test_size,
            "transformacion_objetivo": "log",
            "dataset": args.data_path.name,
            "filas_limpias": len(df),
            "filas_entrenamiento": len(X_train),
            "filas_prueba": len(X_test),
        }
        for nombre, valor in parametros.items():
            mlflow.log_param(nombre, valor)

        # Métricas de entrenamiento y de prueba; la brecha entre ambas mide el sobreajuste
        for conjunto, valores in metricas.items():
            for nombre, valor in valores.items():
                mlflow.log_metric(f"{conjunto}_{nombre}", valor)

        # Artefactos: métricas en JSON y gráficas de evaluación
        with tempfile.TemporaryDirectory() as carpeta:
            carpeta = Path(carpeta)
            ruta_json = carpeta / "metrics.json"
            ruta_json.write_text(json.dumps({"parametros": parametros, "metricas": metricas},
                                            indent=2, ensure_ascii=False, default=str),
                                 encoding="utf-8")
            mlflow.log_artifact(str(ruta_json))

            mlflow.log_artifact(str(plots.grafica_prediccion_vs_real(
                y_test, pred_test, carpeta / "prediccion_vs_real.png")), artifact_path="graficas")
            mlflow.log_artifact(str(plots.grafica_importancias(
                importancias(modelo), carpeta / "importancia_variables.png")),
                artifact_path="graficas")

        # Modelo completo (preprocesamiento + bosque), con firma de entrada y salida
        # Las numéricas se declaran float: un entero no admite faltantes al predecir
        ejemplo = X_train.head().astype({col: "float64" for col in COLS_NUMERICAS})
        firma = infer_signature(ejemplo, modelo.predict(ejemplo))
        # MLflow 3 guarda en formato skops, que solo carga tipos declarados como confiables;
        # el imputador guarda un numpy.dtype
        mlflow.sklearn.log_model(modelo, name="modelo", signature=firma,
                                 skops_trusted_types=["numpy.dtype"])
        return corrida.info.run_id


def main(argv: list[str] | None = None) -> None:
    args = leer_argumentos(argv)
    fijar_semilla(args.seed)
    logger.info(f"Argumentos: {vars(args)}")

    df = cargar_datos_limpios(args.data_path)
    X_train, X_test, y_train, y_test = dividir_datos(df, args.test_size, args.seed)
    logger.info(f"Entrenamiento: {len(X_train):,} filas | Prueba: {len(X_test):,} filas")

    modelo = construir_modelo(args).fit(X_train, y_train)
    pred_test = modelo.predict(X_test)
    metricas = {
        "train": evaluar(y_train, modelo.predict(X_train)),
        "test": evaluar(y_test, pred_test),
    }
    for conjunto, valores in metricas.items():
        logger.info(f"{conjunto}: " + " | ".join(f"{k}={v:,.4f}" for k, v in valores.items()))

    run_id = registrar_en_mlflow(args, modelo, df, X_train, X_test, y_test, pred_test, metricas)
    logger.success(f"Corrida registrada en '{args.experiment_name}' con run_id={run_id}")


if __name__ == "__main__":
    main()
