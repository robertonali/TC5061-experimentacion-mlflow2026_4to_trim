"""Carga y limpieza del dataset de autos usados de CarDekho.

Implementa las decisiones de limpieza justificadas en
notebooks/1.0-rn-eda-precio-autos.ipynb. Uso:

    python -m semana_3_refactored_code.dataset
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from semana_3_refactored_code.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

RUTA_CRUDOS = RAW_DATA_DIR / "car_details_v3.csv"
RUTA_LIMPIOS = PROCESSED_DATA_DIR / "autos_limpios.csv"

COLS_FICHA = ["mileage", "engine", "max_power", "seats"]
KM_MINIMO, KM_MAXIMO = 100, 1_000_000

app = typer.Typer()


def cargar_datos(ruta: Path = RUTA_CRUDOS) -> pd.DataFrame:
    """Lee el CSV original sin modificarlo."""
    df = pd.read_csv(ruta)
    logger.info(f"Datos crudos: {len(df):,} filas y {df.shape[1]} columnas ({ruta.name})")
    return df


def a_numero(serie: pd.Series) -> pd.Series:
    """Convierte textos como '23.4 kmpl' en el número 23.4 (NaN si no hay número)."""
    return pd.to_numeric(serie.str.split().str[0], errors="coerce")


def limpiar_datos(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica las decisiones de limpieza del notebook de exploración, en el mismo orden."""
    df = df_raw.copy()

    # Decisión 1: duplicados exactos, se conserva la primera aparición
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Sin duplicados exactos: {len(df):,} filas")

    # Decisión 2: número + unidad guardado como texto -> número
    for col in ["mileage", "engine", "max_power"]:
        df[col] = a_numero(df[col])

    # Decisión 3: torque es texto libre con dos unidades; se descarta
    df = df.drop(columns="torque")

    # Faltantes ocultos: rendimiento o potencia en 0 son físicamente imposibles
    df["mileage"] = df["mileage"].replace(0, np.nan)
    df["max_power"] = df["max_power"].replace(0, np.nan)

    # Decisión 4: se eliminan las filas a las que les falta toda la ficha técnica
    sin_ficha = df[COLS_FICHA].isna().all(axis=1)
    df = df[~sin_ficha].reset_index(drop=True)
    df["seats"] = df["seats"].astype(int)
    logger.info(f"Sin filas sin ficha técnica: {len(df):,} filas")

    # Decisión 5: los faltantes aislados se quedan y se imputan dentro del pipeline

    # Decisión 6: solo se eliminan los kilometrajes inverosímiles.
    # Decisión 7: los demás valores atípicos (autos de lujo, de 7 asientos) se conservan
    df = df[df["km_driven"].between(KM_MINIMO, KM_MAXIMO)].reset_index(drop=True)
    logger.info(f"Sin kilometrajes inverosímiles: {len(df):,} filas")

    # Decisión 8: name se reemplaza por la marca; se corrigen las marcas de dos palabras
    df["brand"] = df["name"].str.split().str[0].replace(
        {"Land": "Land Rover", "Ashok": "Ashok Leyland"}
    )
    df = df.drop(columns="name")

    logger.info(f"Se conserva el {len(df) / len(df_raw):.1%} de las filas originales")
    return df


def cargar_datos_limpios(ruta: Path = RUTA_LIMPIOS) -> pd.DataFrame:
    """Lee el dataset limpio; si no existe, lo genera a partir de los datos crudos."""
    if not ruta.exists():
        logger.warning(f"No existe {ruta.name}; se genera a partir de los datos crudos")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        limpiar_datos(cargar_datos()).to_csv(ruta, index=False)
    return pd.read_csv(ruta)


@app.command()
def main(
    input_path: Path = RUTA_CRUDOS,
    output_path: Path = RUTA_LIMPIOS,
):
    df = limpiar_datos(cargar_datos(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset limpio guardado en {output_path}")


if __name__ == "__main__":
    app()
