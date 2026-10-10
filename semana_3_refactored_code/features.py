"""Variables del modelo y preprocesamiento.

El preprocesamiento se devuelve sin ajustar para que viva dentro del pipeline del
modelo y aprenda solo con los datos de entrenamiento.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

OBJETIVO = "selling_price"
COLS_NUMERICAS = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
COLS_CATEGORICAS = ["fuel", "seller_type", "transmission", "owner", "brand"]


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las variables de entrada y el precio."""
    return df[COLS_NUMERICAS + COLS_CATEGORICAS], df[OBJETIVO]


def construir_preprocesador() -> ColumnTransformer:
    """Imputa las numéricas con la mediana y codifica las categóricas en columnas 0/1.

    No se escalan las numéricas: un bosque aleatorio parte cada variable por umbrales,
    así que la escala no cambia el resultado.
    """
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), COLS_NUMERICAS),
            # Una categoría nunca vista en entrenamiento se ignora
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLS_CATEGORICAS),
        ]
    )
