"""Gráficas de evaluación que se guardan como archivo y se registran como artefactos."""

from pathlib import Path

# Figure en lugar de pyplot: no abre ventanas ni cambia el backend de quien importe el módulo
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import pandas as pd

COLOR = "#2a78d6"


def formato_miles(valor, _=None):
    """Rotula ejes con números grandes: 450000 -> '450 mil', 2500000 -> '2.5 M'."""
    if valor == 0:
        return "0"
    return f"{valor / 1e6:g} M" if abs(valor) >= 1e6 else f"{valor / 1e3:g} mil"


def grafica_prediccion_vs_real(y_real, y_pred, ruta: Path) -> Path:
    """Dispersión de precio predicho contra precio real; sobre la diagonal la predicción es exacta."""
    fig = Figure(figsize=(6.5, 6))
    eje = fig.subplots()
    eje.scatter(y_real, y_pred, s=10, alpha=0.35, color=COLOR, linewidths=0)
    limites = [min(y_real) * 0.8, max(y_real) * 1.2]
    eje.plot(limites, limites, color="#52514e", linestyle="--", linewidth=1.5,
             label="Predicción exacta")
    eje.set_xscale("log")
    eje.set_yscale("log")
    eje.xaxis.set_major_formatter(FuncFormatter(formato_miles))
    eje.yaxis.set_major_formatter(FuncFormatter(formato_miles))
    eje.set(title="Precio predicho contra precio real (prueba)",
            xlabel="Precio real (INR)", ylabel="Precio predicho (INR)")
    eje.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    return ruta


def grafica_importancias(importancias: pd.Series, ruta: Path, top: int = 15) -> Path:
    """Barras horizontales con las variables más importantes del bosque aleatorio."""
    datos = importancias.sort_values().tail(top)
    # Quita el prefijo del ColumnTransformer: 'cat__brand_Toyota' -> 'brand_Toyota'
    datos.index = datos.index.str.replace(r"^(num|cat)__", "", regex=True)
    fig = Figure(figsize=(7, 0.35 * len(datos) + 1.2))
    eje = fig.subplots()
    eje.barh(datos.index, datos.values, color=COLOR)
    eje.set(title=f"Importancia de variables (top {len(datos)})",
            xlabel="Reducción media de impureza", ylabel="")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    return ruta
