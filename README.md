# Precio de autos usados: experimentación con MLflow

**TC5061 · Operaciones de aprendizaje automático (MLOps)** · Roberto Nájera Lizárraga

Script de entrenamiento parametrizado para estimar el precio de venta de autos usados (`selling_price`) con un `RandomForestRegressor`. Cada corrida queda registrada en MLflow con sus parámetros, métricas, artefactos y el modelo entrenado.

El código parte del notebook baseline de la semana anterior. La exploración y la justificación de la limpieza se quedan en `notebooks/`; la carga, limpieza, entrenamiento y evaluación viven en el paquete `semana_3_refactored_code`.

**Dataset:** *Vehicle dataset* de CarDekho ([Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)), incluido en `data/raw/car_details_v3.csv`.

## Instalación

Requiere Python 3.13. Desde la raíz del proyecto:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

`requirements.txt` fija las versiones exactas con las que se ejecutaron las corridas e instala el paquete en modo editable.

## Ejecución

Todos los comandos se ejecutan **desde la raíz del proyecto**: ahí se crean `mlflow.db` (corridas) y `mlruns/` (artefactos).

```bash
# 1. Limpieza: data/raw/car_details_v3.csv -> data/processed/autos_limpios.csv
python -m semana_3_refactored_code.dataset

# 2. Una corrida con los valores por defecto
python -m semana_3_refactored_code.modeling.train

# 3. Una corrida con otros hiperparámetros
python -m semana_3_refactored_code.modeling.train --n-estimators 600 --max-depth 15 --min-samples-leaf 2 --max-features 0.5 --seed 42

# 4. La serie completa de 10 corridas del experimento
python -m semana_3_refactored_code.modeling.run_experiments

# 5. Interfaz de MLflow en http://127.0.0.1:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

El paso 1 es opcional: si falta el CSV limpio, `train.py` lo genera. Con `make` disponible, los mismos pasos son `make data`, `make train`, `make experiments` y `make mlflow-ui`.

### Argumentos de `train.py`

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--n-estimators` | `300` | Número de árboles del bosque |
| `--max-depth` | `none` | Profundidad máxima de cada árbol (`none` = sin límite) |
| `--min-samples-leaf` | `1` | Mínimo de autos en cada hoja |
| `--max-features` | `1.0` | Variables candidatas por división: fracción, entero, `sqrt` o `log2` |
| `--seed` | `42` | Semilla de la división entrenamiento/prueba y del bosque |
| `--test-size` | `0.2` | Proporción de datos reservada para evaluar |
| `--experiment-name` | `TC5061_precio_autos_random_forest` | Experimento de MLflow |
| `--run-name` | se arma con los hiperparámetros | Nombre de la corrida |
| `--data-path` | `data/processed/autos_limpios.csv` | CSV limpio de entrada |
| `--tracking-uri` | `sqlite:///mlflow.db` | Almacén de MLflow |

`python -m semana_3_refactored_code.modeling.train --help` muestra la misma lista.

## Qué se registra en MLflow

| Tipo | Contenido |
|---|---|
| Parámetros (`log_param`) | Hiperparámetros, semilla, `test_size`, dataset y número de filas de cada conjunto |
| Métricas (`log_metric`) | MAE, RMSE, MAPE (%) y R², en entrenamiento (`train_*`) y en prueba (`test_*`) |
| Artefactos (`log_artifact`) | `metrics.json`, `graficas/prediccion_vs_real.png`, `graficas/importancia_variables.png` |
| Modelo (`mlflow.sklearn.log_model`) | Pipeline completo (imputación + one-hot + bosque) con firma de entrada, en `modelo/` |

El modelo se entrena sobre `log(precio)` con `TransformedTargetRegressor` y predice en rupias, igual que el baseline lineal de la semana anterior.

## Reproducibilidad

- **Semilla:** `--seed` se pasa como `random_state` a `train_test_split` y a `RandomForestRegressor`, y además fija `random` y `numpy`. Dos ejecuciones con los mismos argumentos producen las mismas métricas; la serie de corridas lo comprueba con `referencia` y `referencia_repeticion`.
- **Preprocesamiento dentro del pipeline:** imputación y codificación se ajustan solo con los datos de entrenamiento.
- **Dependencias:** versiones exactas en `requirements.txt`.
- **Datos:** el CSV crudo está versionado; el limpio se regenera de forma determinista con `dataset.py`.

## Estructura

```
├── data/raw/car_details_v3.csv    <- Datos originales (no se modifican)
├── data/processed/                <- CSV limpio generado por dataset.py (no versionado)
├── notebooks/
│   └── 1.0-rn-eda-precio-autos.ipynb   <- Exploración y justificación de la limpieza
├── semana_3_refactored_code/
│   ├── config.py                  <- Rutas del proyecto
│   ├── dataset.py                 <- Carga y limpieza
│   ├── features.py                <- Variables del modelo y preprocesamiento
│   ├── plots.py                   <- Gráficas que se registran como artefactos
│   └── modeling/
│       ├── train.py               <- Entrenamiento parametrizado + MLflow Tracking
│       └── run_experiments.py     <- Serie de corridas del experimento
├── reports/figures/               <- Capturas de MLflow para el reporte
├── requirements.txt               <- Dependencias con versiones exactas
└── pyproject.toml                 <- Metadatos del paquete
```

`mlflow.db` y `mlruns/` no se versionan: se recrean al ejecutar las corridas.
