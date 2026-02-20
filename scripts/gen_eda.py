import os
import json

def generate_eda_notebook(output_path="notebooks/03_eda.ipynb"):
    """
    Genera el notebook de Análisis Exploratorio de Datos (Fase 03) basado en el DataExplorer.
    """
    
    # Celda 1: Configuración
    setup_code = """# Celda 1: Setup
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar 'src' al path
sys.path.insert(0, os.path.abspath('..'))

from src.utils.config_loader import load_config
from src.explorer import DataExplorer

config = load_config("../config.yaml")

# Redirigir reportes y figuras a la carpeta de experimentos si estamos en el notebook (Lab Mode)
config['general']['paths']['reports'] = os.path.join("..", config['general']['paths']['experiments']['phase_03'], "artifacts")
config['general']['paths']['figures'] = os.path.join("..", config['general']['paths']['experiments']['phase_03'], "figures")

# Configurar el explorer para usar las rutas de laboratorio
explorer = DataExplorer(config)
# Forzar la ruta de figuras específica para la fase en el explorer
explorer.figures_path = config['general']['paths']['figures']

print(f"✅ Ambiente de EDA configurado (Modo Laboratorio).")
print(f"📂 Los reportes se guardarán en: {config['general']['paths']['reports']}")
print(f"🖼️ Las figuras se guardarán en: {config['general']['paths']['figures']}")"""

    # Celda 2: Carga del Master Dataset
    data_loading = """# Celda 2: Carga del Master Cleansed Dataset
cleansed_path = os.path.join("..", config["general"]["paths"]["cleansed"], "master_cleansed.parquet")

if os.path.exists(cleansed_path):
    df = pd.read_parquet(cleansed_path)
    print(f"✅ Dataset maestro cargado: {df.shape}")
    display(df.head())
else:
    raise FileNotFoundError(f"❌ No se encontró el archivo maestro en {cleansed_path}. Ejecuta la fase de preprocesamiento primero.")"""

    # Celda 3: Ejecución del EDA
    execution_code = """# Celda 3: Ejecución del Pipeline de EDA
# El DataExplorer ejecuta automáticamente los 7 pasos definidos en el protocolo
report = explorer.run_eda(df)

print(f"\\n✅ EDA completado. Status: {report['status']}")"""

    # Celda 4: Visualización de Resultados Clave
    key_results = """# Celda 4: Resumen Estadístico y Estacionariedad
print("📊 RESULTADOS CLAVE DEL EDA:")
print(f"Status: {report['status']}")
print(f"Test de Dickey-Fuller: {report['results']['time_series']['stationarity']['verdict']}")
print(f"P-Value: {report['results']['time_series']['stationarity']['p_value']:.4f}")

print("\\n🔹 IMPACTO DE EVENTOS DE NEGOCIO:")
for event, result in report['results']['business_events'].items():
    print(f"   - {event.upper()}: {result['conclusion']}")"""

    # Celda 5: Análisis de Drift
    drift_analysis = """# Celda 5: Análisis de Drift (Train vs Test)
print("📑 ANALISIS DE DERIVA DE DATOS (DRIFT):")
target = config['preprocessing']['target_variable']
drift = report['results']['drift_analysis'][target]

print(f"Target: {target}")
print(f"Mean (Train): {drift['train']['mean']:.2f} vs Mean (Test): {drift['test']['mean']:.2f}")
print(f"Range (Train): [{drift['train']['min']:.2f}, {drift['train']['max']:.2f}]")
print(f"Range (Test): [{drift['test']['min']:.2f}, {drift['test']['max']:.2f}]")"""

    cells = [
        {"cell_type": "markdown", "source": [
            "# 📈 Fase 03: Exploratory Data Analysis (EDA)\n",
            "Este notebook implementa el análisis exploratorio profundo siguiendo las reglas de la consultora Triple S para Mi Buñuelito.\n",
            "\n",
            "### Pipeline de Análisis (7 Pasos):\n",
            "1. **Temporal Splitting**: Partición determinística Train/Val/Test (Regla 3.1).\n",
            "2. **Profiling & Drift**: Detección de cambios estadísticos entre periodos.\n",
            "3. **Impact of Calendar**: Análisis de estacionalidad mensual y trimestral (Regla 3.3).\n",
            "4. **Time Series Decomposition**: Extracción de Tendencia y Estacionalidad (Regla 3.4).\n",
            "5. **ACF & PACF Analysis**: Determinación de Lags significativos (Regla 3.5).\n",
            "6. **Stationarity Testing**: Validación mediante Augmented Dickey-Fuller (Regla 3.6).\n",
            "7. **Business Events Validation**: Verificación de impacto de promociones y festivos (Regla 3.7)."
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [setup_code]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [data_loading]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [execution_code]},
        {"cell_type": "markdown", "source": ["## 📊 Hallazgos Principales"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [key_results]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [drift_analysis]}
    ]

    notebook = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)
    
    print(f"✅ Notebook de EDA generado en: {output_path}")

if __name__ == "__main__":
    generate_eda_notebook()
