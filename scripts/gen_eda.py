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
st_res = report['results']['time_series']['stationarity']
print(f"Test de Dickey-Fuller: {'Estacionario' if st_res['is_stationary'] else 'No Estacionario'}")
print(f"P-Value: {st_res['p_value']:.4f}")

print("\\n🔹 IMPACTO DE EVENTOS DE NEGOCIO:")
for event, result in report['results']['business_events'].items():
    print(f"   - {event.upper()}: Impacto de {result['impact_pct']:.1f}%")"""

    # Celda 5: Análisis de Drift
    drift_analysis = """# Celda 5: Análisis de Drift (Train/Val/Test)
print("📑 ANALISIS DE DERIVA DE DATOS (DRIFT):")
target = config['preprocessing']['target_variable']
drift = report['results']['drift_analysis']

print(f"Target: {target}")
for split in ['train', 'val', 'test']:
    print(f"   - {split.upper()}: Mean={drift[split]['mean']:.2f}, Std={drift[split]['std']:.2f}")"""

    # Celda 6: Recomendaciones de Experto
    expert_recs = """# Celda 6: Capa de Inteligencia (Recomendaciones del Experto)
print("🧠 CONCLUSIONES Y MODELADO SUGERIDO:")
recs = report['expert_recommendations']

print("\\n✅ HALLAZGOS:")
for insight in recs['main_insights']:
    print(f"  • {insight}")

print("\\n🛠️ ESTRATEGIA TECNICA:")
print(f"  • Transformación: {recs['technical_diagnostics']['log_transform']}")
print(f"  • Diferenciación: {recs['technical_diagnostics']['stationarity']['action_required']}")
print(f"  • Serie Target Sugerida: {recs['suggested_modeling_strategy']['target_series']}")

print("\\n🚀 CONFIGURACION DE MODELOS (ML):")
print(f"  • Lags Grid: {recs['suggested_modeling_strategy']['lags_grid']}")
print(f"  • Windows Grid: {recs['suggested_modeling_strategy']['windows_grid']}")
print(f"  • Pesos: {recs['suggested_modeling_strategy']['weighting_strategy']}")

print("\\n🧪 VARIABLES EXOGENAS A CONSTRUIR:")
print(f"  • {', '.join(recs['exogenous_to_build'])}")"""

    cells = [
        {"cell_type": "markdown", "source": [
            "# 📈 Fase 03: Exploratory Data Analysis (EDA)\n",
            "Este notebook implementa el análisis exploratorio profundo siguiendo las reglas de la consultora Triple S para Mi Buñuelito.\n",
            "\n",
            "### Pipeline de Análisis (8 Pasos):\n",
            "1. **Temporal Splitting**: Partición determinística Train/Val/Test (Regla 3.1).\n",
            "2. **Profiling & Drift**: Detección de cambios estadísticos entre periodos.\n",
            "3. **Impact of Calendar**: Análisis de estacionalidad mensual y trimestral (Regla 3.3).\n",
            "4. **Time Series Decomposition**: Extracción de Tendencia y Estacionalidad (Regla 3.4).\n",
            "5. **ACF & PACF Analysis**: Determinación de Lags significativos (Regla 3.5).\n",
            "6. **Stationarity Testing**: Validación mediante Augmented Dickey-Fuller (Regla 3.6).\n",
            "7. **Business Events Validation**: Verificación de impacto de promociones y festivos (Regla 3.7).\n",
            "8. **Expert Interpretation**: Capa de inteligencia con recomendaciones automáticas."
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [setup_code]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [data_loading]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [execution_code]},
        {"cell_type": "markdown", "source": ["## 📊 Hallazgos Principales"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [key_results]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [drift_analysis]},
        {"cell_type": "markdown", "source": ["## 🧠 Recomendaciones Estratégicas"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [expert_recs]}
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
