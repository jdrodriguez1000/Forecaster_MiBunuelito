import os
import json

def generate_preprocessing_notebook(output_path="notebooks/02_preprocessing.ipynb"):
    """
    Genera el notebook de Preprocesamiento (Fase 02) basado en el protocolo de 11 pasos.
    """
    
    # Celda 1: Configuración
    setup_code = """# Celda 1: Setup
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar 'src' al path
sys.path.insert(0, os.path.abspath('..'))

from src.utils.config_loader import load_config
from src.loader import DataLoader
from src.preprocessor import Preprocessor

config = load_config("../config.yaml")

# Redirigir reportes a la carpeta de experimentos si estamos en el notebook (Lab Mode)
config['general']['paths']['reports'] = os.path.join("..", config['general']['paths']['experiments']['phase_02'])

# Configuración de visualización
plt.style.use('ggplot')
%matplotlib inline

print(f"✅ Ambiente de Preprocesamiento configurado (Modo Laboratorio).")
print(f"📂 Los reportes se guardarán en: {config['general']['paths']['reports']}")"""

    # Celda 2: Carga de Datos
    data_loading = """# Celda 2: Carga de Datos Raw
loader = DataLoader()
tables = ["ventas_diarias", "macro_economia", "promocion_dia", "redes_sociales"]
raw_data = {}

for table in tables:
    file_path = os.path.join("..", loader.config["general"]["paths"]["raw"], f"{table}.parquet")
    if os.path.exists(file_path):
        raw_data[table] = pd.read_parquet(file_path)
        print(f"✅ Cargada {table}: {raw_data[table].shape}")
    else:
        print(f"⚠️ No se encontró {table}")"""

    # Celda 3: Ejecución del Preprocesador
    execution_code = """# Celda 3: Ejecución del Core Preprocessor (11 Pasos)
preprocessor = Preprocessor(config)
master_df = preprocessor.process(raw_data)

print(f"\\n✅ Preprocesamiento completado. Master dataset: {master_df.shape}")"""

    # Celda 4: Resumen del Reporte
    report_summary = """# Celda 4: Trazabilidad del Proceso
print("📊 DETALLE DE PASOS EJECUTADOS:")
for step, detail in preprocessor.report["steps_detail"].items():
    print(f"\\n🔹 {step.upper()}:")
    if isinstance(detail, list):
        for item in detail:
            print(f"   - {item}")
    else:
        print(f"   - {detail}")"""

    # Celda 5: Análisis del Master Dataset
    master_analysis = """# Celda 5: Perfil del Dataset Maestro
print(f"Dataset Index: {master_df.index.name}")
print(f"Columnas finales: {list(master_df.columns)}")
display(master_df.describe())
display(master_df.head())"""

    # Celda 6: Visualización de la Serie de Tiempo
    visualization = """# Celda 6: Visualización del Target
target_col = config['preprocessing']['target_variable']
plt.figure(figsize=(15, 6))
plt.plot(master_df.index, master_df[target_col], marker='o', linestyle='-', color='darkblue')
plt.title(f"Evolución Mensual de {target_col.replace('_', ' ').title()}")
plt.xlabel("Fecha")
plt.ylabel("Unidades")
plt.grid(True)
plt.show()"""

    cells = [
        {"cell_type": "markdown", "source": [
            "# 🔧 Fase 02: Data Preprocessing & Master Construction\n",
            "Este notebook implementa el protocolo de **11 pasos** para transformar los datos raw provenientes de Supabase en un dataset maestro listo para EDA y Modelamiento.\n",
            "\n",
            "### Protocolo de 11 Pasos:\n",
            "1. **Estandarización de Contratos**: Validación de esquemas.\n",
            "2. **Limpieza Estructural**: Eliminación de IDs y columnas técnicas.\n",
            "3. **Temporal Shielding**: Aplicación de Anti-Leakage (vallas temporales).\n",
            "4. **Gestión de Duplicados**: Estrategia de `keep_last`.\n",
            "5. **Transformación de Centinelas**: Conversión de códigos de error a Nulos.\n",
            "6. **Reindexación de Gaps**: Asegurar continuidad temporal.\n",
            "7. **Protocolos de Imputación**: Ffill, Bfill y constantes pro-negocio.\n",
            "8. **Recálculo de la Verdad**: Reconstrucción de variables financieras/unidades.\n",
            "9. **Cleanup de Columnas de Soporte**: Eliminación de auxiliares anti-pereza.\n",
            "10. **Agregación y Cruce**: Consolidación mensual (Monthly Resampling).\n",
            "11. **Generación de Artefactos**: Exportación Parquet y Reporte JSON de Trazabilidad."
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [setup_code]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [data_loading]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [execution_code]},
        {"cell_type": "markdown", "source": ["## 📈 Auditoría de Trazabilidad"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [report_summary]},
        {"cell_type": "markdown", "source": ["## 🏁 Dataset Final"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [master_analysis]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [visualization]}
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
    
    print(f"✅ Notebook de preprocesamiento generado en: {output_path}")

if __name__ == "__main__":
    generate_preprocessing_notebook()
