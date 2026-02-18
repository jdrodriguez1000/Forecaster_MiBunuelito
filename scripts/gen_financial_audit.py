import os

def generate_financial_audit_notebook(output_path="notebooks/02_financial_audit.ipynb"):
    """
    Genera el notebook de Auditoría Financiera (Fase 02).
    """
    import json

    # Configuración del notebook
    setup_code = """# Celda 1: Configuración de Ambiente
import sys
import os
import pandas as pd
import json

# Agregar 'src' al path
sys.path.insert(0, os.path.abspath('..'))

from src.utils.encoding_patch import apply_utf8_patch
from src.loader import DataLoader
from src.validator import BusinessValidator

# Asegurar codificación UTF-8
apply_utf8_patch()

print("✅ Ambiente de Auditoría Financiera configurado.")"""

    data_loading = """# Celda 2: Carga de Datos Locales
# Cargamos los archivos Parquet que ya fueron descargados por el DataLoader
loader = DataLoader()
tables_to_audit = ["ventas_diarias", "redes_sociales"]
data = {}

for table in tables_to_audit:
    file_path = os.path.join("..", loader.config["general"]["paths"]["raw"], f"{table}.parquet")
    if os.path.exists(file_path):
        data[table] = pd.read_parquet(file_path)
        print(f"✅ Cargada tabla: {table} ({len(data[table])} registros)")
    else:
        print(f"❌ No se encontró el archivo para: {table}")"""

    audit_execution = """# Celda 3: Ejecución de Auditoría de Negocio
validator = BusinessValidator()
financial_report = validator.validate_all(data)

print(f"✅ Auditoría completada el {financial_report['timestamp']}")"""

    results_sales = """# Celda 4: Resultados - Ventas Diarias
v_audit = financial_report["tables"].get("ventas_diarias", {})
if v_audit:
    print("📊 RESULTADOS AUDITORÍA FINANCIERA (VENTAS DIARIAS):")
    for rule, res in v_audit.items():
        status = "✅ CUMPLE" if res["success"] else "❌ FALLA"
        print(f"{status} - {rule}")
        if not res["success"]:
            print(f"   -> Errores detectados: {res.get('errors_count', res.get('violations_count', 0))}")"""

    results_redes = """# Celda 5: Resultados - Redes Sociales
r_audit = financial_report["tables"].get("redes_sociales", {})
if r_audit:
    print("📊 RESULTADOS AUDITORÍA (REDES SOCIALES):")
    for rule, res in r_audit.items():
        status = "✅ CUMPLE" if res["success"] else "❌ FALLA"
        print(f"{status} - {rule}")"""

    persistence = """# Celda 6: Persistencia del Reporte (Fase 02)
output_dir = os.path.join("..", validator.reports_path)
os.makedirs(output_dir, exist_ok=True)
report_file = os.path.join(output_dir, "phase_02_financial_audit.json")

with open(report_file, "w", encoding="utf-8") as f:
    json.dump(financial_report, f, indent=4, ensure_ascii=False)

print(f"✅ Reporte financiero guardado en: {report_file}")"""

    cells = [
        {"cell_type": "markdown", "source": ["# 💰 Fase 02: Financial & Business Logic Audit\n", "Este notebook valida la consistencia de los datos contables y de inversión utilizando las reglas definidas en `config.yaml`."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [setup_code]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [data_loading]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [audit_execution]},
        {"cell_type": "markdown", "source": ["## Hallazgos de Consistencia"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [results_sales]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [results_redes]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [persistence]}
    ]

    notebook = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)
    
    print(f"Notebook de auditoría financiera generado en: {output_path}")

if __name__ == "__main__":
    generate_financial_audit_notebook()
