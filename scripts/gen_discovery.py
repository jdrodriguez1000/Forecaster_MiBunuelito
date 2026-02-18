import nbformat as nbf
import os

def generate_discovery_notebook():
    nb = nbf.v4.new_notebook()
    
    # 1. Title and Documentation
    title_md = """# 📊 Fase 01: Data Discovery & Extraction
Este notebook se encarga de la extracción de datos desde Supabase utilizando la API REST y realiza una auditoría detallada de 13 puntos de calidad."""
    
    # 2. Environment Setup
    setup_code = """# Celda 1: Importacion de librerias y Parche de Codificacion Inteligente
import sys
import os
import builtins
import json
import pandas as pd

# Agregar la carpeta 'src' al path
sys.path.insert(0, os.path.abspath('..'))

from src.utils.encoding_patch import apply_utf8_patch
from src.loader import DataLoader

# Aplicar parche de forma segura
apply_utf8_patch()

print("✅ Ambiente configurado correctamente y parche de codificación aplicado.")"""

    # 3. Data Loading
    loading_code = """# Celda 2: Ejecucion de Auditoria General
# El objeto 'report' contiene los 13 puntos de auditoria para todas las tablas
# DataLoader utiliza automaticamente las rutas definidas en config.yaml
loader = DataLoader()
report = loader.load_and_audit()

print(f"✅ ¡Extraccion completada! Se procesaron {len(report['tables'])} tablas.")"""

    # 4. Section 1: Structural Integrity
    struct_md = "## 1. Integridad Estructural y Contrato\nDiagnóstico sobre el cumplimiento del esquema definido y dimensiones de las tablas."
    struct_code = """# Celda 3: Auditoria de Estructura (Puntos 1.1 al 1.3)
for table, audit in report["tables"].items():
    print(f"TABLA: {table.upper()}")
    s = audit["structural_integrity"]
    print(f"  - Contrato Cumplido: {'✅' if s['contract_fulfilled'] else '❌'}")
    print(f"  - Dimensiones: {s['shape']['rows']} filas x {s['shape']['columns']} columnas")
    if s['additional_columns']:
        print(f"  - Columnas extra detectadas: {s['additional_columns']}")
    print("-" * 30)"""

    # 5. Section 2: Data Quality
    quality_md = "## 2. Calidad de Datos y Limpieza\nIdentificación de vacíos, duplicados y valores centinela."
    quality_code = """# Celda 4: Auditoria de Calidad (Puntos 2.1 al 2.6)
for table, audit in report["tables"].items():
    print(f"TABLA: {table.upper()}")
    q = audit["data_quality"]
    print(f"  - Filas duplicadas: {q['duplicate_rows_count']}")
    print(f"  - Fechas duplicadas (con datos unicos): {q['duplicate_dates_count']}")
    
    # Nulos
    nulls = {k: v for k, v in q['null_counts'].items() if v > 0}
    print(f"  - Columnas con nulos: {nulls if nulls else 'Ninguna'}")
    
    # Centinelas
    sentinels = {k: v for k, v in q['sentinel_counts'].items() if v > 0}
    print(f"  - Valores centinela (atpicos de sistema): {sentinels if sentinels else 'Ninguno'}")
    
    if q['zero_variance_cols']:
        print(f"  - Columnas sin varianza (constantes): {q['zero_variance_cols']}")
    
    print("-" * 30)"""

    # 6. Section 3: Time Series
    ts_md = "## 3. Salud de la Serie de Tiempo\nAnálisis de la continuidad temporal y rangos de fechas."
    ts_code = """# Celda 5: Auditoria Temporal (Puntos 3.1 al 3.2)
for table, audit in report["tables"].items():
    print(f"TABLA: {table.upper()}")
    t = audit["time_series_health"]
    if "date_range" in t:
        print(f"  - Rango: {t['date_range']['min']} hasta {t['date_range']['max']}")
        print(f"  - Huecos detectados (Gaps): {'❌ ' + str(t['gaps_detected_count']) if t['has_gaps'] else '✅ Ninguno'}")
    else:
        print("  - No se detecto columna de fecha para esta tabla.")
    print("-" * 30)"""

    # 7. Section 4: Statistical Profiling
    stat_md = "## 4. Perfilamiento Estadistico y Anomalias\nResumen estadístico y detección de outliers técnicos."
    stat_code = """# Celda 6: Perfilamiento y Outliers (Puntos 4.1 al 4.3)
for table, audit in report["tables"].items():
    print(f"TABLA: {table.upper()}")
    st = audit["statistical_profiling"]
    
    print("  - Resumen de Outliers (IQR):")
    for col, prof in st["numeric_profile"].items():
        count = prof['outliers']['count']
        status = "⚠️" if count > 0 else "✅"
        print(f"    {status} {col}: {count} outliers")
        
    print("-" * 30)"""

    # 8. Persistence (Rule 6.68)
    persistence_md = "## 5. Persistencia del Reporte (Regla 6.68)\nGuardado del reporte de auditoría en la carpeta de experimentos para trazabilidad."
    persistence_code = """# Celda 7: Guardado del reporte oficial (Lab Environment)
from src.utils.helpers import save_report

output_dir = os.path.join("..", "experiments", "phase_01_discovery", "artifacts")
save_report(report, output_dir, "phase_01_discovery")

print(f"✅ Reportes (histórico y latest) guardados en: {output_dir}")"""

    nb.cells = [
        nbf.v4.new_markdown_cell(title_md),
        nbf.v4.new_code_cell(setup_code),
        nbf.v4.new_code_cell(loading_code),
        nbf.v4.new_markdown_cell(struct_md),
        nbf.v4.new_code_cell(struct_code),
        nbf.v4.new_markdown_cell(quality_md),
        nbf.v4.new_code_cell(quality_code),
        nbf.v4.new_markdown_cell(ts_md),
        nbf.v4.new_code_cell(ts_code),
        nbf.v4.new_markdown_cell(stat_md),
        nbf.v4.new_code_cell(stat_code),
        nbf.v4.new_markdown_cell(persistence_md),
        nbf.v4.new_code_cell(persistence_code),
    ]

    notebook_path = os.path.join("notebooks", "01_data_discovery.ipynb")
    os.makedirs("notebooks", exist_ok=True)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook granular generado exitosamente.")

if __name__ == "__main__":
    generate_discovery_notebook()
