---
description: Genera automáticamente el notebook de descubrimiento de datos y extracción de Supabase, aplicando auditoría de integridad y descarga en formato crudo.
---

// turbo-all
---
description: generate and maintain the data discovery and extraction notebook
---

# Workflow: Fase 1 - Creación del Notebook de Discovery (Data Extraction)

EXTREMELY IMPORTANT: DO NOT EDIT THE NOTEBOOK DIRECTLY. EDIT THE SCRIPT `scripts/gen_discovery.py` INSTEAD.
Este flujo de trabajo tiene como objetivo generar automáticamente el notebook `notebooks/01_data_discovery.ipynb` mediante un script generador estandarizado que utiliza la lógica de `src/loader.py`.

## 🛠️ Pasos de Ejecución

### Paso 1: Generación del Notebook
Ejecuta el script generador que crea el notebook con la arquitectura de auditoría de 13 puntos y paginación recursiva.

// turbo
```powershell
python scripts/gen_discovery.py
```

### Paso 2: Validación y Ejecución Manual
* **Acción:** Abre y ejecuta manualmente el notebook `notebooks/01_data_discovery.ipynb`.
* **Objetivo:** Verificar la descarga completa de datos de Supabase y la generación del reporte en `experiments/phase_01_discovery/artifacts/`.

### Paso 3: Limpieza de Archivos Temporales
Este paso mantiene el entorno limpio de scripts de ejecución volátiles y logs innecesarios.

// turbo
```powershell
Remove-Item -Path "notebooks/run_*.py", "notebooks/*.log", "notebooks/*.txt" -ErrorAction SilentlyContinue
```

## 📋 Resultado Esperado
1. Notebook actualizado en `notebooks/01_data_discovery.ipynb`.
2. Datos descargados en `data/01_raw/` en formato Parquet.
3. Reporte de auditoría generado en `experiments/phase_01_discovery/artifacts/phase_01_discovery.json`.
