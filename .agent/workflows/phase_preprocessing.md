---
description: generate and maintain the data preprocessing and master construction notebook
---

# Workflow: Fase 2 - Creación del Notebook de Preprocesamiento

EXTREMELY IMPORTANT: DO NOT EDIT THE NOTEBOOK DIRECTLY. EDIT THE SCRIPT `scripts/gen_preprocessing.py` INSTEAD.
Este flujo de trabajo tiene como objetivo generar automáticamente el notebook `notebooks/02_preprocessing.ipynb` mediante un script generador estandarizado que utiliza la lógica de `src/preprocessor.py`.

## 🛠️ Pasos de Ejecución

### Paso 1: Generación del Notebook
Ejecuta el script generador que crea el notebook con el protocolo de 11 pasos y configuración de modo laboratorio.

// turbo
```powershell
python scripts/gen_preprocessing.py
```

### Paso 2: Validación y Ejecución Manual
* **Acción:** Abre y ejecuta manualmente el notebook `notebooks/02_preprocessing.ipynb`.
* **Objetivo:** Verificar la generación del master dataset en `data/02_cleansed/` y el reporte de trazabilidad en `experiments/phase_02_preprocessing/artifacts/`.

### Paso 3: Limpieza de Archivos Temporales
Este paso mantiene el entorno limpio de scripts de ejecución volátiles y logs innecesarios.

// turbo
```powershell
Remove-Item -Path "notebooks/run_*.py", "notebooks/*.log", "notebooks/*.txt" -ErrorAction SilentlyContinue
```

## 📋 Resultado Esperado
1. Notebook actualizado en `notebooks/02_preprocessing.ipynb`.
2. Master dataset generado en `data/02_cleansed/master_cleansed.parquet`.
3. Reporte de trazabilidad generado en `experiments/phase_02_preprocessing/artifacts/phase_02_preprocessing.json`.
