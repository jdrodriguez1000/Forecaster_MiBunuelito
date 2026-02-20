---
description: generate and maintain the exploratory data analysis notebook
---

// turbo-all
# Workflow: Fase 3 - Creación del Notebook de EDA (Exploratory Data Analysis)

EXTREMELY IMPORTANT: DO NOT EDIT THE NOTEBOOK DIRECTLY. EDIT THE SCRIPT `scripts/gen_eda.py` INSTEAD.
Este flujo de trabajo tiene como objetivo generar automáticamente el notebook `notebooks/03_eda.ipynb` mediante un script generador estandarizado que utiliza la lógica de `src/explorer.py`.

## 🛠️ Pasos de Ejecución

### Paso 1: Generación del Notebook
Ejecuta el script generador que crea el notebook con el pipeline de 7 pasos y configuración de modo laboratorio.

// turbo
```powershell
python scripts/gen_eda.py
```

### Paso 2: Validación y Ejecución Manual
* **Acción:** Abre y ejecuta manualmente el notebook `notebooks/03_eda.ipynb`.
* **Objetivo:** Verificar la generación de visualizaciones en `experiments/phase_03_eda/figures/` y el reporte de trazabilidad en `experiments/phase_03_eda/artifacts/phase_03_eda.json`.

### Paso 3: Limpieza de Archivos Temporales
Este paso mantiene el entorno limpio de scripts de ejecución volátiles y logs innecesarios.

// turbo
```powershell
Remove-Item -Path "notebooks/run_*.py", "notebooks/*.log", "notebooks/*.txt" -ErrorAction SilentlyContinue
```

## 📋 Resultado Esperado
1. Notebook actualizado en `notebooks/03_eda.ipynb`.
2. Visualizaciones generadas en la carpeta de experimentos.
3. Reporte de trazabilidad generado en `experiments/phase_03_eda/artifacts/phase_03_eda.json`.
