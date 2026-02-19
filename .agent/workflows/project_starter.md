---
description: Inicializa la infraestructura física del proyecto (directorios, archivos base y entorno virtual) siguiendo los estándares de MLOps y arquitectura.
---

// turbo-all
---
description: Inicializa la infraestructura física del proyecto (directorios, archivos base y entorno virtual) siguiendo los estándares de la habilidad mlops_infrastructure_architect.
---

# Workflow: Inicialización del Proyecto de Forecasting (Bootstrap Wizard)

Este flujo de trabajo es responsable de la creación física de la infraestructura del proyecto. Su objetivo es asegurar que la jerarquía de directorios y los archivos base cumplan estrictamente con los estándares definidos en la habilidad de Arquitectura MLOps.

## 🛠️ Pasos de Ejecución (Setup Inicial)

### Paso 1: Creación de la Estructura de Directorios
// turbo
1. Generar la jerarquía de carpetas definida en el Skill `mlops_infrastructure_architect`:
    * `data/01_raw`, `data/02_cleansed`, `data/03_features`, `data/04_processed`
    * `notebooks/`, `scripts/`, `src/connectors/`, `src/models/`, `src/utils/`, `tests/`
    * `experiments/phase_01_discovery/artifacts`, `experiments/phase_01_discovery/figures`
    * `experiments/phase_01A_financial_audit/artifacts`
    * `experiments/phase_02_preprocessing/artifacts`, `experiments/phase_02_preprocessing/figures`
    * `experiments/phase_03_eda/figures`
    * `experiments/phase_04_features/artifacts`, `experiments/phase_04_features/figures`
    * `experiments/phase_05_modeling/artifacts`, `experiments/phase_05_modeling/figures`
    * `outputs/models`, `outputs/metrics`, `outputs/figures`, `outputs/forecasts`, `outputs/reports`

### Paso 2: Despliegue de Archivos Base (Scaffolding)
// turbo
1. Crear los archivos base en `src/`, `scripts/` y raíz:
    * `src/connectors/db_connector.py` (Conexión genérica).
    * `src/loader.py` (Lógica de extracción).
    * `src/preprocessor.py` (Limpieza y agregación).
    * `src/features.py` (Ingeniería de variables).
    * `src/models/forecaster.py` (Lógica de skforecast).
    * `src/utils/helpers.py` (Manejo de JSON/Logging).
    * `src/utils/config_loader.py` (Carga de YAML).
    * `scripts/gen_discovery.py` (Generador de notebook fase 01).
    * `main.py` (Orquestador central).
    * `.env.example` y `.env` (Variables de entorno).
    * `notebooks/00_workbench.ipynb` (Scratchpad inicial).

### Paso 3: Configuración y Control
// turbo
1. Crear un `config.yaml` inicial con la estructura jerárquica obligatoria (general, extractions, preprocessing, eda, features, modeling).
2. Generar un `requirements.txt` con las librerías base: `skforecast`, `pandas`, `numpy`, `python-dotenv`, `pyyaml`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `papermill`, `pytest`.
3. Crear un `.gitignore` estándar para Python incluyendo `.venv`, `.env`, `data/`, y archivos temporales.

### Paso 4: Configuración del Entorno Python
// turbo
1. Validar la versión de Python (Recomendada: **3.12.10**).
2. Crear entorno virtual: `py -3.12 -m venv .venv`.
3. Activar entorno virtual.
4. Ejecutar instalación: `pip install -r requirements.txt`.

### Paso 5: Validación Final
1. Verificar que toda la nomenclatura (carpetas, archivos, variables) esté en **Inglés**.
2. Confirmar que el proyecto está listo para iniciar la **Fase 1: Data Discovery**.

---

## 🚦 Salida Esperada
Un árbol de directorios confirmado, entorno `.venv` configurado y archivo `config.yaml` listo para ser personalizado según el proyecto específico.
