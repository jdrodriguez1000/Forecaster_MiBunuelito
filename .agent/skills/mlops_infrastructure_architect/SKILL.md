---
name: mlops_infrastructure_architect
description: Define los estándares de ingeniería, jerarquía de almacenamiento y protocolos de calidad para asegurar que los proyectos de forecasting sean reproducibles, modulares y auditables bajo la metodología Production-First.
---

# Skill: Arquitecto de Infraestructura MLOps (Forecasting Genérico)

Esta habilidad define el ecosistema técnico y de almacenamiento para cualquier proyecto de pronóstico de series de tiempo. Su objetivo es garantizar que la transición del experimento a la producción sea fluida y libre de errores de refactorización.

## 📂 1. Estándar de Almacenamiento (Data Layers)
Garantiza la inmutabilidad y el orden del flujo de datos:

*   **`data/01_raw/`**: Datos crudos obtenidos directamente de la fuente original (API, DB, CSV). Inmutables.
*   **`data/02_cleansed/`**: Datos tras limpieza inicial, estandarización de columnas y manejo de nulos.
*   **`data/03_features/`**: Datasets intermedios enriquecidos con ingeniería de variables (lags, estacionalidades, exógenas).
*   **`data/04_processed/`**: Dataset final listo para el entrenamiento del modelo (frecuencia agregada y alineada).

## 🏗️ 2. Metodología de Trabajo Industrializada (Production-First)
Este es el pilar del desarrollo. No se experimenta en notebooks para luego refactorizar; la lógica de producción es la base y los notebooks son una extensión automatizada para la validación visual.

1.  **Configuración y Parametrización ([CONFIG]):** Todo cambio nace en `config.yaml`. Se definen rutas, reglas de negocio e hiperparámetros.
2.  **Desarrollo del Core Técnico ([CORE]):** La lógica de procesamiento, modelos y utilidades se escribe directamente en módulos profesionales dentro de `src/`.
3.  **Pruebas Unitarias ([UNIT-TEST]):** El desarrollo de componentes atómicos debe validarse en `tests/unit/` antes de su integración.
4.  **Orquestación de Producción ([ORCHESTRATE]):** Se integra la lógica en el orquestador principal (`main.py`) para asegurar una ejecución determinística.
5.  **Generación de Salidas Oficiales ([PROD-OUT]):** La ejecución en producción genera reportes JSON y artefactos oficiales en la carpeta `outputs/`.
6.  **Pruebas de Integración ([INTEGRATION-TEST]):** Validación del flujo completo y contratos E2E en `tests/integration/`.
7.  **Automatización de Laboratorio ([GEN-SCRIPT]):** Creación de scripts generadores que construyen notebooks inyectando la lógica de `src/`.
8.  **Despliegue de Workflow Automático ([LAB-WORKFLOW]):** Creación de workflows `.md` para permitir la regeneración automatizada de notebooks.
9.  **Cierre y Sincronización ([CLOSE]):** Documentación, auditoría y commit final.

## 💻 3. Arquitectura de Código (`src/`)
Los módulos deben ser genéricos y orientados a objetos:

1.  **`src/connectors/`**: Clientes de base de datos o APIs (ej. `db_connector.py`).
2.  **`src/loader.py`**: Clase para la extracción de datos y validación de contratos iniciales.
3.  **`src/preprocessor.py`**: Lógica de limpieza, tratamiento de valores atípicos y agregaciones temporales.
4.  **`src/features.py`**: Generación de variables deterministas (calendario, festivos) y dinámicas (Moving Averages, Lags).
5.  **`src/models/`**: Definición de clases para entrenamiento, búsqueda de hiperparámetros y lógica de pronóstico (ej. `ForecasterDirect`).
6.  **`src/utils/`**: Helpers compartidos para logging, exportación a JSON y carga de archivos de configuración.

## ✅ 4. Capa de Validación y QA (`tests/`)
Cada fase técnica debe cerrar con pruebas que garanticen la integridad del pipeline:
*   **Tests Unitarios**: En `tests/unit/` para lógica atómica y contratos de entrada/salida de módulos individuales.
*   **Tests de Integración**: En `tests/integration/` para flujos E2E, persistencia de datos y consistencia entre fases.
*   **Herramienta**: Ejecución obligatoria vía `pytest`.

## ⚙️ 5. Protocolo de Configuración
*   **Zero Hardcoding**: Absolutamente todos los parámetros (rutas, horizonte de predicción, nombres de columnas, semillas de azar, hiperparámetros) deben residir en `config.yaml`.
*   **Estructura del Config**: El archivo debe estar organizado por bloques lógicos para facilitar su mantenimiento:
    1.  `general`: Configuración global y rutas.
    2.  `extractions`: Parámetros de conexión y carga.
    3.  `preprocessing`: Reglas de limpieza y agregación temporal.
    4.  `eda`: Parámetros de gráficos y análisis.
    5.  `features`: Configuración de ingeniería de variables.
    6.  `modeling`: Hiperparámetros, modelos y configuración de backtesting.
*   **Entorno**: Uso obligatorio de `.venv` y archivo `requirements.txt` actualizado.
*   **Seguridad**: Credenciales y tokens en archivo `.env`, excluido del control de versiones.

## 📊 6. Segregación de Salidas (Lab vs. Prod)

### 🔬 Laboratorio (`experiments/`)
*   Resultados de ejecución de **Notebooks / Phase_XX**.
*   `experiments/phase_0X_name/artifacts/`: Reportes JSON de experimentación.
*   `experiments/phase_0X_name/figures/`: Gráficos exploratorios y de diagnóstico.

### 🏭 Producción (`outputs/`)
*   Resultados de ejecución de **`main.py`** o triggers automáticos.
*   `outputs/reports/`: Reportes JSON finales y oficiales. Siguen el **Patrón de Persistencia Dual** para permitir que agentes de IA analicen el histórico (dentro de subcarpeta `history/` con formato `_YYYYMMDD_HHMMSS.json`) y el estado actual (en raíz como `_latest.json`).
*   `outputs/models/`: Binarios de los modelos campeones (`.joblib`, `.pkl`).
*   `outputs/forecasts/`: Resultados finales del pronóstico aplicados a datos reales.
*   `outputs/metrics/`: Resúmenes de desempeño (MAPE, RMSE, etc.) del set de test/evaluación.
