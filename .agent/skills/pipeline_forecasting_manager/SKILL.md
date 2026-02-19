---
name: pipeline_forecasting_manager
description: Gestiona la ejecución secuencial del pipeline de forecasting, asegurando la adherencia a la Metodología Production-First y los estándares de ciencia de datos.
---

# Skill: Gestor del Pipeline de Forecasting (Pipeline Manager)

Esta habilidad dirige el ciclo de vida de un proyecto de forecasting, desde la extracción de datos hasta la generación del pronóstico de negocio, garantizando que el código sea productivo desde su concepción.

## 🔄 Metodología de Ejecución (Production-First)
En cada fase técnica, el agente debe seguir obligatoriamente este flujo secuencial:

1.  **[CONFIG]**: Parametrización en `config.yaml`. Definición de rutas y reglas.
2.  **[CORE]**: Desarrollo de la lógica en archivos `.py` dentro de `src/`.
3.  **[ORCHESTRATE]**: Integración y prueba de la lógica en `main.py`.
4.  **[PROD-OUT]**: Ejecución en terminal para generar reportes oficiales en `outputs/`.
5.  **[TEST]**: Implementación y aprobación de pruebas unitarias en `tests/`.
6.  **[GEN-SCRIPT]**: Creación del script generador de notebooks en `scripts/` (ej: `gen_phase.py`).
7.  **[LAB-WORKFLOW]**: Creación del workflow `.agent/workflows/` para generar el notebook de experimentación.
8.  **[CLOSE]**: Commit a GitHub y aprobación formal del usuario.

## 🔬 Fases del Pipeline de Forecasting

### Fase 01: Data Discovery & Audit (Salud de Datos)
*   **Acción**: Conexión a la fuente de datos, carga inicial (o incremental) y auditoría de integridad.
*   **Controles Críticos**:
    *   **Data Contract**: Validar que las columnas y tipos de datos coincidan con lo definido en `config.yaml`.
    *   **Mínimo Histórico**: Verificar que existan suficientes datos para capturar estacionalidad (ej. 36 meses).
    *   **Salud Estadística**: Identificar nulos, valores centinela, duplicados y huecos temporales.
    *   **Integridad de Negocio**: Verificar consistencia interna de los datos (ej. sumatorias financieras, coherencia entre unidades).
*   **Resultados**: Reporte de salud de datos y almacenamiento en `data/01_raw/`.

### Fase 02: Preprocesamiento Robusto (Limpieza y Alineación)
*   **Acción**: Transformación de datos crudos en un dataset limpio y alineado temporalmente.
*   **Controles Críticos**:
    *   **Estandarización**: Formateo de nombres (snake_case) y tipos de datos.
    *   **Reindexación Temporal**: Asegurar una frecuencia continua (Diaria/Mensual) sin saltos en el tiempo.
    *   **Imputación Lógica**: Aplicar reglas de negocio para llenar huecos (Interpolación, Rolling Mean, etc.).
    *   **Anti-Data Leakage**: Eliminar periodos incompletos (como el mes en curso) para evitar sesgos en el entrenamiento.
    *   **Agregación**: Resample del dataset a la frecuencia del pronóstico final (ej. diario a mensual).
*   **Resultados**: Dataset maestro en `data/02_cleansed/`.

### Fase 03: EDA (Análisis Exploratorio de Datos)
*   **Acción**: Análisis profundo orientado al modelado bajo el principio **"Ojos solo en el Pasado"**.
*   **Controles Críticos**:
    *   **Segmentación**: Análisis exclusivo sobre el set de entrenamiento (Train) para evitar fuga de información.
    *   **Estacionariedad**: Ejecución de pruebas estadísticas (ej. ADF - Dickey-Fuller).
    *   **Patrones**: Descomposición estacional (Tendencia, Estacionalidad, Residuo) y análisis de autocorrelación (ACF/PACF).
    *   **Atípicos**: Identificación de shocks externos (eventos especiales, pandemias) y tratamiento de outliers.
*   **Resultados**: Insights de modelado y figuras en `experiments/phase_03_eda/`.

### Fase 04: Feature Engineering (Variables Exógenas)
*   **Acción**: Enriquecimiento del dataset con variables externas y proyecciones del horizonte futuro que expliquen la varianza de la demanda.
*   **Controles Críticos**:
    *   **Variables Deterministas**: Creación de indicadores basados en el calendario, eventos cíclicos, hitos históricos y dinámicas de mercado locales.
    *   **Exógenas Futuras**: Implementación obligatoria de lógica de proyección para todas las variables externas durante el horizonte de predicción (ej. escenarios o interpolaciones) para alimentar el modelo en los pasos futuros.
    *   **Nota Técnica**: La creación de *Lags* y *Window Features* no se realiza en esta fase, ya que se delega a la configuración paramétrica de `skforecast` en la fase de modelado.
*   **Resultados**: Dataset enriquecido en `data/03_features/` y `data/04_processed/`.

### Fase 05: Modelado y Pronóstico (Backtesting y Producción)
*   **Acción**: Entrenamiento competitivo y generación del forecast final.
*   **Controles Críticos**:
    *   **Tournament**: Competencia entre modelos (Ridge, RF, Boosting) contra un **Seasonal Naive Baseline**.
    *   **Backtesting**: Evaluación mediante validación cruzada temporal (Rolling Window).
    *   **Diagnóstico Residencial**: Análisis de errores (MAE, MAPE, RMSE) y búsqueda de sesgos en los residuos.
    *   **Incertidumbre**: Generación de intervalos de confianza (ej. Bootstrapping).
    *   **Champion Model**: Exportación del mejor modelo y reporte ejecutivo de proyecciones.
*   **Resultados**: Modelo en `outputs/models/` y pronósticos en `outputs/forecasts/`.

## 📊 Protocolo de Trazabilidad
Cada fase debe generar un artefacto JSON con:
*   `phase`: Nombre de la fase.
*   `timestamp`: Fecha y hora de ejecución.
*   `metrics`: Resultados clave de la fase (ej. % nulos, error del modelo).
*   `status`: Resultado de las pruebas unitarias relacionadas.
