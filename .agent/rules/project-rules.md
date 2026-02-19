---
trigger: always_on
description: Restricciones de dominio, arquitectura MLOps y metodología de trabajo para el proyecto Mi Buñuelito.
---

# Project Rules: Mi Buñuelito Forecasting

Este archivo constituye la autoridad máxima de restricciones cognitivas y técnicas para el proyecto. Todo agente o colaborador debe asegurar el cumplimiento estricto de estas reglas antes de ejecutar cualquier Skill o Workflow.

---

## 1. 🎯 Restricciones de Dominio y Negocio
*   **Consultora:** Sabbia Solutions & Services (Triple S).
*   **Cliente:** Corporación Comercial de Alimentos SAS.
*   **Marca/Producto:** Mi Buñuelito / Buñuelo (Producto Estrella).
*   **Variable Objetivo:** `total_unidades_entregadas` (Forecasting de demanda mensual).
*   **Regla de Oro (Anti-Data Leakage):** 
    *   **Temporalidad**: El entrenamiento para el mes $X$ debe detenerse estrictamente en el cierre del mes $X-1$. Queda prohibido el uso de información parcial o total del mes en curso para predecir el futuro.
    *   **Atomicidad**: Prohibido el uso de variables exógenas que sean resultado de operaciones matemáticas con la variable objetivo del mismo periodo (ej. Ingresos Totales, Costos Totales). Solo se permiten variables "atómicas" o indicadores independientes.
*   **Horizonte de Predicción:** El sistema debe generar siempre un pronóstico de 6 meses (mes actual $X$ hasta $X+5$).
*   **Métricas de Éxito:** El modelo final es válido solo si supera al baseline *Seasonal Naive* y mantiene un **MAPE < 30%**.

## 2. 🏗️ Arquitectura de Software y Estándares
*   **Estrategia de Modelado:** Uso obligatorio de la librería `skforecast` mediante la estrategia `ForecasterDirect`.
*   **Batería de Modelos Autorizados:** Solo se permite la experimentación y competencia entre:
    *   `Ridge`, `RandomForestRegressor`, `LGBMRegressor`, `XGBRegressor`, `GradientBoostingRegressor` y `HistGradientBoostingRegressor`.
*   **Configuración:** Prohibido el uso de valores "hardcoded". Rutas, hiperparámetros, fechas de corte y nombres de variables deben residir en `config.yaml`. Este archivo debe seguir una estructura jerárquica estricta por fases:
    1.  `general`: Parámetros globales (semillas, rutas base).
    2.  `extractions`: Carga y validación inicial de datos.
    3.  `preprocessing`: Limpieza, agregación y nulos.
    4.  `eda`: Visualizaciones y análisis estadístico.
    5.  `features`: Ingeniería de variables y proyecciones.
    6.  `modeling`: Entrenamiento, modelos y backtesting.
*   **Idioma:** Código y estructura de archivos en **Inglés**; contexto y reglas de negocio en **Español**.
*   **Persistencia:** La fuente de verdad histórica es **Supabase (PostgreSQL)**. Tablas: `ventas_diarias`, `redes_sociales`, `promocion_dia`, `macro_economia`.
*   **Carga de Datos:** La descarga de información debe ser estrictamente **incremental** (descargando solo la diferencia faltante), salvo en la carga inicial o cuando se fuerce una actualización completa.

## 3. 🔬 Rigor en Ciencia de Datos y Validación
*   **Estrategia de Partición (Backtesting):** Se debe aplicar un esquema de validación cruzada temporal con lógica rodante (Rolling Window):
    *   **Test:** Últimos 12 meses del dataset.
    *   **Validación:** 12 meses inmediatamente anteriores al bloque de Test.
    *   **Entrenamiento:** Todo el histórico restante previo a Validación.
*   **Umbral de Datos Mínimos:** El pipeline debe validar la existencia de al menos **36 meses** de datos históricos antes de proceder con el modelado.
*   **Tratamiento de Exógenas Futuras:** Las variables macroeconómicas para el horizonte de 6 meses deben proyectarse mediante **Promedio Móvil Recursivo de 2 meses**.
*   **Lógica de Negocio (Features Obligatorias):**
    *   **Pandemia:** Flag para el periodo `2020-04-01` a `2021-05-31`.
    *   **Promociones (2x1):** Meses `Abr-May` y `Sep-Oct` (desde el año 2020).
    *   **Novenas Navideñas:** Incremento específico del `16 al 23 de diciembre`.
    *   **Festivos:** Deben ser tratados con el mismo peso/importancia que un **Sábado**.
    *   **Patrones de Pago:** Marcar Quincenas (15 y 30) y Primas (Junio y Diciembre).
*   **Reproducibilidad:** Se debe garantizar un comportamiento determinista utilizando la semilla global `random_state=42`.

## 4. 🛠️ Protocolo de Integridad y Verdad de Datos
Para garantizar la calidad del pipeline, se aplican las siguientes leyes de limpieza obligatorias:
*   **Fechas Duplicadas:** En caso de existir múltiples registros para una misma fecha con valores distintos, se debe conservar únicamente el **último registro** (considerado como la actualización más reciente).
*   **Filas Duplicadas:** Si una fila completa se encuentra duplicada, se debe conservar solo la **última instancia**.
*   **Continuidad Temporal (Reindexación):** El dataset debe ser cronológicamente completo. Si falta un registro para una fecha específica, este debe ser **creado con valores nulos** para asegurar la integridad de los lags y la frecuencia de la serie de tiempo.

## 5. ⚙️ Metodología de Trabajo Industrializada (Production-First)
Se adopta un enfoque lineal y riguroso para garantizar que la lógica de producción sea la base de toda experimentación:

1.  **Configuración y Parametrización:** Todo cambio nace en `config.yaml`. Se definen rutas, reglas de negocio e hiperparámetros. Prohibido el uso de valores "hardcoded".
2.  **Desarrollo del Core Técnico (`src/`):** La lógica de procesamiento, modelos y utilidades se escribe directamente en módulos profesionales dentro de `src/`.
3.  **Orquestación de Producción (`main.py`):** Se integra la lógica en el orquestador principal para asegurar una ejecución determinística desde la terminal.
4.  **Generación de Salidas Oficiales (`outputs/`):** La ejecución en producción genera reportes JSON y artefactos oficiales en la carpeta `outputs/`.
5.  **Validación Rigurosa (`tests/`):** Creación y ejecución de pruebas unitarias para garantizar que la lógica del Paso 2 cumpla con los contratos y reglas del negocio.
6.  **Automatización de Laboratorio (`scripts/`):** Creación del script generador (ej: `gen_phase.py`) que construye el notebook de la fase inyectando la lógica de `src/` y configurando el "Modo Laboratorio".
7.  **Despliegue de Workflow Automático (`.agent/workflows/`):** Creación del archivo de workflow que permite al agente o usuario regenerar el notebook de forma automatizada.
8.  **Cierre y Sincronización:** Documentación final, commit/push a GitHub y aprobación formal de la fase.

## 6. 📂 Segregación de Salidas (Ambientes Lab vs. Prod)
Queda estrictamente prohibido mezclar salidas de experimentación con las de producción:
*   **Entorno Lab (Notebooks):** Todas las salidas deben dirigirse a `experiments/phase_0X_name/`.
    *   Los reportes JSON de experimentación van en la subcarpeta `artifacts/` y su nombre inicia por `phase_0X_name.json`.
    *   Toda visualización va en la subcarpeta `figures/`.
*   **Entorno Prod (Módulos .py y main.py):** Todas las salidas oficiales deben dirigirse a `outputs/`.
    *   Los reportes JSON finales se guardan en `outputs/reports/` en subcarpetas por fase.
    *   Visualizaciones oficiales en `outputs/figures/`.
    *   Modelos (.pkl), pronósticos y métricas en sus respectivas carpetas raíz de `outputs/`.

## 7. 📤 Protocolo de Entregables y Trazabilidad
*   **Reportes de Fase (Trazabilidad):** Cada proceso debe generar obligatoriamente archivos `.json` siguiendo el **Patrón de Persistencia Dual**:
    *   **Versión Histórica:** `nombre_fase_YYYYMMDD_HHMMSS.json` (Inmutable).
    *   **Versión Puntero:** `nombre_fase_latest.json` (Sobrescrita en cada ejecución).
    *   **Contenido:** Debe incluir encabezado con `phase`, `timestamp` y `description`.
*   **Gestión de Entorno:** Ejecución obligatoria dentro de ambiente virtual `.venv` y mantenimiento riguroso de `requirements.txt`.
*   **Aprobación de Fase (Gatekeeper):** Queda estrictamente prohibido avanzar a una nueva fase del proyecto sin la **aprobación explícita y completa** del usuario sobre los entregables de la fase actual.