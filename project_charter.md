# Project Charter: Pronóstico de Ventas - Mi Buñuelito

## 1. Información General
*   **Empresa Consultora:** Sabbia Solutions & Services (Triple S)
*   **Cliente:** Corporación Comercial de Alimentos SAS
*   **Línea de Negocio:** Mi Buñuelito
*   **Producto Estrella:** Buñuelo
*   **Fecha de Creación:** 18 de febrero de 2026

## 2. Definición del Problema
Actualmente, el pronóstico de ventas de unidades de buñuelos es realizado de forma empírica por un comité de expertos (gerentes comerciales, financieros y generales). Este proceso carece de una metodología científica, basándose solo en históricos simples y juicios subjetivos.

**Dolores detectados:**
*   **Inexactitud:** Desviaciones cercanas al **25%**.
*   **Quiebre de Stock:** Pérdida de ventas en meses de alta demanda por subestimación del pronóstico (temor al inventario).
*   **Sesgo Político:** El pronóstico es influenciado por metas gerenciales más que por datos.

## 3. Objetivos del Proyecto
*   **Objetivo Principal:** Desarrollar un modelo de Machine Learning capaz de pronosticar las **unidades totales entregadas** de buñuelos con un horizonte de **6 meses**.
*   **Frecuencia:** Pronóstico mensual realizado los primeros días de cada mes (Mes X).
*   **Regla de Oro:** Para predecir el Mes X hasta X+5, solo se puede utilizar información disponible hasta el Mes X-1.

## 4. Fuentes de Datos
Se integrarán cuatro fuentes principales de información, almacenadas en **Supabase**:
1.  **`ventas_diarias`:** Histórico desde 2018-01-01 (unidades full, promo, costos e ingresos).
2.  **`redes_sociales`:** Inversión diaria en Facebook e Instagram.
3.  **`promocion_dia`:** Indicador diario de promociones (tipo 2x1).
4.  **`macro_economia`:** Datos mensuales de Colombia (IPC, TRM, desempleo, costo insumos, confianza consumidor).

## 5. Contexto y Reglas de Negocio (Features)
El modelo deberá capturar las siguientes dinámicas:
*   **Estacionalidad:** Picos en diciembre (especialmente Novenas 16-23 dic), enero, junio y julio.
*   **Días de Venta:** Fines de semana (Sáb/Dom) son los días de mayor volumen.
*   **Festivos:** Deben tratarse como sábados en términos de volumen de ventas.
*   **Efecto Calendario:** Impacto positivo de quincenas y meses de pago de prima.
*   **Promociones:** Impacto de las campañas 2x1 en Abr/May y Sep/Oct (iniciadas en 2022).
*   **Pandemia:** Manejo del periodo anómalo entre abril 2020 y mayo 2021.

## 6. Criterios de Éxito
El proyecto se considerará exitoso si:
1.  El modelo seleccionado supera consistentemente al Baseline (Seasonal Naive) en métricas de error (RMSE/MAE/MAPE) en el set de prueba.
2.  La herramienta es adoptada por el comité de expertos como insumo principal para sus proyecciones.

## 7. Alcance Técnico y Modelado
*   **Librería Principal:** `skforecast`.
*   **Estrategia de Pronóstico:** `ForecasterAutoregDirect` (6 modelos independientes).
*   **Línea Base (Baseline):** Modelo ingenuo estacional (Seasonal Naive).
*   **Modelos de Machine Learning a Evaluar:**
    *   Ridge Regression
    *   Random Forest
    *   LightGBM (LGBM)
    *   XGBoost
    *   Gradient Boosting
    *   HistGradientBoosting

## 8. Entregables
*   Estructura de datos en Supabase.
*   Pipeline de procesamiento y limpieza.
*   Reporte de entrenamiento y selección del "Champion Model".
*   Sistema de pronóstico recurrente de 6 meses.
