---
name: forecasting_domain_expert
description: Encapsula el conocimiento experto sobre las dinámicas de venta, estacionalidad y proyecciones matemáticas específicas para el negocio de buñuelos de Mi Buñuelito.
---

# Skill: Experto en el Dominio de Pronóstico (Mi Buñuelito)

Esta habilidad dota al agente del conocimiento experto sobre el comportamiento del consumidor, ciclos comerciales y factores externos que afectan la demanda del producto estrella de **Mi Buñuelito**.

## 1. 🏢 Contexto Estratégico
*   **Producto Estrella:** Buñuelo.
*   **Variable Objetivo:** `total_unidades_entregadas` (Unidades físicas que salen de planta).
*   **Horizonte de Decisión:** 6 meses (Pronóstico de corto y mediano plazo).
*   **Regla de Tiempo Crucial:** Las decisiones se basan en información con cierre al mes $X-1$.

## 2. 🧠 Lógica de Proyección de Exógenas
Debido a que el modelo requiere conocer las variables externas para los 6 meses futuros (donde no hay datos reales), se debe aplicar una proyección determinista:
*   **Método:** Promedio Móvil Recursivo de 2 meses ($MA_2$).
*   **Variables:** `ipc_mensual`, `trm_promedio`, `tasa_desempleo`, `costo_insumos_index`, `confianza_consumidor`.
*   **Propósito:** Proporcionar una estimación estable que capture la inercia reciente de la economía colombiana.

## 3. 📅 Calendario de Negocio (Business Features)

### A. Estacionalidad Mensual (Picos de Demanda)
*   **Meses de Alta Venta:** Diciembre (Novenas y Navidad), Enero (Vacaciones), Junio y Julio (Temporada media y vacaciones escolares).
*   **Acción:** Creación de variables indicadoras para estos periodos específicos.

### B. Ciclos de Flujo de Caja (Patrones de Pago)
*   **Quincenas:** Aumento de consumo los días 15 y 30/31 de cada mes.
*   **Primas Legales:** Incrementos significativos en los meses de **Junio y Diciembre**.
*   **Días de la Semana:** El volumen se concentra en Sábados y Domingos.
*   **Festivos:** Homologación estadística. Un festivo tiene un comportamiento de ventas comparable al de un **Sábado**.

### C. Estrategia Promocional (Efecto 2x1)
*   **Temporadas:** Abril-Mayo y Septiembre-Octubre (Iniciadas en 2022).
*   **Mecánica:** "Compre uno, lleve otro gratis". 
*   **Impacto:** El volumen de `total_unidades_entregadas` se duplica potencialmente en estos periodos, aunque coexisten ventas a precio full.
*   **Acción:** Variable binaria de campaña para capturar el salto en volumen.

### D. Evento Crítico: Novenas Navideñas
*   **Ventana Temporal:** **16 al 23 de Diciembre**.
*   **Comportamiento:** Es el pico de demanda más agresivo del año debido a reuniones familiares y empresariales.
*   **Acción:** Flag específico para estos 8 días del año.

## 4. 📈 Tratamiento de Anomalías Históricas
*   **Pandemia (Outlier Estructural):** Periodo comprendido entre **Abril 2020 y Mayo 2021**.
*   **Acción:** Implementar una variable indicadora `is_pandemic` para que el modelo identifique que la caída extrema en ventas no es una tendencia natural, sino un shock externo.

## 5. 🧮 Configuración del Motor de Pronóstico
*   **Estrategia:** `ForecasterDirect` de `skforecast`.
*   **Modelado:** Uso de variables exógenas futuras proyectadas (Macro, Promos, Calendario) para cada uno de los 6 pasos del horizonte de predicción.

## 6. 🛠️ Protocolo de Imputación por Dominio
Para los valores nulos que persistan tras la limpieza inicial, se deben aplicar las siguientes reglas basadas en el conocimiento del negocio:

### A. Variables Macroeconómicas
*   **Regla:** Aplicar `Forward Fill` (propagar el último valor conocido) como política primaria para mantener la persistencia económica.
*   **Respaldo:** Aplicar `Back Fill` únicamente si el nulo se encuentra en el inicio de la serie histórica.

### B. Promociones (`es_promo`)
*   **Regla de Negocio:** La promoción se considera activa (`1`) basándose estrictamente en el calendario corporativo para fechas superiores o iguales al año 2022:
    *   **Ciclo Primavera:** Del 1 de Abril al 31 de Mayo.
    *   **Ciclo Otoño:** Del 1 de Septiembre al 31 de Octubre.
*   **Resto del Tiempo:** Cualquier nulo fuera de estos rangos, o cualquier fecha anterior al año 2022, debe imputarse obligatoriamente con `0`.

### C. Inversión en Redes Sociales (`redes_sociales`)
*   **Hito Pre-Estrategia (Hasta 17-Mar-2022)**: Cualquier nulo en esta ventana debe imputarse con `0` (campos numéricos) y `No ciclo` (campo ciclo).
*   **Periodos de Campaña (Refactorizados)**: En estas ventanas, los nulos numéricos usan `Forward Fill` y el campo ciclo se etiqueta según corresponda:
    *   **Ventana Abr-May** (15-Mar al 25-May): `Ciclo Abr-May`.
    *   **Ventana Sep-Oct** (15-Sep al 25-Oct): `Ciclo Sep-Oct`.
*   **Periodos Valle**: Cualquier otro caso de valores faltantes fuera de las condiciones anteriores se imputa con `0` y `No ciclo`.

### D. Ventas Diarias (`ventas_diarias`)
*   **Unidades Normales**: Imputar con `Forward Fill` (respaldo `Back Fill`) para mantener la continuidad del volumen base.
*   **Unidades Promocionales**:
    *   Si `es_promo == 0` $\rightarrow$ Imputar con `0`.
    *   Si `es_promo == 1` $\rightarrow$ Imputar con `Forward Fill` (respaldo `Back Fill`).
*   **Consistencia de Totales**: El campo `total_unidades_entregadas` debe ser recalculado como la suma de: `unidades_precio_normal` + `unidades_promo_pagadas` + `unidades_promo_bonificadas`.
*   **Precios y Costos**: En caso de nulos en `precio_unitario_full` o `costo_unitario`, imputar utilizando el valor representativo del mes correspondiente (Moda o promedio mensual).
*   **Campos Financieros (Auditoría)**: 
    *   `ingresos_totales` = `precio_unitario_full` $\times$ (`unidades_precio_normal` + `unidades_promo_pagadas`).
    *   `costo_total` = `costo_unitario` $\times$ `total_unidades_entregadas`.
    *   `utilidades` = `ingresos_totales` - `costo_total`.
    *   **Nota Técnica**: Aunque estos campos se calculan para la integridad del dataset procesado, la **Regla de Atomicidad** impide su uso como variables exógenas en el modelo final.
