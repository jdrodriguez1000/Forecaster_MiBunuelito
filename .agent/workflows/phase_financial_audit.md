---
description: generate and maintain the financial and business logic audit notebook
---

# Workflow: Fase 2 - Auditoría de Lógica de Negocio (Financial Audit)

EXTREMELY IMPORTANT: DO NOT EDIT THE NOTEBOOK DIRECTLY. EDIT THE SCRIPT `scripts/gen_financial_audit.py` INSTEAD.
Este flujo de trabajo tiene como objetivo generar automáticamente el notebook `notebooks/02_financial_audit.ipynb` utilizando las reglas financieras definidas en `config.yaml` y el validador en `src/validator.py`.

## 🛠️ Pasos de Ejecución

### Paso 1: Generación del Notebook
Ejecuta el script generador que crea el notebook con las 8 reglas de validación financiera y de marketing.

// turbo
```powershell
python scripts/gen_financial_audit.py
```

### Paso 2: Validación y Ejecución Manual
* **Acción:** Abre y ejecuta manualmente el notebook `notebooks/02_financial_audit.ipynb`.
* **Objetivo:** Confirmar que los datos extraídos en la Fase 01 sean consistentes contablemente (Ventas vs Costos vs Utilidad).

### Paso 3: Limpieza de Archivos Temporales
Mantiene la carpeta de notebooks limpia de residuos de ejecución.

// turbo
```powershell
Remove-Item -Path "notebooks/run_*.py", "notebooks/*.log", "notebooks/*.txt" -ErrorAction SilentlyContinue
```

## 📋 Resultado Esperado
1. Notebook actualizado en `notebooks/02_financial_audit.ipynb`.
2. Reporte de auditoría financiera generado en `experiments/phase_02_financial_audit/artifacts/phase_02_financial_audit.json`.
