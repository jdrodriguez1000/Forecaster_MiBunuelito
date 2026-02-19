// turbo-all
---
description: Inicializa y sincroniza el proyecto local con GitHub y configura el entorno de Supabase.
---

# Workflow: Deploy GitHub (Forecaster Mi Buñuelito)

Este workflow automatiza la creación del repositorio en GitHub, la gestión del control de versiones y la preparación del despliegue inicial.

## Pasos de Ejecución

1. **Inicializar Repositorio Local**
   - Comando: `git init -b main`

2. **Verificar Remoto Existente**
   - Comando: `git remote -v`
   - *Decisión*: Si ya existe `origin` apuntando a `Forecaster_MiBunuelito`, saltar al paso 7.

3. **Verificar Existencia en GitHub**
   - Herramienta: `mcp_remote-github_search_repositories`
   - Consulta: `user:@me Forecaster_MiBunuelito`
   - *Decisión*: Si existe, obtener la `clone_url`. Si no, continuar al paso 4.

4. **Crear Repositorio Remoto**
   - Herramienta: `mcp_remote-github_create_repository`
   - Argumentos:
     - `name`: "Forecaster_MiBunuelito"
     - `private`: true
     - `description`: "Proyecto de forecasting para predicción de ventas de Mi Buñuelito (Triple S / Corporación Comercial de Alimentos SAS)."

5. **Configurar Origen**
   - Comando: `git remote add origin <CLONE_URL>`

6. **Gestión de Archivos Críticos (.gitignore)**
   - Revisar estado: `git status`
   - *Seguridad*: Asegurar que `.env`, `__pycache__`, `.venv/` y `outputs/` estén ignorados.
   - Si un archivo sensible está trackeado: `git rm --cached <archivo>`

7. **Preparación de Archivos (Stage)**
   - Comando: `git add .`

8. **Commit Inicial**
   - Comando: `git commit -m "feat: Estructura inicial del proyecto y configuración base de MLOps"`

9. **Push a GitHub**
   - Comando: `git push -u origin main`

## Verificación de Supabase (Opcional)
- Una vez en GitHub, asegurar que las credenciales en `.env` coincidan con el proyecto de Supabase correspondiente para garantizar la conectividad del pipeline.
