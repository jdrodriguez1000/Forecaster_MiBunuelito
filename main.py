import argparse
import sys
import logging
import os
import pandas as pd
from src.utils.config_loader import load_config
from src.utils.helpers import setup_logging, save_report
from src.loader import DataLoader
from src.validator import BusinessValidator
from src.preprocessor import Preprocessor

def main(args_list=None):
    parser = argparse.ArgumentParser(description="Mi Buñuelito Forecasting Orchestrator")
    parser.add_argument("--phase", type=str, required=False, default=None,
                        choices=["discovery", "financial_audit", "preprocessing", "eda", "features", "modeling"],
                        help="Execution phase to run (if omitted, runs all implemented phases)")
    
    args = parser.parse_args(args_list)
    config = load_config()
    setup_logging()
    
    logger = logging.getLogger("Orchestrator")
    
    # Define phases to run
    if args.phase:
        phases_to_run = [args.phase]
        logger.info(f"🚀 Iniciando fase individual: {args.phase}")
    else:
        phases_to_run = ["discovery", "financial_audit", "preprocessing"] # Fases implementadas
        logger.info(f"🚀 Iniciando Pipeline Completo (Fases: {', '.join(phases_to_run)})")

    base_reports_path = config["general"]["paths"]["reports"]

    try:
        for phase in phases_to_run:
            if phase == "discovery":
                _run_discovery(config, base_reports_path, logger)
            
            elif phase == "financial_audit":
                _run_financial_audit(config, base_reports_path, logger)
            
            elif phase == "preprocessing":
                _run_preprocessing(config, base_reports_path, logger)
            
            else:
                if args.phase: # Solo advertir si el usuario pidió explícitamente una fase no implementada
                    logger.warning(f"⚠️ La fase '{phase}' aún no está implementada en el orquestador.")

    except Exception as e:
        logger.error(f"❌ Error crítico en la ejecución: {str(e)}")
        sys.exit(1)

def _run_discovery(config, base_reports_path, logger):
    logger.info("--- Ejecutando Fase: DISCOVERY ---")
    loader = DataLoader()
    report = loader.load_and_audit()
    
    output_dir = os.path.join(base_reports_path, "phase_01_discovery")
    save_report(report, output_dir, "phase_01_discovery")
    logger.info(f"✅ Discovery completado. Reportes guardados en: {output_dir}")

def _run_financial_audit(config, base_reports_path, logger):
    logger.info("--- Ejecutando Fase: FINANCIAL_AUDIT ---")
    loader = DataLoader() 
    data = {}
    tables = ["ventas_diarias", "redes_sociales"]
    for table in tables:
        file_path = os.path.join(config["general"]["paths"]["raw"], f"{table}.parquet")
        if os.path.exists(file_path):
            data[table] = pd.read_parquet(file_path)
    
    validator = BusinessValidator()
    report = validator.validate_all(data)
    
    output_dir = os.path.join(base_reports_path, "phase_01A_financial_audit")
    save_report(report, output_dir, "phase_01A_financial_audit")
    logger.info(f"✅ Financial Audit completado. Reportes guardados en: {output_dir}")

def _run_preprocessing(config, base_reports_path, logger):
    logger.info("--- Ejecutando Fase: PREPROCESSING ---")
    
    # 1. Cargar datos raw (Parquet)
    data = {}
    raw_path = config["general"]["paths"]["raw"]
    tables = config["extractions"]["tables"].keys()
    
    for table in tables:
        file_path = os.path.join(raw_path, f"{table}.parquet")
        if os.path.exists(file_path):
            data[table] = pd.read_parquet(file_path)
            logger.info(f"Cargada tabla para preprocesar: {table}")
        else:
            logger.warning(f"⚠️ Tabla {table} no encontrada en {raw_path}")

    if not data:
        raise ValueError("No se encontraron datos para preprocesar.")

    # 2. Ejecutar Preprocessor
    preprocessor = Preprocessor(config)
    master_df = preprocessor.process(data)
    
    logger.info(f"✅ Preprocessing completado. Master dataset generado con {len(master_df)} registros.")

if __name__ == "__main__":
    main()
