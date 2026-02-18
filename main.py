import argparse
import sys
import logging
import os
import pandas as pd
from src.utils.config_loader import load_config
from src.utils.helpers import setup_logging, save_report
from src.loader import DataLoader
from src.validator import BusinessValidator

def main():
    parser = argparse.ArgumentParser(description="Mi Buñuelito Forecasting Orchestrator")
    parser.add_argument("--phase", type=str, required=False, default=None,
                        choices=["discovery", "financial_audit", "preprocessing", "eda", "features", "modeling"],
                        help="Execution phase to run (if omitted, runs all implemented phases)")
    
    args = parser.parse_args()
    config = load_config()
    setup_logging()
    
    logger = logging.getLogger("Orchestrator")
    
    # Define phases to run
    if args.phase:
        phases_to_run = [args.phase]
        logger.info(f"🚀 Iniciando fase individual: {args.phase}")
    else:
        phases_to_run = ["discovery", "financial_audit"] # Fases implementadas hasta ahora
        logger.info(f"🚀 Iniciando Pipeline Completo (Fases: {', '.join(phases_to_run)})")

    base_reports_path = config["general"]["paths"]["reports"]

    try:
        for phase in phases_to_run:
            if phase == "discovery":
                _run_discovery(config, base_reports_path, logger)
            
            elif phase == "financial_audit":
                _run_financial_audit(config, base_reports_path, logger)
            
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
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "phase_01_discovery.json")
    save_report(report, report_path)
    logger.info(f"✅ Discovery completado. Reporte: {report_path}")

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
    
    output_dir = os.path.join(base_reports_path, "phase_02_financial_audit")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "phase_02_financial_audit.json")
    save_report(report, report_path)
    logger.info(f"✅ Financial Audit completado. Reporte: {report_path}")

if __name__ == "__main__":
    main()
