import matplotlib
matplotlib.use('Agg')
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
from src.explorer import DataExplorer
from src.features import FeatureEngineer
from src.trainer import ForecasterTrainer

def main(args_list=None):
    parser = argparse.ArgumentParser(description="Mi Buñuelito Forecasting Orchestrator")
    parser.add_argument("--phase", type=str, required=False, default=None,
                        choices=["discovery", "financial_audit", "preprocessing", "eda", "features", "modeling", "inference"],
                        help="Execution phase to run (if omitted, runs all phases for the selected mode)")
    parser.add_argument("--mode", "-m", type=str, required=False, default="train",
                        choices=["train", "forecast"],
                        help="Execution mode: 'train' (full development) or 'forecast' (production inference). Default: 'train'")
    
    args = parser.parse_args(args_list)
    config = load_config()
    setup_logging()
    
    logger = logging.getLogger("Orchestrator")
    logger.info(f"🛠️ Modo de ejecuci\u00f3n: {args.mode.upper()}")
    
    # Define phases to run
    if args.phase:
        phases_to_run = [args.phase]
        logger.info(f"🚀 Iniciando fase individual: {args.phase}")
    else:
        if args.mode == "train":
            phases_to_run = ["discovery", "financial_audit", "preprocessing", "eda", "features", "modeling"]
        else: # forecast mode
            phases_to_run = ["discovery", "preprocessing", "features", "inference"]
            
        logger.info(f"🚀 Iniciando Pipeline ({args.mode}): {', '.join(phases_to_run)}")

    base_reports_path = config["general"]["paths"]["reports"]

    try:
        for phase in phases_to_run:
            if phase == "discovery":
                _run_discovery(config, base_reports_path, logger)
            
            elif phase == "financial_audit":
                _run_financial_audit(config, base_reports_path, logger)
            
            elif phase == "preprocessing":
                _run_preprocessing(config, base_reports_path, logger)
            
            elif phase == "eda":
                _run_eda(config, base_reports_path, logger)
            
            elif phase == "features":
                _run_features(config, base_reports_path, logger)

            elif phase == "modeling":
                _run_modeling(config, base_reports_path, logger)

            elif phase == "inference":
                _run_inference(config, base_reports_path, logger)

            else:
                logger.warning(f"⚠️ La fase '{phase}' no tiene un ejecutor asignado en el orquestador.")

    except Exception as e:
        logger.error(f"❌ Error crítico en la ejecución: {str(e)}")
        sys.exit(1)

def _run_discovery(config, base_reports_path, logger):
    logger.info("--- Ejecutando Fase: DISCOVERY ---")
    loader = DataLoader(config=config)
    report = loader.load_and_audit()
    
    output_dir = os.path.join(base_reports_path, "phase_01_discovery")
    save_report(report, output_dir, "phase_01_discovery")
    logger.info(f"✅ Discovery completado. Reportes guardados en: {output_dir}")

def _run_financial_audit(config, base_reports_path, logger):
    logger.info("--- Ejecutando Fase: FINANCIAL_AUDIT ---")
    loader = DataLoader(config=config) 
    data = {}
    tables = ["ventas_diarias", "redes_sociales"]
    for table in tables:
        file_path = os.path.join(config["general"]["paths"]["raw"], f"{table}.parquet")
        if os.path.exists(file_path):
            data[table] = pd.read_parquet(file_path)
    
    validator = BusinessValidator(config=config)
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

def _run_eda(config, base_reports_path, logger):
    logger.info("--- Ejecutando Fase: EDA ---")
    
    # 1. Cargar Master Cleansed
    cleansed_path = os.path.join(config["general"]["paths"]["cleansed"], "master_cleansed.parquet")
    if not os.path.exists(cleansed_path):
        raise FileNotFoundError(f"No se encontró el archivo maestro en {cleansed_path}. Ejecuta la fase de preprocesamiento primero.")
    
    df = pd.read_parquet(cleansed_path)
    
    # 2. Ejecutar Explorer
    explorer = DataExplorer(config)
    report = explorer.run_eda(df)
    
    logger.info(f"✅ EDA completado. Resultados en el reporte y visualizaciones generadas.")

def _run_features(config, base_reports_path, logger):
    logger.info("--- Ejecutando Fase: FEATURES ---")
    
    # 1. Cargar Master Cleansed
    cleansed_path = os.path.join(config["general"]["paths"]["cleansed"], "master_cleansed.parquet")
    if not os.path.exists(cleansed_path):
        raise FileNotFoundError(f"No se encontró el archivo maestro en {cleansed_path}. Ejecuta la fase de preprocesamiento primero.")
    
    df = pd.read_parquet(cleansed_path)
    
    # 2. Ejecutar FeatureEngineer
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer(df)
    
    logger.info(f"✅ Feature Engineering completado. Dataset enriquecido con {df_features.shape[1]} variables y {len(df_features)} registros.")

def _run_modeling(config, base_reports_path, logger):
    logger.info("--- Ejecutando Fase: MODELING ---")
    
    # 1. Instanciar Trainer
    trainer = ForecasterTrainer(config=config)
    
    # 2. Pipeline de Modelado
    trainer.load_and_split_data()
    trainer.run_baselines()
    trainer.run_all_experiments()
    
    # 3. Finalizar y Reportar
    trainer.save_final_report()
    
    # 4. Champion y Diagnósticos
    trainer.retrain_and_save_champion()
    trainer.generate_champion_diagnostics()
    
    logger.info("✅ Modeling completado. Champion guardado y diagnósticos generados.")

def _run_inference(config, base_reports_path, logger):
    logger.info("--- Ejecutando Fase: INFERENCE (FORECAST MODE) ---")
    logger.warning("🚧 Fase INFERENCE en desarrollo. Implementaci\u00f3n pendiente.")

if __name__ == "__main__":
    main()
