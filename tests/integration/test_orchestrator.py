import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from main import main

def test_orchestrator_phase_routing(integration_config, mock_supabase_responses):
    """
    Escenario 4: Sincronización del Orchestrator.
    Verifica que al pasar --phase discovery el orquestador ejecuta esa fase
    y genera los artefactos esperados en las rutas temporales.
    """
    
    # Preparamos los argumentos de CLI
    test_args = ["--phase", "discovery"]
    
    # Patch load_config globally in relevant modules
    with patch("src.utils.config_loader.load_config", return_value=integration_config), \
         patch("src.loader.load_config", return_value=integration_config), \
         patch("src.trainer.load_config", return_value=integration_config), \
         patch("main.load_config", return_value=integration_config):
        
        with patch("src.loader.DataLoader._fetch_all_with_pagination") as mock_fetch:
            # El mock devuelve el dataframe correspondiente según los tests
            def side_effect(*args, **kwargs):
                # args puede contener (self, supabase, table_name) o (supabase, table_name)
                # según cómo pytest/unittest maneje el parcheo del método de clase.
                # En ambos casos, el nombre de la tabla suele estar al final de los posicionales requeridos.
                table_name = args[2] if len(args) > 2 else args[1]
                return mock_supabase_responses[table_name]
            mock_fetch.side_effect = side_effect
            
            # Ejecutamos main pasando los argumentos directamente
            main(test_args)
            
    # Verificamos que se generó el reporte de discovery
    report_dir = os.path.join(integration_config["general"]["paths"]["reports"], "phase_01_discovery")
    assert os.path.exists(os.path.join(report_dir, "phase_01_discovery_latest.json"))

def test_orchestrator_phase_eda(integration_config, mock_supabase_responses):
    """
    Verifica que el orquestador ejecute la fase EDA correctamente.
    Requiere que master_cleansed.parquet exista.
    """
    # 1. Preparar datos (Simular que ya pasó el preprocesamiento)
    import pandas as pd
    from src.preprocessor import Preprocessor
    preprocessor = Preprocessor(integration_config)
    
    # Necesitamos datos raw para el preprocessor
    # (En una prueba real de orquesta, esto vendría de discovery, pero aquí lo simulamos)
    master_df = preprocessor.process(mock_supabase_responses) 
    
    # Guardar en disco para el orquestador
    cleansed_dir = integration_config["general"]["paths"]["cleansed"]
    os.makedirs(cleansed_dir, exist_ok=True)
    master_df.to_parquet(os.path.join(cleansed_dir, "master_cleansed.parquet"))
    
    # 2. Ejecutar EDA vía Orchestrator
    test_args = ["--phase", "eda"]
    with patch("src.utils.config_loader.load_config", return_value=integration_config), \
         patch("main.load_config", return_value=integration_config):
        
        # Necesitamos que matplotlib no abra ventanas
        import matplotlib
        matplotlib.use('Agg')
        
        main(test_args)
        
    # 3. Verificar resultados
    report_dir = os.path.join(integration_config["general"]["paths"]["reports"], "phase_03_eda")
    assert os.path.exists(os.path.join(report_dir, "phase_03_eda_latest.json"))
    
    # Verificar que se generó al menos una figura
    figures_dir = os.path.join(integration_config["general"]["paths"]["figures"], "phase_03_eda")
    assert os.path.exists(figures_dir)
    assert len(os.listdir(figures_dir)) > 0

def test_orchestrator_phase_features(integration_config, mock_supabase_responses):
    """
    Verifica que el orquestador ejecute la fase FEATURES correctamente.
    """
    # 1. Preparar datos (Master Cleansed)
    import pandas as pd
    from src.preprocessor import Preprocessor
    preprocessor = Preprocessor(integration_config)
    df_cleansed = preprocessor.process(mock_supabase_responses) 
    
    # Guardar en disco para el orquestador
    cleansed_dir = integration_config["general"]["paths"]["cleansed"]
    os.makedirs(cleansed_dir, exist_ok=True)
    df_cleansed.to_parquet(os.path.join(cleansed_dir, "master_cleansed.parquet"))
    
    # 2. Ejecutar Features vía Orchestrator
    test_args = ["--phase", "features"]
    with patch("src.utils.config_loader.load_config", return_value=integration_config), \
         patch("main.load_config", return_value=integration_config):
        main(test_args)
        
    # 3. Verificar resultados
    report_dir = os.path.join(integration_config["general"]["paths"]["reports"], "phase_04_feature_engineering")
    assert os.path.exists(os.path.join(report_dir, "phase_04_feature_engineering_latest.json"))
    
    # Verificar existencia del dataset generado
    features_path = os.path.join(integration_config["general"]["paths"]["features"], "master_features.parquet")
    assert os.path.exists(features_path)

def test_orchestrator_phase_modeling(integration_config, mock_supabase_responses):
    """
    Verifica que el orquestador ejecute la fase MODELING correctamente.
    """
    # 1. Preparar datos (Master Features)
    from src.preprocessor import Preprocessor
    from src.features import FeatureEngineer
    preprocessor = Preprocessor(integration_config)
    df_cleansed = preprocessor.process(mock_supabase_responses) 
    engineer = FeatureEngineer(integration_config)
    df_features = engineer.engineer(df_cleansed)
    
    # IMPORTANTE: Guardar en disco para que el Trainer lo encuentre
    processed_dir = integration_config["general"]["paths"]["processed"]
    os.makedirs(processed_dir, exist_ok=True)
    target_path = os.path.join(processed_dir, "master_features.parquet")
    df_features.to_parquet(target_path)
    
    # 2. Ejecutar Modeling vía Orchestrator
    test_args = ["--phase", "modeling"]
    with patch("src.utils.config_loader.load_config", return_value=integration_config), \
         patch("src.trainer.load_config", return_value=integration_config), \
         patch("main.load_config", return_value=integration_config):
        
        # Backend Agg ya está configurado en los módulos, pero lo aseguramos
        import matplotlib
        matplotlib.use('Agg')
        
        main(test_args)
        
    # 3. Verificar resultados
    report_dir = os.path.join(integration_config["general"]["paths"]["reports"], "phase_05_modeling")
    assert os.path.exists(os.path.join(report_dir, "phase_05_modeling_latest.json"))
    
    # 3. El modelo champion debe existir
    models_dir = integration_config["general"]["paths"]["models"]
    assert os.path.exists(os.path.join(models_dir, "champion_forecaster_latest.pkl"))
    
    # Verificar figuras de diagnóstico
    figures_dir = os.path.join(integration_config["general"]["paths"]["figures"], "phase_05_modeling")
    assert os.path.exists(figures_dir)
    assert len(os.listdir(figures_dir)) > 0

def test_orchestrator_invalid_phase():
    """
    Verifica que el orquestador falla ante una fase inexistente (validación de argparse).
    """
    test_args = ["--phase", "inexistente"]
    with pytest.raises(SystemExit): 
        main(test_args)

def test_orchestrator_modes(integration_config):
    """
    Verifica que el orquestador acepta los modos train y forecast y enruta correctamente.
    """
    # Probar modo train (explícito)
    test_args = ["--mode", "train", "--phase", "features"]
    
    # Mockeamos _run_features para que no intente leer el disco
    with patch("main.load_config", return_value=integration_config), \
         patch("main.setup_logging"), \
         patch("main._run_features") as mock_run:
        main(test_args)
        assert mock_run.called

    # Probar modo forecast
    test_args = ["--mode", "forecast", "--phase", "inference"]
    # Mockeamos print o logger para verificar que llega al warning de inferencia
    with patch("main.load_config", return_value=integration_config), \
         patch("main.setup_logging"):
        # Inference aún no está implementado y main imprime un warning
        main(test_args) 

    # Probar modo inválido
    test_args = ["--mode", "invalid"]
    with pytest.raises(SystemExit):
        main(test_args)
