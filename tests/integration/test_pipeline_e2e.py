import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.loader import DataLoader
from src.preprocessor import Preprocessor

def test_pipeline_full_integration(integration_config, mock_supabase_responses):
    """
    Escenario 1: Flujo End-to-End.
    Prueba que los datos viajan desde el Mock de DB hasta el Master Cleansed 
    pasando por todas las validaciones de contrato y reglas de negocio.
    """
    
    # --- PASO 1: MOCK DE CARGA (DISCOVERY) ---
    with patch("src.loader.load_config", return_value=integration_config):
        with patch("src.loader.DBConnector") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value.get_client.return_value = mock_client
            
            # Configuramos el mock para que devuelva los datos de conftest según la tabla pedida
            def mock_fetch(*args, **kwargs):
                print(f"DEBUG mock_fetch ARGS: {args}")
                # args[0]: DataLoader instance, args[1]: supabase, args[2]: table_name
                table_name = args[2] if len(args) > 2 else args[1] # fallback
                return mock_supabase_responses[table_name]
            
            with patch.object(DataLoader, "_fetch_all_with_pagination", side_effect=mock_fetch):
                loader = DataLoader()
                discovery_report = loader.load_and_audit()
                
                # Verificamos que se crearon los archivos RAW
                raw_path = integration_config["general"]["paths"]["raw"]
                for table in mock_supabase_responses.keys():
                    assert os.path.exists(os.path.join(raw_path, f"{table}.parquet"))
                
                assert "Phase 01: Data Discovery & Extraction" in discovery_report["phase"]
                
                # Manual save just like main.py does
                from src.utils.helpers import save_report
                reports_base = integration_config["general"]["paths"]["reports"]
                discovery_dir = os.path.join(reports_base, "phase_01_discovery")
                save_report(discovery_report, discovery_dir, "phase_01_discovery")

    # --- PASO 2: MOCK DE PREPROCESAMIENTO ---
    # Cargamos los datos raw que acabamos de crear para simular la Fase 2
    raw_data = {}
    for table in mock_supabase_responses.keys():
        path = os.path.join(raw_path, f"{table}.parquet")
        raw_data[table] = pd.read_parquet(path)
        
    # NO mockeamos save_report para verificar que se cree en la carpeta temporal
    preprocessor = Preprocessor(integration_config)
    master_df = preprocessor.process(raw_data)
        
    # --- PASO 3: MOCK DE EDA ---
    # El archivo master_cleansed.parquet ya fue creado en el paso anterior
    from src.explorer import DataExplorer
    
    # Necesitamos que matplotlib no abra ventanas
    import matplotlib
    matplotlib.use('Agg')
    
    explorer = DataExplorer(integration_config)
    eda_report = explorer.run_eda(master_df)
    
    assert eda_report["status"] == "success"
    
    # --- PASO 4: MOCK DE FEATURES ---
    from src.features import FeatureEngineer
    engineer = FeatureEngineer(integration_config)
    df_features = engineer.engineer(master_df)
    
    assert df_features.shape[0] == master_df.shape[0]
    assert df_features.shape[1] > master_df.shape[1]
    
    # --- PASO 5: VALIDACIONES FINALES ---
    # 1. El archivo maestro cleansed debe existir
    cleansed_path = os.path.join(integration_config["general"]["paths"]["cleansed"], "master_cleansed.parquet")
    assert os.path.exists(cleansed_path)

    # 2. El archivo maestro features debe existir
    features_path = os.path.join(integration_config["general"]["paths"]["features"], "master_features.parquet")
    assert os.path.exists(features_path)
    
    # 3. El archivo maestro debe ser mensual (aprox 60 meses)
    assert len(df_features) >= 59 
    
    # 4. Verificamos presencia de componentes clave (Ventas, Redes, Macro, Features)
    cols = df_features.columns
    assert "total_unidades_entregadas" in cols
    assert "month_sin" in cols
    assert "time_drift_index" in cols
    
    # 5. Verificamos que los reportes de fase se generaron en la ruta temporal
    reports_base = integration_config["general"]["paths"]["reports"]
    report_dir_v1 = os.path.join(reports_base, "phase_01_discovery")
    report_dir_v2 = os.path.join(reports_base, "phase_02_preprocessing")
    report_dir_v3 = os.path.join(reports_base, "phase_03_eda")
    report_dir_v4 = os.path.join(reports_base, "phase_04_feature_engineering")
    
    assert os.path.exists(report_dir_v1)
    assert os.path.exists(report_dir_v2)
    assert os.path.exists(report_dir_v3)
    assert os.path.exists(report_dir_v4)

    # 5. Verificar que hay figuras en EDA
    figures_base = integration_config["general"]["paths"]["figures"]
    eda_fig_dir = os.path.join(figures_base, "phase_03_eda")
    assert os.path.exists(eda_fig_dir)
    assert len(os.listdir(eda_fig_dir)) > 0
