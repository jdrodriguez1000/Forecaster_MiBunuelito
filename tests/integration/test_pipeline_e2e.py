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
        
    # --- PASO 3: VALIDACIONES FINALES ---
    # 1. El archivo maestro debe existir
    cleansed_path = os.path.join(integration_config["general"]["paths"]["cleansed"], "master_cleansed.parquet")
    assert os.path.exists(cleansed_path)
    
    # 2. El archivo maestro debe ser mensual (Diciembre 2023 y Enero 2024)
    assert len(master_df) == 2
    
    # 3. Verificamos presencia de componentes clave (Ventas, Redes, Macro)
    cols = master_df.columns
    assert "total_unidades_entregadas" in cols
    assert "inversion_facebook" in cols
    assert "ipc_mensual" in cols
    
    # 4. Verificamos que los reportes de fase se generaron en la ruta temporal
    reports_base = integration_config["general"]["paths"]["reports"]
    report_dir_v1 = os.path.join(reports_base, "phase_01_discovery")
    report_dir_v2 = os.path.join(reports_base, "phase_02_preprocessing")
    
    # Nota: Preprocessor.py guarda el reporte internamente, verificamos carpeta
    assert os.path.exists(report_dir_v1), f"Discovery report dir missing: {report_dir_v1}"
    assert os.path.exists(report_dir_v2), f"Preprocessing report dir missing: {report_dir_v2}"
