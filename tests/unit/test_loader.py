import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import src.loader

@pytest.fixture
def mock_config():
    return {
        "general": {
            "paths": {
                "raw": "data/01_raw",
                "reports": "outputs/reports",
                "experiments": {"phase_01": "experiments/phase_01/artifacts"}
            }
        },
        "extractions": {
            "tables": {
                "ventas_diarias": {
                    "table_name": "ventas_diarias",
                    "date_column": "fecha",
                    "contract": {"fecha": "datetime", "valor": "float"}
                }
            },
            "sentinels": {"numeric": [-999], "categorical": ["NA"], "datetime": [], "boolean": []}
        }
    }

def test_dataloader_init(mock_config):
    """Prueba que el DataLoader se inicializa y resuelve rutas correctamente."""
    with patch("src.loader.load_config", return_value=mock_config):
        with patch("src.utils.encoding_patch.apply_utf8_patch"):
            loader = src.loader.DataLoader()
            assert loader.config == mock_config
            assert os.path.exists(loader.project_root)

def test_fetch_all_with_pagination(mock_config):
    """Prueba la paginación recursiva de Supabase."""
    with patch("src.loader.load_config", return_value=mock_config):
        with patch("src.loader.DBConnector") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value.get_client.return_value = mock_client
            
            # Setup mock for pagination
            mock_execute = MagicMock()
            mock_execute.side_effect = [
                MagicMock(data=[{"id": i} for i in range(1000)]),
                MagicMock(data=[{"id": i} for i in range(1000, 1500)])
            ]
            mock_client.table.return_value.select.return_value.range.return_value.execute = mock_execute
            
            loader = src.loader.DataLoader()
            df = loader._fetch_all_with_pagination(mock_client, "test_table")
            
            assert len(df) == 1500
            assert mock_execute.call_count == 2

def test_load_and_audit_incremental(mock_config):
    """Prueba que la carga incremental funciona y maneja duplicados."""
    with patch("src.loader.load_config", return_value=mock_config):
        # Evitar inicializar BD real
        with patch("src.loader.DBConnector"): 
            loader = src.loader.DataLoader()
            
            # Mock de archivos locales
            with patch("os.path.exists", return_value=True):
                with patch("pandas.read_parquet") as mock_read:
                    df_local = pd.DataFrame({"fecha": ["2023-01-01"], "valor": [10]})
                    mock_read.return_value = df_local
                    
                    # Mock de datos nuevos
                    df_new = pd.DataFrame({"fecha": ["2023-01-01", "2023-01-02"], "valor": [20, 30]})
                    with patch.object(loader, "_fetch_all_with_pagination", return_value=df_new):
                        with patch("pandas.DataFrame.to_parquet"):
                            report = loader.load_and_audit()
                            assert "ventas_diarias" in report["tables"]
                            # La lógica de auditoría debe ser exitosa
                            assert report["tables"]["ventas_diarias"]["structural_integrity"]["contract_fulfilled"]

def test_check_table_sync(mock_config):
    """Prueba que el detector de desincronía identifica fechas diferentes."""
    with patch("src.loader.load_config", return_value=mock_config):
        loader = src.loader.DataLoader()
        
        # Simular reportes con fechas sincronizadas
        sync_reports = {
            "table1": {"time_series_health": {"date_range": {"max": "2023-01-01"}}},
            "table2": {"time_series_health": {"date_range": {"max": "2023-01-01"}}}
        }
        res_ok = loader._check_table_sync(sync_reports)
        assert res_ok["synchronized"] is True
        assert res_ok["status"] == "OK"
        
        # Simular reportes con fechas desincronizadas
        async_reports = {
            "table1": {"time_series_health": {"date_range": {"max": "2023-01-01"}}},
            "table2": {"time_series_health": {"date_range": {"max": "2023-02-01"}}}
        }
        res_fail = loader._check_table_sync(async_reports)
        assert res_fail["synchronized"] is False
        assert res_fail["status"] == "WARNING"
