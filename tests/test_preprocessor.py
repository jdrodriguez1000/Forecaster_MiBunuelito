import pytest
import pandas as pd
import numpy as np
import os
from src.preprocessor import Preprocessor
from src.utils.config_loader import load_config

@pytest.fixture
def config(tmp_path):
    cfg = load_config("config.yaml")
    # Redirigir salidas a carpetas temporales para no ensuciar el proyecto real
    test_reports = tmp_path / "reports"
    test_cleansed = tmp_path / "cleansed"
    test_reports.mkdir()
    test_cleansed.mkdir()
    
    cfg['general']['paths']['reports'] = str(test_reports)
    cfg['general']['paths']['cleansed'] = str(test_cleansed)
    return cfg

@pytest.fixture
def preprocessor(config):
    return Preprocessor(config)

@pytest.fixture
def mock_data():
    # Ventas Diarias
    vd = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2026-12-31"]),
        "total_unidades_entregadas": [10, 10, 15, 20],
        "unidades_precio_normal": [5, 5, 10, 15],
        "unidades_promo_pagadas": [2, 2, 2, 2],
        "unidades_promo_bonificadas": [3, 3, 3, 3],
        "precio_unitario_full": [1000, 1000, 1000, 1000],
        "costo_unitario": [500, 500, 500, 500],
        "ingresos_totales": [7000, 7000, 12000, 17000],
        "costo_total": [5000, 5000, 7500, 10000],
        "utilidad": [2000, 2000, 4500, 7000],
        "id": [1, 1, 2, 3],
        "created_at": ["2024-01-01", "2024-01-01", "2024-01-02", "2026-12-31"]
    })
    
    # Macro Economia
    me = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "ipc_mensual": [0.5, 0.6],
        "trm_promedio": [3900, 3950],
        "tasa_desempleo": [10.2, 10.1],
        "costo_insumos_index": [105, 106],
        "confianza_consumidor": [12, 13],
        "id": [1, 2],
        "created_at": ["2024-01-01", "2024-02-01"]
    })
    
    # Promocion Dia
    pd_df = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "es_promo": [1, 0],
        "id": [1, 2],
        "created_at": ["2024-01-01", "2024-01-02"]
    })
    
    # Redes Sociales
    rs = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "campaña": ["A", "B"],
        "inversion_facebook": [100.0, 200.0],
        "inversion_instagram": [50.0, 75.0],
        "inversion_total_diaria": [150.0, 275.0],
        "id": [1, 2],
        "created_at": ["2024-01-01", "2024-01-02"]
    })
    
    return {
        "ventas_diarias": vd,
        "macro_economia": me,
        "promocion_dia": pd_df,
        "redes_sociales": rs
    }

def test_step_01_validate_contracts(preprocessor, mock_data):
    # Success case
    preprocessor._step_01_validate_contracts(mock_data)
    assert "01_contract_validation" in preprocessor.report["steps_detail"]
    
    # Error case: missing column
    bad_data = {"ventas_diarias": mock_data["ventas_diarias"].drop(columns=["total_unidades_entregadas"])}
    with pytest.raises(ValueError, match="CRITICAL: Missing columns in ventas_diarias"):
        preprocessor._step_01_validate_contracts(bad_data)

def test_step_02_drop_immediate(preprocessor, mock_data):
    preprocessor._step_02_drop_immediate(mock_data)
    vd = mock_data["ventas_diarias"]
    assert "id" not in vd.columns
    assert "created_at" not in vd.columns
    assert "ingresos_totales" not in vd.columns
    
    rs = mock_data["redes_sociales"]
    assert "campaña" not in rs.columns
    assert "inversion_total_diaria" not in rs.columns

def test_step_03_anti_leakage(preprocessor, mock_data):
    # The current local time is Feb 2026 in the simulation, 
    # but the mock data has a record for Dec 2026.
    # Anti-leakage should remove Dec 2026 if today is Feb 2026.
    preprocessor._step_03_anti_leakage(mock_data)
    vd = mock_data["ventas_diarias"]
    # Check that the 2026-12-31 record is gone
    assert len(vd[vd["fecha"] == "2026-12-31"]) == 0

def test_step_04_handle_duplicates(preprocessor, mock_data):
    # mock_data has two Jan 1st records in ventas_diarias
    initial_len = len(mock_data["ventas_diarias"])
    preprocessor._step_04_handle_duplicates(mock_data)
    final_len = len(mock_data["ventas_diarias"])
    assert final_len < initial_len
    # Check uniqueness of dates
    assert mock_data["ventas_diarias"]["fecha"].is_unique

def test_step_05_sentinels_to_null(preprocessor, config):
    # Create data with sentinels
    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "valor": [-999, 10.0]
    })
    data = {"ventas_diarias": df}
    p = Preprocessor(config)
    p._step_05_sentinels_to_null(data)
    assert np.isnan(data["ventas_diarias"]["valor"][0])
    assert data["ventas_diarias"]["valor"][1] == 10.0

def test_step_06_reindex_gaps(preprocessor):
    # Create data with gap
    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-01-03"]),
        "unidades": [10, 20]
    })
    data = {"ventas_diarias": df}
    preprocessor._step_06_reindex_gaps(data)
    # Range should be 1st, 2nd, 3rd
    assert len(data["ventas_diarias"]) == 3
    assert data["ventas_diarias"]["fecha"].iloc[1] == pd.Timestamp("2024-01-02")
    assert np.isnan(data["ventas_diarias"]["unidades"].iloc[1])

def test_step_08_recalculate_target(preprocessor, mock_data):
    # Recalculate total units based on components
    # We use a clean copy of mock_data components
    df = pd.DataFrame({
        "total_unidades_entregadas": [0, 0],
        "unidades_precio_normal": [10, 20],
        "unidades_promo_pagadas": [5, 5],
        "unidades_promo_bonificadas": [5, 5]
    })
    data = {"ventas_diarias": df}
    preprocessor._step_08_recalculate_target(data)
    assert data["ventas_diarias"]["total_unidades_entregadas"].iloc[0] == 20
    assert data["ventas_diarias"]["total_unidades_entregadas"].iloc[1] == 30

def test_full_process(preprocessor, mock_data):
    # Test the entire pipeline
    master_df = preprocessor.process(mock_data)
    
    # print(f"Master columns: {master_df.columns}")
    # print(f"Master index name: {master_df.index.name}")
    
    assert isinstance(master_df, pd.DataFrame)
    # The preprocessor sets the index in step 11
    assert master_df.index.name == "fecha"
    assert "total_unidades_entregadas" in master_df.columns
    assert "ipc_mensual" in master_df.columns
    # Support columns should be gone
    assert "unidades_precio_normal" not in master_df.columns
    
    # Check report
    assert preprocessor.report["status"] == "success"
    assert "final_master" in preprocessor.report["data_profile"]
