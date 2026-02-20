import pytest
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.explorer import DataExplorer
from src.utils.config_loader import load_config
import warnings

# Ignorar ruidos de librerías en los tests
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*vert.*")
warnings.filterwarnings("ignore", message=".*orientation.*")

@pytest.fixture
def config(tmp_path):
    # Cargar configuración real y redirigir rutas a carpetas temporales
    cfg = load_config("config.yaml")
    
    test_reports = tmp_path / "reports"
    test_figures = tmp_path / "figures"
    test_reports.mkdir()
    test_figures.mkdir()
    
    cfg['general']['paths']['reports'] = str(test_reports)
    cfg['general']['paths']['figures'] = str(test_figures)
    
    # Asegurar que los parámetros de EDA existan para el test
    cfg['eda']['partitioning']['test_size'] = 2
    cfg['eda']['partitioning']['val_size'] = 2
    
    return cfg

@pytest.fixture
def explorer(config):
    return DataExplorer(config)

@pytest.fixture
def mock_master_df():
    # Crear un dataset mensual ficticio con al menos 60 meses para satisfacer requisitos de statsmodels
    # (PACF necesita 2*lags, Decomposición necesita 2*periodo después del split)
    dates = pd.date_range(start="2018-01-01", periods=60, freq="MS")
    # Simular una serie con tendencia y estacionalidad
    t = np.arange(60)
    ventas = 100 + 2*t + 20*np.sin(2*np.pi*t/12) + np.random.normal(0, 2, 60)
    
    df = pd.DataFrame({
        "total_unidades_entregadas": ventas
    }, index=dates)
    df.index.name = "fecha"
    return df

def test_step_01_split_data(explorer, mock_master_df):
    train, val, test = explorer._step_01_split_data(mock_master_df)
    
    assert len(test) == explorer.config["eda"]["partitioning"]["test_size"]
    assert len(val) == explorer.config["eda"]["partitioning"]["val_size"]
    assert len(train) == len(mock_master_df) - len(test) - len(val)
    assert "splitting" in explorer.report["results"]

def test_step_02_profile_drift(explorer, mock_master_df):
    train, val, test = explorer._step_01_split_data(mock_master_df)
    explorer._step_02_profile_drift(train, val, test)
    
    assert "drift_analysis" in explorer.report["results"]
    assert "train" in explorer.report["results"]["drift_analysis"]
    assert explorer.report["results"]["drift_analysis"]["train"]["mean"] > 0

def test_step_03_analyze_calendar(explorer, mock_master_df):
    # Test planning: check if stats and plots are generated
    explorer._step_03_analyze_calendar(mock_master_df)
    
    impact = explorer.report["results"]["calendar_impact"]
    assert "month" in impact
    assert "statistics" in impact["month"]
    # Check if median is present for month 1
    assert "1" in impact["month"]["statistics"]
    assert "median" in impact["month"]["statistics"]["1"]
    assert "outliers" in impact["month"]["statistics"]["1"]

def test_step_04_decompose_series(explorer, mock_master_df):
    explorer._step_04_decompose_series(mock_master_df)
    
    decomp = explorer.report["results"]["time_series"]["decomposition"]
    assert "values" in decomp
    assert "trend" in decomp["values"]
    assert "seasonal" in decomp["values"]
    # Seasonal values should not be all None
    assert any(v is not None for v in decomp["values"]["seasonal"].values())

def test_step_05_analyze_autocorrelation(explorer, mock_master_df):
    explorer._step_05_analyze_autocorrelation(mock_master_df)
    
    auto = explorer.report["results"]["time_series"]["autocorrelation"]
    assert "acf" in auto
    assert "pacf" in auto
    assert len(auto["acf"]["values"]) == explorer.config["eda"]["time_series"]["autocorrelation"]["max_lags"] + 1

def test_step_06_verify_stationarity(explorer, mock_master_df):
    explorer._step_06_verify_stationarity(mock_master_df)
    
    stationarity = explorer.report["results"]["time_series"]["stationarity"]
    assert "p_value" in stationarity
    assert isinstance(stationarity["is_stationary"], bool)

def test_step_07_validate_business_events(explorer, mock_master_df):
    explorer._step_07_validate_business_events(mock_master_df)
    
    events = explorer.report["results"]["business_events"]
    assert "pandemia" in events
    assert "impact_pct" in events["pandemia"]

def test_full_run_eda(explorer, mock_master_df):
    report = explorer.run_eda(mock_master_df)
    assert report["status"] == "success"
    # Check if report file exists
    report_dir = os.path.join(explorer.config["general"]["paths"]["reports"], "phase_03_eda")
    assert os.path.exists(os.path.join(report_dir, "phase_03_eda_latest.json"))
