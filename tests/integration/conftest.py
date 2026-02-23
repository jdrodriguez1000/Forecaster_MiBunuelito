import pytest
import os
import yaml
import shutil
import pandas as pd
from unittest.mock import MagicMock, patch
import warnings

# Ignorar ruidos de librerías en los tests
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*vert.*")
warnings.filterwarnings("ignore", message=".*orientation.*")

@pytest.fixture
def integration_config(tmp_path):
    """
    Carga de configuración base y sobreescritura con rutas temporales para integración.
    """
    config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Sobreescribir rutas para que apunten al directorio temporal
    base_data = tmp_path / "data"
    base_outputs = tmp_path / "outputs"
    base_experiments = tmp_path / "experiments"
    
    config["general"]["paths"]["raw"] = str(base_data / "01_raw")
    config["general"]["paths"]["cleansed"] = str(base_data / "02_cleansed")
    config["general"]["paths"]["features"] = str(base_data / "03_features")
    config["general"]["paths"]["processed"] = str(base_data / "04_processed")
    config["general"]["paths"]["models"] = str(base_outputs / "models")
    config["general"]["paths"]["reports"] = str(base_outputs / "reports")
    config["general"]["paths"]["figures"] = str(base_outputs / "figures")
    
    # Ajustar parámetros de EDA para que funcionen con datos de prueba
    config["eda"]["partitioning"]["test_size"] = 10
    config["eda"]["partitioning"]["val_size"] = 10
    config["eda"]["time_series"]["autocorrelation"]["max_lags"] = 10
    
    # Ajustar parámetros de modeling para velocidad en integración
    config["training_parameters"]["models_to_train"] = ["LightGBM", "Ridge"]
    config["training_parameters"]["hyperparameter_grids"] = {
        "LightGBM": {"n_estimators": [5, 10], "learning_rate": [0.1]},
        "Ridge": {"alpha": [1.0]}
    }
    config["training_parameters"]["grid_search_cv_params"]["n_iter"] = 1
    
    # Simplificar experimentos en config
    if "experiments" in config:
        for exp in config["experiments"]:
            exp["enabled"] = False # Deshabilitar todos por defecto
        
        # Habilitar solo un experimento simple para test
        config["experiments"][0]["enabled"] = True
        config["experiments"][0]["models_to_train"] = ["LightGBM"]
        config["experiments"][0]["forecasting_parameters"]["lags_grid"] = [[1, 2]]
    
    return config

@pytest.fixture
def mock_supabase_responses():
    """
    Genera datos ficticios para las 4 tablas principales que simulan Supabase.
    Generamos 60 meses para permitir que las pruebas de EDA (descomposición, etc.) funcionen.
    """
    periods_days = 60 * 30 # Aprox
    dates_daily = pd.date_range("2019-01-01", periods=periods_days, freq="D")
    
    # Ventas Diarias
    ventas = pd.DataFrame({
        "fecha": dates_daily.strftime("%Y-%m-%d"),
        "total_unidades_entregadas": [100 + i%10 for i in range(len(dates_daily))],
        "unidades_precio_normal": [60] * len(dates_daily),
        "unidades_promo_pagadas": [20] * len(dates_daily),
        "unidades_promo_bonificadas": [20] * len(dates_daily),
        "precio_unitario_full": [5000] * len(dates_daily),
        "costo_unitario": [2000.0] * len(dates_daily),
        "ingresos_totales": [400000.0] * len(dates_daily),
        "costo_total": [200000.0] * len(dates_daily),
        "utilidad": [200000.0] * len(dates_daily)
    })
    
    # Redes Sociales
    redes = pd.DataFrame({
        "fecha": dates_daily.strftime("%Y-%m-%d"),
        "campaign": ["Camp"] * len(dates_daily),
        "inversion_facebook": [50000.0] * len(dates_daily),
        "inversion_instagram": [30000.0] * len(dates_daily),
        "inversion_total_diaria": [80000.0] * len(dates_daily)
    })
    
    # Promocion Dia
    promo = pd.DataFrame({
        "fecha": dates_daily.strftime("%Y-%m-%d"),
        "es_promo": [0] * len(dates_daily)
    })
    
    # Macro Economia (Mensual)
    dates_monthly = pd.date_range("2019-01-01", periods=60, freq="MS")
    macro = pd.DataFrame({
        "fecha": dates_monthly.strftime("%Y-%m-%d"),
        "ipc_mensual": [0.5] * 60,
        "trm_promedio": [3900.0] * 60,
        "tasa_desempleo": [10.2] * 60,
        "costo_insumos_index": [115.0] * 60,
        "confianza_consumidor": [12.0] * 60
    })
    
    return {
        "ventas_diarias": ventas,
        "redes_sociales": redes,
        "promocion_dia": promo,
        "macro_economia": macro
    }
