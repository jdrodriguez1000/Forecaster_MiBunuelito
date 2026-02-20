import pytest
import os
import yaml
import shutil
import pandas as pd
from unittest.mock import MagicMock, patch

@pytest.fixture
def integration_config(tmp_path):
    """
    Carga de configuración base y sobreescritura con rutas temporales para integración.
    """
    config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path, "r") as f:
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
    
    # Crear los directorios físicos
    os.makedirs(config["general"]["paths"]["raw"], exist_ok=True)
    os.makedirs(config["general"]["paths"]["cleansed"], exist_ok=True)
    os.makedirs(config["general"]["paths"]["reports"], exist_ok=True)
    
    return config

@pytest.fixture
def mock_supabase_responses():
    """
    Genera datos ficticios para las 4 tablas principales que simulan Supabase.
    """
    # Ventas Diarias (Diciembre 2023 a Enero 2024)
    ventas = pd.DataFrame({
        "fecha": pd.date_range("2023-12-01", "2024-01-31", freq="D").strftime("%Y-%m-%d"),
        "total_unidades_entregadas": [100] * 62,
        "unidades_precio_normal": [60] * 62,
        "unidades_promo_pagadas": [20] * 62,
        "unidades_promo_bonificadas": [20] * 62,
        "precio_unitario_full": [5000] * 62,
        "costo_unitario": [2000.0] * 62,
        "ingresos_totales": [400000.0] * 62,
        "costo_total": [200000.0] * 62,
        "utilidad": [200000.0] * 62
    })
    
    # Redes Sociales
    redes = pd.DataFrame({
        "fecha": pd.date_range("2023-12-01", "2024-01-31", freq="D").strftime("%Y-%m-%d"),
        "campaign": ["Navidad"] * 31 + ["Enero"] * 31,
        "inversion_facebook": [50000.0] * 62,
        "inversion_instagram": [30000.0] * 62,
        "inversion_total_diaria": [80000.0] * 62
    })
    
    # Promocion Dia
    promo = pd.DataFrame({
        "fecha": pd.date_range("2023-12-01", "2024-01-31", freq="D").strftime("%Y-%m-%d"),
        "es_promo": [0] * 62
    })
    
    # Macro Economia
    macro = pd.DataFrame({
        "fecha": ["2023-12-01", "2024-01-01"],
        "ipc_mensual": [0.5, 0.4],
        "trm_promedio": [3900.0, 3950.0],
        "tasa_desempleo": [10.2, 10.5],
        "costo_insumos_index": [115.0, 116.0],
        "confianza_consumidor": [12.0, 11.5]
    })
    
    return {
        "ventas_diarias": ventas,
        "redes_sociales": redes,
        "promocion_dia": promo,
        "macro_economia": macro
    }
