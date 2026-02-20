import pytest
import pandas as pd
import numpy as np
from src.validator import BusinessValidator

@pytest.fixture
def base_data():
    ventas = pd.DataFrame({
        "total_unidades_entregadas": [10, 20],
        "unidades_precio_normal": [5, 10],
        "unidades_promo_pagadas": [2.5, 5],
        "unidades_promo_bonificadas": [2.5, 5],
        "precio_unitario_full": [1000.0, 1000.0],
        "costo_unitario": [500.0, 500.0],
        "ingresos_totales": [7500.0, 15000.0], # (5+2.5)*1000
        "costo_total": [5000.0, 10000.0], # 10*500
        "utilidad": [2500.0, 5000.0]     # 7500-5000
    })
    
    redes = pd.DataFrame({
        "inversion_total_diaria": [50.0],
        "inversion_facebook": [25.0],
        "inversion_instagram": [25.0]
    })
    
    return {"ventas_diarias": ventas, "redes_sociales": redes}

def test_validator_success(base_data):
    """Prueba que el validador aprueba datos que cumplen todas las reglas."""
    validator = BusinessValidator()
    report = validator.validate_all(base_data)
    
    # Verificaciones para ventas_diarias
    v_results = report["tables"]["ventas_diarias"]
    assert v_results["total_units_consistency"]["success"]
    assert v_results["promo_balance_consistency"]["success"]
    assert v_results["income_logic"]["success"]
    assert v_results["cost_logic"]["success"]
    assert v_results["utility_logic"]["success"]

    # Verificaciones para redes_sociales
    r_results = report["tables"]["redes_sociales"]
    assert r_results["investment_consistency"]["success"]

def test_validator_income_logic_fail(base_data):
    """Prueba la detección de errores en el cálculo de ingresos."""
    data = base_data.copy()
    # Provocamos error en ingresos (debería ser 7500)
    data["ventas_diarias"].loc[0, "ingresos_totales"] = 8000.0
    
    validator = BusinessValidator()
    report = validator.validate_all(data)
    
    res = report["tables"]["ventas_diarias"]["income_logic"]
    assert res["success"] is False
    assert res["errors_count"] == 1
    assert res["max_diff"] == 500.0

def test_validator_redes_sum_fail(base_data):
    """Prueba la detección de errores en la suma de inversión de redes."""
    data = base_data.copy()
    # Provocamos error en suma (debería ser 50)
    data["redes_sociales"].loc[0, "inversion_total_diaria"] = 100.0
    
    validator = BusinessValidator()
    report = validator.validate_all(data)
    
    res = report["tables"]["redes_sociales"]["investment_consistency"]
    assert res["success"] is False
    assert res["errors_count"] == 1
    assert res["max_diff"] == 50.0

def test_validator_missing_table():
    """Prueba que el validador ignora tablas no configuradas."""
    data = {"unknown_table": pd.DataFrame({"a": [1]})}
    validator = BusinessValidator()
    report = validator.validate_all(data)
    
    assert "unknown_table" not in report["tables"]
