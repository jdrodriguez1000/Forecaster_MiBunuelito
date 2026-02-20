import os
import pytest
import pandas as pd
from datetime import datetime
from src.preprocessor import Preprocessor

def test_anti_leakage_strict_compliance(integration_config):
    """
    Escenario 2: Prueba de la Regla de Oro (Anti-Leakage).
    Asegura que registros del mes en curso o futuros sean eliminados 
    antes de llegar al master dataset.
    """
    
    # Preparamos datos con un registro del mes pasado (válido) y uno de hoy (inválido)
    today = datetime.now()
    last_month = today.replace(day=1) - pd.Timedelta(days=1)
    
    # Simulación de Ventas Diarias
    ventas = pd.DataFrame({
        "fecha": [last_month.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")],
        "total_unidades_entregadas": [100, 200],
        "unidades_precio_normal": [100, 200],
        "unidades_promo_pagadas": [0, 0],
        "unidades_promo_bonificadas": [0, 0],
        "precio_unitario_full": [5000, 5000],
        "costo_unitario": [2000, 2000],
        "ingresos_totales": [500000, 1000000],
        "costo_total": [200000, 400000],
        "utilidad": [300000, 600000]
    })
    
    # Otras tablas mínimas requeridas para el merge (deben cumplir el contrato de config.yaml)
    redes = pd.DataFrame({
        "fecha": [last_month.strftime("%Y-%m-%d")], 
        "campaign": ["Test"],
        "inversion_facebook": [0.0], 
        "inversion_instagram": [0.0],
        "inversion_total_diaria": [0.0]
    })
    
    macro = pd.DataFrame({
        "fecha": [last_month.strftime("%Y-%m-%d")], 
        "ipc_mensual": [0.5], 
        "trm_promedio": [4000.0], 
        "tasa_desempleo": [10.0], 
        "costo_insumos_index": [100.0], 
        "confianza_consumidor": [10.0]
    })
    
    promo = pd.DataFrame({
        "fecha": [last_month.strftime("%Y-%m-%d")], 
        "es_promo": [0]
    })
    
    data_dict = {
        "ventas_diarias": ventas,
        "redes_sociales": redes,
        "macro_economia": macro,
        "promocion_dia": promo
    }
    
    # Ejecutamos preprocesamiento
    preprocessor = Preprocessor(integration_config)
    master_df = preprocessor.process(data_dict)
    
    # La prueba de fuego: 
    # El índice debe contener el último día del mes pasado, pero NO hoy.
    # Como es mensual, verificamos que no exista el registro del mes actual.
    current_month_start = today.replace(day=1).strftime("%Y-%m-%d")
    
    # El master_df tiene el índice con la fecha formateada por el resample
    dates_in_master = [d.strftime("%Y-%m-%d") for d in master_df.index]
    
    assert current_month_start not in dates_in_master, f"¡ERROR! Se fugó información del mes actual: {current_month_start}"
    assert len(master_df) >= 1 # Al menos el mes pasado debe estar
