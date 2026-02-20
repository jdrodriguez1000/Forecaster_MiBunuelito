import os
import json
import pytest
import pandas as pd
from unittest.mock import patch
from src.utils.helpers import save_report
from src.preprocessor import Preprocessor

def test_dual_persistence_logic(tmp_path):
    """
    Escenario 3: Verificación de Persistencia Dual y Trazabilidad.
    Asegura que save_report crea tanto el archivo timestamped como el latest.
    """
    report_dir = tmp_path / "reports"
    report_data = {"phase": "test", "status": "success", "metrics": {"acc": 0.95}}
    prefix = "fase_test"
    
    latest_path, hist_path = save_report(report_data, str(report_dir), prefix)
    
    # 1. Verificar existencia de ambos archivos
    assert os.path.exists(latest_path)
    assert os.path.exists(hist_path)
    
    # 2. Verificar que 'latest' apunta al prefijo correcto
    assert f"{prefix}_latest.json" in latest_path
    
    # 3. Verificar contenido
    with open(latest_path, "r") as f:
        data = json.load(f)
        assert data["metrics"]["acc"] == 0.95

def test_cascading_contract_integrity(integration_config):
    """
    Escenario 5: Integridad de Contratos en Cascada.
    Verifica que si los datos RAW guardados por la Fase 1 no cumplen el contrato
    esperado por la Fase 2, el Preprocesador lanza un error explícito.
    """
    # Creamos un dato RAW malformado (falta columna obligatoria)
    bad_raw = {
        "ventas_diarias": pd.DataFrame({"fecha": ["2024-01-01"], "unidades": [10]}) # Falta total_unidades_entregadas
    }
    
    p = Preprocessor(integration_config)
    
    with pytest.raises(ValueError, match="CRITICAL: Missing columns in ventas_diarias"):
        p.process(bad_raw)
