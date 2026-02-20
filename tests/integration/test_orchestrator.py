import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from main import main

def test_orchestrator_phase_routing(integration_config, mock_supabase_responses):
    """
    Escenario 4: Sincronización del Orchestrator.
    Verifica que al pasar --phase discovery el orquestador ejecuta esa fase
    y genera los artefactos esperados en las rutas temporales.
    """
    
    # Preparamos los argumentos de CLI
    test_args = ["--phase", "discovery"]
    
    # Patch load_config globally in relevant modules
    with patch("src.utils.config_loader.load_config", return_value=integration_config), \
         patch("src.loader.load_config", return_value=integration_config), \
         patch("main.load_config", return_value=integration_config):
        
        with patch("src.loader.DataLoader._fetch_all_with_pagination") as mock_fetch:
            # El mock devuelve el dataframe correspondiente según los tests
            def side_effect(*args, **kwargs):
                # args puede contener (self, supabase, table_name) o (supabase, table_name)
                # según cómo pytest/unittest maneje el parcheo del método de clase.
                # En ambos casos, el nombre de la tabla suele estar al final de los posicionales requeridos.
                table_name = args[2] if len(args) > 2 else args[1]
                return mock_supabase_responses[table_name]
            mock_fetch.side_effect = side_effect
            
            # Ejecutamos main pasando los argumentos directamente
            main(test_args)
            
    # Verificamos que se generó el reporte de discovery
    report_dir = os.path.join(integration_config["general"]["paths"]["reports"], "phase_01_discovery")
    assert os.path.exists(os.path.join(report_dir, "phase_01_discovery_latest.json"))

def test_orchestrator_invalid_phase():
    """
    Verifica que el orquestador falla ante una fase inexistente (validación de argparse).
    """
    test_args = ["--phase", "inexistente"]
    with pytest.raises(SystemExit): 
        main(test_args)
