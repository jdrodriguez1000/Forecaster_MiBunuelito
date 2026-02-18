import os
import pytest
from src.connectors.db_connector import DBConnector
from supabase import Client
from unittest.mock import patch

def test_db_connector_init_success():
    """Prueba que el conector se inicializa correctamente con variables de entorno presentes."""
    with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"}):
        connector = DBConnector()
        assert connector.url == "https://test.supabase.co"
        assert connector.key == "test-key"

def test_db_connector_init_failure():
    """Prueba que falla si faltan las variables de entorno."""
    # Limpiamos el entorno y evitamos que cargue el .env real
    with patch.dict(os.environ, {}, clear=True):
        with patch("src.connectors.db_connector.load_dotenv", return_value=None):
            with pytest.raises(EnvironmentError) as excinfo:
                DBConnector()
            assert "Missing critical Supabase credentials" in str(excinfo.value)

def test_db_connector_get_client_singleton():
    """Prueba que el patrón Singleton funciona y siempre devuelve la misma instancia de cliente."""
    with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"}):
        connector = DBConnector()
        # Mock de create_client para no hacer llamadas reales
        with patch("src.connectors.db_connector.create_client") as mock_create:
            mock_create.return_value = "ClientInstance"
            
            client1 = connector.get_client()
            client2 = connector.get_client()
            
            assert client1 == client2
            assert mock_create.call_count == 1

def test_db_connector_get_client_error():
    """Prueba que se propaga el error si falla la creación del cliente (Ej: URL mal formada)."""
    with patch.dict(os.environ, {"SUPABASE_URL": "not-a-url", "SUPABASE_KEY": "test-key"}):
        connector = DBConnector()
        with patch("src.connectors.db_connector.create_client", side_effect=ValueError("Invalid URL")):
            with pytest.raises(ValueError) as excinfo:
                connector.get_client()
            assert "Invalid URL" in str(excinfo.value)
