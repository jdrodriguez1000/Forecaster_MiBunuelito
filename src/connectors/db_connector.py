import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DBConnector:
    """
    Class responsible for managing connection to Supabase via REST API.
    Uses SUPABASE_URL and SUPABASE_KEY (anon/service_role).
    """
    
    def __init__(self):
        """
        Initializes the connector by loading environment variables from project root.
        """
        # Search for .env in project root relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../.."))
        dotenv_path = os.path.join(project_root, ".env")
        
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            logger.info(f"Loaded .env from {dotenv_path}")
        else:
            load_dotenv()
            logger.info("Loaded .env from default path")
        
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            logger.error("Error: SUPABASE_URL or SUPABASE_KEY not found in .env file")
            raise EnvironmentError("Missing critical Supabase credentials (URL/KEY) in .env")

        self._client = None

    def get_client(self) -> Client:
        """
        Returns a Supabase Client. Singleton pattern.
        """
        if self._client is None:
            try:
                logger.info(f"Initializing Supabase client for {self.url}...")
                self._client = create_client(self.url, self.key)
                logger.info("Supabase client initialized successfully.")
            except Exception as e:
                logger.error(f"Critical error initializing Supabase client: {str(e)}")
                raise
        
        return self._client
