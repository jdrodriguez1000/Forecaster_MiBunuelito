import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Load project configuration from YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
