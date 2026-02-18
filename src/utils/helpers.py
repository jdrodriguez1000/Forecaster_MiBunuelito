import json
import logging
from datetime import datetime

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def save_report(report_data, path):
    """
    Save phase metrics and metadata to JSON.
    """
    with open(path, 'w') as f:
        json.dump(report_data, f, indent=4)
