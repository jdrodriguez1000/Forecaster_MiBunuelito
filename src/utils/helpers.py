import os
import json
import logging
import numpy as np
from datetime import datetime

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle pandas Timestamps, datetimes, and numpy types.
    """
    def default(self, obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if hasattr(obj, 'item'): # Handle numpy scalars
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_report(report_data, output_dir, file_prefix):
    """
    Saves the report in two formats:
    1. Historical: file_prefix_YYYYMMDD_HHMMSS.json
    2. Pointer: file_prefix_latest.json
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    hist_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.json")
    latest_path = os.path.join(output_dir, f"{file_prefix}_latest.json")
    
    # Save both versions
    for path in [hist_path, latest_path]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False, cls=CustomJSONEncoder)
    
    return latest_path, hist_path
