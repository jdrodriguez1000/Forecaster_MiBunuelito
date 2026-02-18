import argparse
import sys
from src.utils.config_loader import load_config
from src.utils.helpers import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Mi Buñuelito Forecasting Orchestrator")
    parser.add_argument("--phase", type=str, required=True, help="Execution phase (discovery, preprocessing, eda, features, modeling)")
    
    args = parser.parse_args()
    config = load_config()
    setup_logging()
    
    print(f"Starting phase: {args.phase}")
    # Logic to orchestrate phases here

if __name__ == "__main__":
    main()
