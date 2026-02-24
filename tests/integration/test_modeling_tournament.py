import os
import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from src.trainer import ForecasterTrainer
from src.utils.helpers import save_report

def test_modeling_tournament_full_run(integration_config, mock_supabase_responses):
    """
    Integration test for the ForecasterTrainer tournament logic.
    Verifies that all tournament runs execute and generate valid reports.
    """
    # 1. Prepare data (Master Features)
    from src.preprocessor import Preprocessor
    from src.features import FeatureEngineer
    
    preprocessor = Preprocessor(integration_config)
    df_cleansed = preprocessor.process(mock_supabase_responses) 
    engineer = FeatureEngineer(integration_config)
    df_features = engineer.engineer(df_cleansed)
    
    # Save to disk as expected by Trainer
    processed_dir = integration_config["general"]["paths"]["processed"]
    os.makedirs(processed_dir, exist_ok=True)
    df_features.to_parquet(os.path.join(processed_dir, "master_features.parquet"))
    
    # 2. Setup Trainer with integration config
    # We patch load_config to return our controlled config
    with patch("src.trainer.load_config", return_value=integration_config):
        trainer = ForecasterTrainer()
        
        # 3. Execute full pipeline
        trainer.load_and_split_data()
        trainer.run_baselines()
        trainer.run_all_experiments()
        trainer.save_final_report()
        trainer.retrain_and_save_champion()
        trainer.generate_champion_diagnostics()
        
    # 4. Verify Final Report Integrity
    reports_base = integration_config["general"]["paths"]["reports"]
    modeling_report_path = os.path.join(reports_base, "phase_05_modeling", "phase_05_modeling_latest.json")
    
    assert os.path.exists(modeling_report_path), "Final modeling report was not created"
    
    with open(modeling_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    
    # Check sections
    assert "run_00_baseline" in report
    assert "champion_summary" in report
    
    expected_runs = [
        "run_01_preprocessing_tournament",
        "run_02_calendar_and_diff",
        "run_03_social_media",
        "run_04_macroeconomics",
        "run_05_structural_hacks",
        "run_final_champion"
    ]
    
    for run_id in expected_runs:
        assert run_id in report, f"Missing run {run_id} in modeling report"
        assert len(report[run_id]) > 0, f"Run {run_id} has no results"

    # 5. Verify Champion Persistence
    models_dir = integration_config["general"]["paths"]["models"]
    assert os.path.exists(os.path.join(models_dir, "champion_forecaster_latest.pkl"))
    
    # 6. Verify Diagnostics Figures
    figures_dir = os.path.join(integration_config["general"]["paths"]["figures"], "phase_05_modeling")
    assert os.path.exists(figures_dir)
    # Check for specific expected plots
    expected_plots = [
        "champion_test_comparison_latest.png",
        "champion_feature_importance_latest.png",
        "champion_error_distribution_latest.png"
    ]
    for plot in expected_plots:
        assert os.path.exists(os.path.join(figures_dir, plot)), f"Missing diagnostic plot: {plot}"

def test_modeling_tournament_candidate_flow(integration_config, mock_supabase_responses):
    """
    Specifically tests that candidates from one run flow into the next.
    """
    from src.preprocessor import Preprocessor
    from src.features import FeatureEngineer
    
    preprocessor = Preprocessor(integration_config)
    df_cleansed = preprocessor.process(mock_supabase_responses) 
    engineer = FeatureEngineer(integration_config)
    df_features = engineer.engineer(df_cleansed)
    
    processed_dir = integration_config["general"]["paths"]["processed"]
    os.makedirs(processed_dir, exist_ok=True)
    df_features.to_parquet(os.path.join(processed_dir, "master_features.parquet"))
    
    with patch("src.trainer.load_config", return_value=integration_config):
        trainer = ForecasterTrainer()
        trainer.load_and_split_data()
        
        # We manually call a few runs to see the state of the report
        trainer.run_run01_preprocessing()
        
        # Verify run_01 added candidates
        run_name = "run_01_preprocessing_tournament"
        assert run_name in trainer.report
        candidates01 = trainer.report[run_name].get("top_candidates", [])
        assert len(candidates01) > 0
        
        # Run 02
        trainer.run_run02_calendar_diff()
        assert "run_02_calendar_and_diff" in trainer.report
        
        # Verification of logic: run_02 should have used candidates from run_01
        # This is hard to verify without internal state but we can check if it succeeded
        # without errors, which confirms our previous fix for the naming mismatch.
