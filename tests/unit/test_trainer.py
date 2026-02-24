import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock, patch
from src.trainer import ForecasterTrainer
from skforecast.direct import ForecasterDirect
from sklearn.linear_model import Ridge

@pytest.fixture
def mock_config():
    return {
        "general": {
            "random_state": 42,
            "horizon": 6,
            "paths": {
                "processed": "data/02_processed",
                "reports": "outputs/reports",
                "models": "outputs/models",
                "figures": "outputs/figures"
            }
        },
        "preprocessing": {
            "target_variable": "total_unidades_entregadas"
        },
        "experiments": [
            {
                "name": "run_01_preprocessing_tournament",
                "enabled": True,
                "models_to_train": ["Ridge"],
                "preprocessing_options": {
                    "transformations": [None]
                },
                "forecasting_parameters": {
                    "lags_grid": [[1, 2]]
                },
                "advance_criteria": {
                    "top_n": 5
                }
            }
        ],
        "baseline_models": [
            {
                "name": "run_00_naive",
                "enabled": True,
                "forecasting_parameters": {
                    "offset": 12,
                    "n_offsets": 1
                }
            }
        ]
    }

@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2018-01-01", periods=60, freq="MS") # 5 years
    df = pd.DataFrame({
        "total_unidades_entregadas": np.random.randint(100, 1000, size=60).astype(float),
        "exog1": np.random.randn(60)
    }, index=dates)
    df.index.freq = 'MS'
    return df

@patch("src.trainer.load_config")
@patch("src.trainer.setup_logging")
def test_trainer_init(mock_setup, mock_load, mock_config):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    assert trainer.target == "total_unidades_entregadas"
    assert trainer.random_state == 42
    assert "phase" in trainer.report

@patch("src.trainer.pd.read_parquet")
@patch("os.path.exists")
@patch("src.trainer.load_config")
def test_load_and_split_data(mock_load, mock_exists, mock_read, mock_config, sample_df):
    mock_load.return_value = mock_config
    mock_exists.return_value = True
    mock_read.return_value = sample_df
    
    trainer = ForecasterTrainer()
    trainer.load_and_split_data()
    
    # Check splits (12 val, 12 test as per trainer.py logic)
    assert len(trainer.data_test) == 12
    assert len(trainer.data_val) == 12
    assert len(trainer.data_train) == 60 - 12 - 12
    assert trainer.report["data_summary"]["total_months"] == 60

@patch("src.trainer.load_config")
def test_get_feature_importances_ridge(mock_load, mock_config):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    
    # Mock a forecaster that returns coefficients (like Ridge)
    forecaster = MagicMock()
    importance_df = pd.DataFrame({
        "feature": ["lag_1", "exog1"],
        "coefficient": [10.5, -2.3]
    })
    forecaster.get_feature_importances.return_value = importance_df
    
    importances = trainer._get_feature_importances(forecaster)
    
    assert len(importances) == 2
    assert importances[0]["feature"] == "lag_1"
    assert importances[0]["magnitude"] == 10.5
    assert importances[1]["magnitude"] == 2.3

@patch("src.trainer.save_report")
@patch("src.trainer.load_config")
def test_save_intermediate_report(mock_load, mock_save_rep, mock_config):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    trainer.save_intermediate_report()
    assert mock_save_rep.called

@patch("src.trainer.save_model")
@patch("src.trainer.load_config")
def test_retrain_and_save_champion_minimal(mock_load, mock_save_mod, mock_config, sample_df):
    mock_load.return_value = mock_config
    mock_save_mod.return_value = ("latest_path.pkl", "hist_path.pkl")
    trainer = ForecasterTrainer()
    trainer.data_train = sample_df.iloc[:36]
    trainer.data_val = sample_df.iloc[36:48]
    trainer.data_test = sample_df.iloc[48:]
    trainer.target = "total_unidades_entregadas"
    
    # Setup state in "champion_summary" as expected by trainer.py
    trainer.report["champion_summary"] = {
        "model_name": "Ridge",
        "original_model": "Ridge",
        "features_used": [],
        "lags": [1],
        "params": {},
        "transformation": None,
        "differentiation": 0,
        "used_era_weights": False
    }
    
    with patch("src.trainer.ForecasterDirect") as mock_fd:
        fd_instance = mock_fd.return_value
        # Mock fit
        fd_instance.fit.return_value = None
        # Mock predict and predict_interval
        test_index = trainer.data_test.index
        fd_instance.predict.return_value = pd.Series([100.0]*12, index=test_index)
        fd_instance.predict_interval.return_value = pd.DataFrame({
            "pred": [100.0]*12,
            "lower_bound": [90.0]*12,
            "upper_bound": [110.0]*12
        }, index=test_index)
        # Mock get_feature_importances for steps inside retrain_and_save_champion
        fd_instance.get_feature_importances.return_value = pd.DataFrame({
            "feature": ["intercept"], # Minimal
            "importance": [1.0]
        })
        
        trainer.retrain_and_save_champion()
        
    assert mock_save_mod.called
    assert "final_model_artifacts" in trainer.report

@patch("src.trainer.save_report")
@patch("src.trainer.load_config")
def test_save_final_report_logic(mock_load, mock_save_rep, mock_config):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    
    # Mock a final run result
    trainer.report["run_final_champion"] = {
        "top_candidates": [{
            "model_name": "WinnerRidge", 
            "original_model": "Ridge",
            "mae": 5.0, 
            "mape": 0.08,
            "features_used": ["lag_1"],
            "params": {}, 
            "lags": [1], 
            "transformation": None, 
            "differentiation": 0,
            "weights_applied": True
        }]
    }
    
    trainer.save_final_report()
    
    assert "champion_summary" in trainer.report
    assert trainer.report["champion_summary"]["mae"] == 5.0
    assert trainer.report["champion_summary"]["mape"] == 0.08
    assert trainer.report["champion_summary"]["used_era_weights"] == True
    assert mock_save_rep.called

@patch("src.trainer.load_config")
def test_run_preprocessing_tournament_minimal(mock_load, mock_config, sample_df):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    trainer.data_train = sample_df.iloc[:36]
    trainer.data_val = sample_df.iloc[36:48]
    trainer.data_test = sample_df.iloc[48:]
    
    # We need to mock skforecast.model_selection.backtesting_forecaster
    # because run_run01_preprocessing calls it.
    with patch("src.trainer.ForecasterDirect") as mock_fd, \
         patch("src.trainer.TimeSeriesFold") as mock_tsf, \
         patch("src.trainer.grid_search_forecaster") as mock_gs, \
         patch("src.trainer.backtesting_forecaster") as mock_bt, \
         patch("src.trainer.ForecasterTrainer._get_feature_importances") as mock_fi:
        
        mock_fi.return_value = [] # Minimal result
        
        # Mock results for pre-processing run
        mock_gs.return_value = pd.DataFrame({
            "lags": [[1]],
            "params": [{"alpha": 1.0}],
            "mean_absolute_error": [15.0],
        })
        
        mock_bt.return_value = (
            pd.DataFrame({"mean_absolute_error": [15.0], "mean_absolute_percentage_error": [0.15]}),
            pd.DataFrame()
        )
        
        # Mock forecaster instance lags
        mock_fd.return_value.lags = [1]
        
        trainer.run_run01_preprocessing()
    
    assert "run_01_preprocessing_tournament" in trainer.report
    results = trainer.report["run_01_preprocessing_tournament"]["all_results"]
    assert len(results) > 0
    assert results[0]["model_name"] == "Ridge"
    assert results[0]["mae"] == 15.0

