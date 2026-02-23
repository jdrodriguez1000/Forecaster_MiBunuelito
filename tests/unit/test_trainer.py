import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock, patch
from src.trainer import ForecasterTrainer
from skforecast.direct import ForecasterDirect
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

@pytest.fixture
def mock_config():
    return {
        "general": {
            "random_state": 42,
            "paths": {
                "processed": "data/02_processed",
                "reports": "outputs/reports",
                "models": "outputs/models",
                "figures": "outputs/figures"
            }
        },
        "preprocessing": {
            "target_variable": "total_unidades_entregadas",
            "differentiation": [None, 1]
        },
        "training_parameters": {
            "forecast_horizon": 6,
            "metric_for_tuning": "mean_absolute_error",
            "hyperparameter_grids": {
                "XGBRegressor": {"n_estimators": [10]},
                "LightGBM": {"n_estimators": [10]}
            },
            "grid_search_cv_params": {
                "refit": False
            }
        },
        "baseline_models": [
            {
                "name": "Seasonal Naive",
                "enabled": True,
                "description": "Baseline naive",
                "forecasting_parameters": {
                    "offset": 12,
                    "n_offsets": 1
                }
            }
        ],
        "modeling": {
            "backtesting": {
                "initial_train_size": 36,
                "test_size": 12
            },
            "experiments": [
                {
                    "name": "run01_test",
                    "description": "Test Run 01",
                    "models_to_train": ["XGBRegressor"],
                    "forecasting_parameters": {
                        "lags_grid": [[1, 2]]
                    }
                }
            ]
        }
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
    assert trainer.horizon == 6
    assert trainer.random_state == 42
    assert isinstance(trainer.all_candidates, list)

@patch("src.trainer.load_config")
def test_get_regressor(mock_load, mock_config):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    
    reg_xgb = trainer._get_regressor("XGBRegressor")
    assert isinstance(reg_xgb, XGBRegressor)
    
    reg_lgbm = trainer._get_regressor("LightGBM")
    assert isinstance(reg_lgbm, LGBMRegressor)
    
    # Returns None for unknown
    assert trainer._get_regressor("UnknownModel") is None

@patch("src.trainer.pd.read_parquet")
@patch("os.path.exists")
@patch("src.trainer.load_config")
def test_load_and_split_data(mock_load, mock_exists, mock_read, mock_config, sample_df):
    mock_load.return_value = mock_config
    mock_exists.return_value = True
    mock_read.return_value = sample_df
    
    trainer = ForecasterTrainer()
    trainer.load_and_split_data()
    
    # Check splits (12 val, 12 test)
    assert len(trainer.data_test) == 12
    assert len(trainer.data_val) == 12
    assert len(trainer.data_train) == 60 - 12 - 12
    assert trainer.report["data_summary"]["total_months"] == 60

@patch("src.trainer.load_config")
def test_run_baselines(mock_load, mock_config, sample_df):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    trainer.data_train = sample_df.iloc[:36]
    trainer.data_val = sample_df.iloc[36:48]
    trainer.data_test = sample_df.iloc[48:]
    trainer.data_train_val_test = sample_df
    
    trainer.run_baselines()
    
    assert any("Naive" in cand["name"] for cand in trainer.all_candidates)
    assert "baselines" in trainer.report

@patch("src.trainer.load_config")
def test_run_generic_experiment(mock_load, mock_config, sample_df):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    trainer.data_train = sample_df.iloc[:36]
    trainer.data_val = sample_df.iloc[36:48]
    trainer.data_test = sample_df.iloc[48:]
    trainer.data_train_val_test = sample_df
    
    exp_config = {
        "name": "test_exp",
        "description": "desc",
        "models_to_train": ["XGBRegressor"],
        "forecasting_parameters": {
            "lags_grid": [[1, 2]]
        }
    }
    
    trainer._run_generic_experiment(exp_config, is_endogenous=True)
    
    assert any(c["name"] == "test_exp_XGBRegressor" for c in trainer.all_candidates)

@patch("src.trainer.save_report")
@patch("src.trainer.load_config")
def test_save_final_report_logic(mock_load, mock_save_rep, mock_config, sample_df):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    trainer.data_train = sample_df.iloc[:36]
    trainer.data_val = sample_df.iloc[36:48]
    trainer.data_test = sample_df.iloc[48:]
    trainer.data_train_val_test = sample_df
    
    # Mock a candidate
    forecaster = MagicMock()
    # Mock backtesting_forecaster results as it's called inside save_final_report
    metrics_mock = pd.DataFrame([{"mae": 10.0, "mape": 0.1}])
    preds_mock = pd.DataFrame({"pred": [100.0]*12}, index=sample_df.index[48:])
    
    candidate = {
        "name": "Champ",
        "forecaster": forecaster,
        "metrics_val": {"mae": 5.0},
        "exog_cols": []
    }
    trainer.all_candidates = [candidate]
    
    with patch("src.trainer.backtesting_forecaster", return_value=(metrics_mock, preds_mock)):
        trainer.save_final_report()
    
    assert "champion_model" in trainer.report
    assert trainer.report["champion_model"]["name"] == "Champ"
    assert "test_comparative_table" in trainer.report["champion_model"]

@patch("src.trainer.save_model")
@patch("src.trainer.load_config")
def test_retrain_and_save_champion(mock_load, mock_save_mod, mock_config, sample_df):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    trainer.data_train = sample_df.iloc[:36]
    trainer.data_val = sample_df.iloc[36:48]
    trainer.data_test = sample_df.iloc[48:]
    
    forecaster_mock = MagicMock()
    trainer.champion_obj = {
        "name": "Winner",
        "forecaster": forecaster_mock,
        "exog_cols": []
    }
    
    trainer.retrain_and_save_champion()
    
    assert forecaster_mock.fit.called
    assert mock_save_mod.called

@patch("src.trainer.save_figure")
@patch("src.trainer.save_report")
@patch("src.trainer.load_config")
def test_generate_champion_diagnostics(mock_load, mock_save_rep, mock_save_fig, mock_config, sample_df):
    mock_load.return_value = mock_config
    trainer = ForecasterTrainer()
    
    # Setup test data
    trainer.data_train = sample_df.iloc[:36]
    trainer.data_val = sample_df.iloc[36:48]
    trainer.data_test = sample_df.iloc[48:]
    trainer.data_train_val_test = sample_df
    trainer.target = "total_unidades_entregadas"
    trainer.reports_dir = "outputs/reports"
    trainer.report["champion_model"] = {}
    
    # Mock Champion
    forecaster_mock = MagicMock()
    # Mock predict
    preds_mock = pd.Series([100.0]*len(trainer.data_test), index=trainer.data_test.index)
    forecaster_mock.predict.return_value = preds_mock
    # Mock backtesting_forecaster (nested call)
    metrics_mock = pd.DataFrame([{"mae": 10.0}])
    preds_df_mock = pd.DataFrame({
        "pred": [100.0]*len(trainer.data_test),
        "lower_bound": [90.0]*len(trainer.data_test),
        "upper_bound": [110.0]*len(trainer.data_test)
    }, index=trainer.data_test.index)
    
    # Mock get_feature_importances
    forecaster_mock.get_feature_importances.return_value = pd.DataFrame({
        "feature": ["exog1", "lag_1"],
        "importance": [0.8, 0.2]
    })
    
    trainer.champion_obj = {
        "name": "ChampPlot",
        "forecaster": forecaster_mock,
        "exog_cols": []
    }
    
    with patch("src.trainer.backtesting_forecaster", return_value=(metrics_mock, preds_df_mock)):
        trainer.generate_champion_diagnostics()
        
    # Check if report was updated with residuals and coefficients
    assert "analisis_residuos" in trainer.report["champion_model"]
    assert "magnitud_coeficientes" in trainer.report["champion_model"]
    
    # Check if figures were "saved" (mock called)
    assert mock_save_fig.called
    assert mock_save_rep.called
