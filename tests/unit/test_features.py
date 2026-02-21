import pytest
import pandas as pd
import numpy as np
from src.features import FeatureEngineer

@pytest.fixture
def sample_config():
    return {
        "general": {
            "horizon": 6,
            "paths": {
                "features": "data/03_features",
                "reports": "outputs/reports"
            }
        },
        "features": {
            "projection": {
                "method": "recursive_moving_average",
                "window_size": 2,
                "columns_to_project": ["macro_var"]
            },
            "engineering": {
                "cyclical": {
                    "month": {"period": 12},
                    "quarter": {"period": 4},
                    "semester": {"period": 2}
                },
                "events": {
                    "pandemia": {"start": "2020-04-01", "end": "2021-05-31"},
                    "novenas": {"month": 12, "start_day": 16, "end_day": 23, "total_days": 8},
                    "bonus_months": [6, 12]
                },
                "calendar_technical": {
                    "calculate_days_in_month": True,
                    "calculate_weekend_days": True,
                    "country_holidays": "Colombia"
                },
                "trend": {
                    "include_time_drift": True
                }
            }
        }
    }

@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2019-01-01", periods=24, freq="MS")
    df = pd.DataFrame({
        "total_unidades_entregadas": np.random.randint(100, 1000, size=24),
        "macro_var": np.linspace(10, 20, 24),
        "es_promo": [0]*24,
        "precio_unitario_full": [100.0]*24
    }, index=dates)
    df.index.name = "fecha"
    return df

def test_cyclical_features(sample_config, sample_data):
    engineer = FeatureEngineer(sample_config)
    # Step 1 test
    df_transformed = engineer._step_01_cyclical_features(sample_data.copy())
    assert "month_sin" in df_transformed.columns
    assert "month_cos" in df_transformed.columns
    assert "quarter_sin" in df_transformed.columns
    assert "semester_sin" in df_transformed.columns
    # Check Jan (Month 1): sin(2*pi*1/12) should be > 0 (0.5)
    assert df_transformed.loc["2019-01-01", "month_sin"] == pytest.approx(0.5)

def test_event_features(sample_config, sample_data):
    engineer = FeatureEngineer(sample_config)
    df_transformed = engineer._step_02_event_features(sample_data.copy())
    
    # 2020-04-01 is pandemic
    assert df_transformed.loc["2020-04-01", "is_pandemic"] == 1
    # 2019-01-01 is not pandemic
    assert df_transformed.loc["2019-01-01", "is_pandemic"] == 0
    
    # Novenas intensity in Dec
    expected_weight = 8 / 31.0
    assert df_transformed.loc["2019-12-01", "novenas_intensity"] == pytest.approx(expected_weight)
    
    # Bonus months (Jun, Dec)
    assert df_transformed.loc["2019-06-01", "is_bonus_month"] == 1
    assert df_transformed.loc["2019-07-01", "is_bonus_month"] == 0

def test_engineer_pipeline_no_extension(sample_config, sample_data):
    """
    Test that the full engineer() pipeline preserves the original length 
    as per user requirement.
    """
    engineer = FeatureEngineer(sample_config)
    df_final = engineer.engineer(sample_data.copy())
    
    # Must preserve original length
    assert len(df_final) == len(sample_data)
    # Must have new features
    assert "month_sin" in df_final.columns
    assert "time_drift_index" in df_final.columns

def test_internal_projection_logic(sample_config, sample_data):
    """
    Test the internal projection method directly, even if not called 
    by the main engineer() pipeline yet.
    """
    engineer = FeatureEngineer(sample_config)
    df_projected = engineer._step_05_project_exogenous(sample_data.copy())
    
    # Original 24 + Horizon 6 = 30
    assert len(df_projected) == 30
    
    # Check macro_var is projected and not null
    assert not df_projected["macro_var"].isnull().any()
    
    # Deterministic features should also exist for future rows
    last_date = df_projected.index.max()
    assert df_projected.loc[last_date, "month_sin"] is not None
