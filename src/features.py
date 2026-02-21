import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from src.utils.helpers import save_report

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature Engineering phase following the Production-First methodology.
    Responsible for creating cyclical, event, technical, and trend features,
    as well as projecting exogenous variables for the forecast horizon.
    """
    def __init__(self, config):
        self.config = config
        self.report = {
            "phase": "04_feature_engineering",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "steps_detail": {},
            "transformed_shape": None,
            "exogenous_created": [],
            "artifacts": {}
        }

    def engineer(self, df: pd.DataFrame):
        """
        Execute the feature engineering pipeline.
        Args:
            df (pd.DataFrame): Cleansed master dataset with datetime index.
        Returns:
            pd.DataFrame: Enriched dataset with extended horizon.
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Capturing initial metadata for the report (Requirement 2)
        self.report["initial_metadata"] = {
            "column_names": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        df_eng = df.copy()

        try:
            # Step 1: Add time-based cyclical features
            df_eng = self._step_01_cyclical_features(df_eng)

            # Step 2: Add business event features (Pandemic, Novenas, Bonus)
            df_eng = self._step_02_event_features(df_eng)

            # Step 3: Add technical calendar features (days, weekends)
            df_eng = self._step_03_calendar_technical(df_eng)

            # Step 4: Add trend features
            df_eng = self._step_04_trend_features(df_eng)

            # Step 6: Final check and artifact generation
            df_eng = self._step_06_generate_artifacts(df_eng)

            return df_eng

        except Exception as e:
            self.report["status"] = "error"
            self.report["error_message"] = str(e)
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

    def _step_01_cyclical_features(self, df):
        logger.info("Step 1: Adding cyclical features...")
        cyclical_cfg = self.config['features']['engineering']['cyclical']
        
        # Monthly
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / cyclical_cfg['month']['period'])
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / cyclical_cfg['month']['period'])
        
        # Quarterly
        df['quarter'] = df.index.quarter
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / cyclical_cfg['quarter']['period'])
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / cyclical_cfg['quarter']['period'])
        
        # Semester (Cyclicity of 2)
        semester = np.where(df.index.month <= 6, 1, 2)
        df['semester_sin'] = np.sin(2 * np.pi * semester / cyclical_cfg['semester']['period'])
        df['semester_cos'] = np.cos(2 * np.pi * semester / cyclical_cfg['semester']['period'])

        # Cleanup numeric support columns
        df = df.drop(columns=['month', 'quarter'])
        
        self.report["steps_detail"]["01_cyclical_features"] = "Added month, quarter, and semester sin/cos features."
        return df

    def _step_02_event_features(self, df):
        logger.info("Step 2: Adding business event features...")
        events_cfg = self.config['features']['engineering']['events']
        
        # Pandemic Flag
        p_start = pd.to_datetime(events_cfg['pandemia']['start'])
        p_end = pd.to_datetime(events_cfg['pandemia']['end'])
        df['is_pandemic'] = ((df.index >= p_start) & (df.index <= p_end)).astype(int)
        
        # Novenas Intensity (Monthly weight)
        novenas_cfg = events_cfg['novenas']
        days_weight = novenas_cfg['total_days'] / 31.0 # Prorated impact
        df['novenas_intensity'] = 0.0
        df.loc[df.index.month == novenas_cfg['month'], 'novenas_intensity'] = days_weight
        
        # Bonus Months (Jun, Dec)
        df['is_bonus_month'] = df.index.month.isin(events_cfg['bonus_months']).astype(int)
        
        self.report["steps_detail"]["02_event_features"] = "Added is_pandemic, novenas_intensity, and is_bonus_month."
        return df

    def _step_03_calendar_technical(self, df):
        logger.info("Step 3: Adding calendar technical features...")
        tech_cfg = self.config['features']['engineering']['calendar_technical']
        
        # Days in month
        if tech_cfg['calculate_days_in_month']:
            df['days_in_month'] = df.index.days_in_month
        
        # Weekend count (Saturdays and Sundays)
        if tech_cfg['calculate_weekend_days']:
            # Using a lambda to count weekends in each month
            df['weekend_days_count'] = df.index.map(lambda d: len([1 for i in range(1, d.days_in_month + 1) 
                                                                  if pd.Timestamp(year=d.year, month=d.month, day=i).dayofweek >= 5]))
        
        # Holidays Count (Colombia)
        # Note: In a production environment, we should use the 'holidays' library for precision.
        # For now, we use a monthly estimation or fixed holiday counts per month as a baseline.
        # Fixed holidays in Colombia (aprox): Ene:2, Mar:1, Abr:2, May:2, Jun:2, Jul:2, Ago:2, Oct:1, Nov:2, Dic:2
        colombia_holidays_map = {1:2, 2:0, 3:1, 4:2, 5:2, 6:2, 7:2, 8:2, 9:0, 10:1, 11:2, 12:2}
        df['holidays_count'] = df.index.month.map(colombia_holidays_map)
        
        self.report["steps_detail"]["03_calendar_technical"] = "Added days_in_month, weekend_days_count, and (basic) holidays_count."
        return df

    def _step_04_trend_features(self, df):
        logger.info("Step 4: Adding trend features...")
        if self.config['features']['engineering']['trend']['include_time_drift']:
            # Time index to capture long term drift/growth
            df['time_drift_index'] = np.arange(len(df))
            
        self.report["steps_detail"]["04_trend_features"] = "Added time_drift_index."
        return df

    def _step_05_project_exogenous(self, df):
        logger.info("Step 5: Projecting exogenous variables (recursive moving average)...")
        proj_cfg = self.config['features']['projection']
        horizon = self.config['general']['horizon']
        cols_to_project = proj_cfg['columns_to_project']
        
        # Current index end
        last_date = df.index.max()
        future_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
        
        # Create empty future dataframe
        future_df = pd.DataFrame(index=future_index)
        
        # Concatenate for projection
        full_df = pd.concat([df, future_df], sort=False)
        
        # Project each column
        for col in cols_to_project:
            if col not in full_df.columns:
                logger.warning(f"Exogenous column {col} not found in dataset. Skipping projection.")
                continue
            
            # Recursive Moving Average logic
            for i in range(len(df), len(full_df)):
                # Take window_size observations before current index
                window = full_df[col].iloc[i - proj_cfg['window_size'] : i]
                full_df.iloc[i, full_df.columns.get_loc(col)] = window.mean()
        
        # Re-apply non-projected features to the future rows
        # (These are deterministic features like sin/cos, days in month, etc.)
        full_df.index.name = 'fecha'
        
        # Recalculate deterministic features for the whole range including future
        full_df = self._step_01_cyclical_features(full_df)
        full_df = self._step_02_event_features(full_df)
        full_df = self._step_03_calendar_technical(full_df)
        full_df = self._step_04_trend_features(full_df)
        
        # Special case: es_promo and investments
        # We assume 0 for future if not provided (Simplification for now)
        other_exog = ['es_promo', 'inversion_facebook', 'inversion_instagram', 'precio_unitario_full', 'costo_unitario']
        for col in other_exog:
            if col in full_df.columns:
                # Fill future NaNs with 0 or last value
                if col in ['precio_unitario_full', 'costo_unitario']:
                    full_df[col] = full_df[col].ffill()
                else:
                    full_df[col] = full_df[col].fillna(0)

        self.report["steps_detail"]["05_projection"] = f"Projected {cols_to_project} for {horizon} months."
        return full_df

    def _step_06_generate_artifacts(self, df):
        logger.info("Step 6: Generating artifacts...")
        # Adhering to Rule 6: Segregation of outputs
        output_dir = self.config['general']['paths']['features']
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "master_features.parquet")
        df.to_parquet(output_path)
        
        # Capturing final metadata for the report (Requirement 3)
        self.report["final_metadata"] = {
            "column_names": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }

        # Row previews (Requirement 4)
        # We reset index to make 'fecha' a column for serialization in JSON
        df_preview = df.reset_index().copy()
        df_preview['fecha'] = df_preview['fecha'].dt.strftime('%Y-%m-%d')
        
        self.report["data_preview"] = {
            "first_3": df_preview.head(3).to_dict(orient='records'),
            "last_3": df_preview.tail(3).to_dict(orient='records'),
            "random_3": df_preview.sample(3, random_state=self.config.get('general', {}).get('random_state', 42)).to_dict(orient='records')
        }

        self.report["transformed_shape"] = df.shape
        self.report["exogenous_created"] = list(df.columns)
        self.report["artifacts"]["master_features"] = output_path
        
        # Save official report
        report_dir = os.path.join(self.config['general']['paths']['reports'], "phase_04_feature_engineering")
        save_report(self.report, report_dir, "phase_04_feature_engineering")
        
        return df
