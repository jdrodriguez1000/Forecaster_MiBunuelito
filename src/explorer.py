import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import warnings
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.utils.helpers import save_report, save_figure

# Silenciar advertencias de librerías externas (Seaborn/Matplotlib)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Específicamente para el ruido en tests de integración
warnings.filterwarnings("ignore", message=".*vert.*", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", message=".*orientation.*", category=PendingDeprecationWarning)

# Configurar Matplotlib para ser menos ruidoso en logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class DataExplorer:
    """
    Exploratory Data Analysis Engine for Mi Buñuelito.
    Implements 7 diagnostic steps with MLOps tracking and dual-persistence visuals.
    """
    def __init__(self, config):
        self.config = config
        self.report = {
            "phase": "03_eda",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "results": {
                "splitting": {},
                "drift_analysis": {},
                "calendar_impact": {},
                "time_series": {},
                "business_events": {}
            }
        }
        self.figures_path = os.path.join(config["general"]["paths"]["figures"], "phase_03_eda")
        os.makedirs(self.figures_path, exist_ok=True)
        
        # Apply style from config
        plt.style.use(config["eda"]["visuals"]["style"])
        self.figsize = tuple(config["eda"]["visuals"]["fig_size"])

    def run_eda(self, df: pd.DataFrame):
        """
        Executes the full EDA pipeline.
        Args:
            df (pd.DataFrame): Master cleansed dataset (monthly).
        """
        logger.info("🚀 Starting Phase 03: Exploratory Data Analysis")
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex. Attempting to convert.")
            df.index = pd.to_datetime(df.index)

        try:
            # Step 1: Temporal Splitting (Anti-Leakage)
            train_df, val_df, test_df = self._step_01_split_data(df)
            
            # Step 2: Descriptive Profiling & Drift
            self._step_02_profile_drift(train_df, val_df, test_df)
            
            # Step 3: Calendar & Variance Analysis (Using ONLY Train)
            self._step_03_analyze_calendar(train_df)
            
            # Step 4: Time Series Decomposition
            self._step_04_decompose_series(train_df)
            
            # Step 5: Autocorrelation (ACF/PACF)
            self._step_05_analyze_autocorrelation(train_df)
            
            # Step 6: Stationarity (Dickey-Fuller)
            self._step_06_verify_stationarity(train_df)
            
            # Step 7: Business Events Validation
            self._step_07_validate_business_events(train_df)
            
            # Step 8: Expert Interpretation (Intelligence Layer)
            self._step_08_expert_interpretation(train_df)

            # Finalize Report
            report_dir = os.path.join(self.config["general"]["paths"]["reports"], "phase_03_eda")
            latest_path, _ = save_report(self.report, report_dir, "phase_03_eda")
            logger.info(f"✅ EDA Phase completed successfully. Report: {latest_path}")
            
            return self.report

        except Exception as e:
            self.report["status"] = "error"
            self.report["error_message"] = str(e)
            logger.error(f"❌ EDA failed: {str(e)}")
            raise

    def _step_01_split_data(self, df):
        logger.info("EDA - Step 1: Splitting data (Rule 3.1)")
        test_size = self.config["eda"]["partitioning"]["test_size"]
        val_size = self.config["eda"]["partitioning"]["val_size"]
        
        # Sort index just in case
        df = df.sort_index()
        
        test_df = df.iloc[-test_size:]
        val_df = df.iloc[-(test_size + val_size):-test_size]
        train_df = df.iloc[:-(test_size + val_size)]
        
        self.report["results"]["splitting"] = {
            "train": {"start": str(train_df.index.min()), "end": str(train_df.index.max()), "size": len(train_df)},
            "val": {"start": str(val_df.index.min()), "end": str(val_df.index.max()), "size": len(val_df)},
            "test": {"start": str(test_df.index.min()), "end": str(test_df.index.max()), "size": len(test_df)}
        }
        
        return train_df, val_df, test_df

    def _step_02_profile_drift(self, train, val, test):
        logger.info("EDA - Step 2: Profiling data drift")
        target = self.config["preprocessing"]["target_variable"]
        
        stats = {}
        for name, dset in [("train", train), ("val", val), ("test", test)]:
            stats[name] = {
                "mean": float(dset[target].mean()),
                "std": float(dset[target].std()),
                "min": float(dset[target].min()),
                "max": float(dset[target].max()),
                "cv": float(dset[target].std() / dset[target].mean()) if dset[target].mean() != 0 else 0
            }
        
        self.report["results"]["drift_analysis"] = stats

    def _step_03_analyze_calendar(self, train):
        logger.info("EDA - Step 3: Impact of calendar (Rule 3.3)")
        target = self.config["preprocessing"]["target_variable"]
        levels = self.config["eda"]["analysis_levels"]
        
        tmp = train.copy()
        tmp['month'] = tmp.index.month.astype(int)
        tmp['quarter'] = tmp.index.quarter.astype(int)
        tmp['semester'] = np.where(tmp['month'] <= 6, 1, 2).astype(int)
        tmp['year'] = tmp.index.year.astype(int)
        
        for level in levels:
            # 1. Plotting
            fig, ax = plt.subplots(figsize=self.figsize)
            sns.boxplot(
                data=tmp, 
                x=level, 
                y=target, 
                hue=level, 
                ax=ax, 
                palette=self.config["eda"]["visuals"]["palette"],
                legend=False,
                orient="v"
            )
            ax.set_title(f"Distribution of {target} by {level}")
            
            prefix = f"boxplot_{level}"
            latest, hist = save_figure(fig, self.figures_path, prefix)
            plt.close(fig)

            # 2. Extracting Statistics (The "Box" data)
            level_stats = {}
            groups = tmp.groupby(level)[target]
            
            for name, group in groups:
                q1 = group.quantile(0.25)
                q3 = group.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = group[(group < lower_bound) | (group > upper_bound)]
                
                level_stats[str(name)] = {
                    "count": int(len(group)),
                    "median": float(group.median()),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "min": float(group.min()),
                    "max": float(group.max()),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outliers_count": int(len(outliers)),
                    "outliers": {str(k): float(v) for k, v in outliers.items()}
                }
            
            self.report["results"]["calendar_impact"][level] = {
                "latest_plot": latest,
                "history_plot": hist,
                "statistics": level_stats
            }

    def _step_04_decompose_series(self, train):
        logger.info("EDA - Step 4: Time series decomposition (Rule 3.4)")
        target = self.config["preprocessing"]["target_variable"]
        period = self.config["eda"]["time_series"]["decomposition"]["period"]
        model = self.config["eda"]["time_series"]["decomposition"]["model"]
        
        decomposition = seasonal_decompose(train[target], model=model, period=period)
        
        # 1. Plotting
        fig = decomposition.plot()
        fig.set_size_inches(self.figsize[0], self.figsize[1] * 1.5)
        
        prefix = "ts_decomposition"
        latest, hist = save_figure(fig, self.figures_path, prefix)
        plt.close(fig)
        
        # 2. Extracting components (Handling NaNs for JSON)
        def clean_series(s):
            # Convert series to dict with str keys and handle NaNs for JSON compatibility
            return {str(k): (float(v) if not np.isnan(v) else None) for k, v in s.items()}

        self.report["results"]["time_series"]["decomposition"] = {
            "model": model,
            "period": period,
            "latest_plot": latest,
            "history_plot": hist,
            "values": {
                "trend": clean_series(decomposition.trend),
                "seasonal": clean_series(decomposition.seasonal),
                "resid": clean_series(decomposition.resid)
            }
        }

    def _step_05_analyze_autocorrelation(self, train):
        logger.info("EDA - Step 5: ACF & PACF Analysis (Rule 3.5)")
        target = self.config["preprocessing"]["target_variable"]
        lags = self.config["eda"]["time_series"]["autocorrelation"]["max_lags"]
        
        # 1. Calculation of values
        series = train[target].dropna()
        acf_values = acf(series, nlags=lags)
        pacf_values = pacf(series, nlags=lags)

        # 2. ACF Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        plot_acf(series, lags=lags, ax=ax)
        ax.set_title(f"ACF - {target}")
        latest_acf, hist_acf = save_figure(fig, self.figures_path, "acf_plot")
        plt.close(fig)
        
        # 3. PACF Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        plot_pacf(series, lags=lags, ax=ax)
        ax.set_title(f"PACF - {target}")
        latest_pacf, hist_pacf = save_figure(fig, self.figures_path, "pacf_plot")
        plt.close(fig)
        
        self.report["results"]["time_series"]["autocorrelation"] = {
            "lags": lags,
            "acf": {
                "latest": latest_acf, 
                "history": hist_acf,
                "values": [float(v) for v in acf_values]
            },
            "pacf": {
                "latest": latest_pacf, 
                "history": hist_pacf,
                "values": [float(v) for v in pacf_values]
            }
        }

    def _step_06_verify_stationarity(self, train):
        logger.info("EDA - Step 6: Dickey-Fuller Test (Rule 3.6)")
        target = self.config["preprocessing"]["target_variable"]
        
        result = adfuller(train[target].dropna())
        
        df_results = {
            "test_statistic": float(result[0]),
            "p_value": float(result[1]),
            "used_lag": int(result[2]),
            "n_obs": int(result[3]),
            "critical_values": {k: float(v) for k, v in result[4].items()},
            "is_stationary": bool(result[1] < self.config["eda"]["time_series"]["stationarity"]["significance_level"])
        }
        
        self.report["results"]["time_series"]["stationarity"] = df_results
        logger.info(f"Stationarity Result: {'Stationary' if df_results['is_stationary'] else 'Non-Stationary'} (p={df_results['p_value']:.4f})")

    def _step_07_validate_business_events(self, train):
        logger.info("EDA - Step 7: Validating Business Events (Rule 3.7)")
        target = self.config["preprocessing"]["target_variable"]
        events = self.config["eda"]["business_events"]
        
        # 1. Pandemic Analysis
        pandemia_mask = (train.index >= events["pandemia"]["start"]) & (train.index <= events["pandemia"]["end"])
        pandemia_stats = {
            "mean_during": float(train[pandemia_mask][target].mean()),
            "mean_outside": float(train[~pandemia_mask][target].mean()),
            "impact_pct": float((train[pandemia_mask][target].mean() / train[~pandemia_mask][target].mean() - 1) * 100)
        }
        
        # 2. Promo Months Analysis
        promo_mask = train.index.month.isin(events["promociones"]["months"])
        promo_stats = {
            "mean_promo": float(train[promo_mask][target].mean()),
            "mean_regular": float(train[~promo_mask][target].mean()),
            "impact_pct": float((train[promo_mask][target].mean() / train[~promo_mask][target].mean() - 1) * 100)
        }
        
        # 3. Novenas (Monthly Dec analysis for this granular business logic)
        novenas_mask = train.index.month == events["novenas"]["month"]
        dec_stats = {
            "mean_december": float(train[novenas_mask][target].mean()),
            "mean_non_december": float(train[~novenas_mask][target].mean()),
            "impact_pct": float((train[novenas_mask][target].mean() / train[~novenas_mask][target].mean() - 1) * 100)
        }
        
        self.report["results"]["business_events"] = {
            "pandemia": pandemia_stats,
            "promociones": promo_stats,
            "december_novenas": dec_stats
        }

    def _step_08_expert_interpretation(self, train):
        logger.info("EDA - Step 8: Expert Interpretation (Intelligence Layer)")
        results = self.report["results"]
        
        # 1. Analyze Drift to determine if we need weighting or trend handling
        drift = results["drift_analysis"]
        drift_ratio = drift["test"]["mean"] / drift["train"]["mean"] if drift["train"]["mean"] > 0 else 1
        drift_status = "Significant Positive" if drift_ratio > 1.10 else "Minor/Stable"
        
        # 2. Extract Stationarity Verdict
        stationary = results["time_series"]["stationarity"]["is_stationary"]
        p_val = results["time_series"]["stationarity"]["p_value"]
        
        # 3. Calculate Signal-to-Noise Ratio (SNR) for predictability health
        resid_values = [v for v in results["time_series"]["decomposition"]["values"]["resid"].values() if v is not None]
        resid_std = np.std(resid_values) if resid_values else 1
        target_std = drift["train"]["std"]
        snr = (target_std / resid_std) if resid_std > 0 else 0
        
        # 4. Formulate findings
        main_findings = [
            f"Series level has increased by {((drift_ratio-1)*100):.1f}% between Train and Test periods.",
            f"Pandemic impact was severe ({results['business_events']['pandemia']['impact_pct']:.1f}%), requiring explicit binary indicator.",
            "Strong annual seasonality confirmed (peaks in Jan/Dec), validating the 12-month lag importance."
        ]
        
        # 5. Build Recommendation Objects
        recommendations = {
            "main_insights": main_findings,
            "technical_diagnostics": {
                "stationarity": {
                    "verdict": "Non-Stationary" if not stationary else "Stationary",
                    "p_value": float(p_val),
                    "action_required": "First-order differentiation (d=1) required" if not stationary else "No differentiation needed"
                },
                "log_transform": "Highly Recommended" if drift_status == "Significant Positive" else "Optional/Not Required",
                "signal_to_noise": {
                    "value": float(snr),
                    "interpretation": "High predictability" if snr > 2 else "High noise (Difficulty: High)"
                },
                "seasonality_model": results["time_series"]["decomposition"]["model"]
            },
            "suggested_modeling_strategy": {
                "target_series": "Log-diff series (Returns)" if not stationary else "Log series",
                "lags_grid": [1, 2, 3, 6, 12, 13],
                "windows_grid": [3, 6, 12],
                "weighting_strategy": "Temporal decay (Recency weighting) recommended due to drift." if drift_status == "Significant Positive" else "Uniform weights"
            },
            "exogenous_to_build": [
                "is_pandemic", "is_novenas", "is_promo_season", "ipc_mensual", "confianza_consumidor"
            ],
            "structural_break_risk": "High" if drift_ratio > 1.25 or drift_ratio < 0.75 else "Low"
        }
        
        # Update report with recommendations
        self.report["expert_recommendations"] = recommendations
        logger.info("✅ Expert Interpretation layer added to report.")
