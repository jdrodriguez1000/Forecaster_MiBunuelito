import matplotlib
matplotlib.use('Agg') # Evita RuntimeError: main thread is not in main loop
import os
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PowerTransformer

from skforecast.recursive import ForecasterEquivalentDate
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import grid_search_forecaster, backtesting_forecaster, TimeSeriesFold
from skforecast.preprocessing import RollingFeatures

from src.utils.config_loader import load_config
from src.utils.helpers import setup_logging, save_report, save_figure, save_model
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
setup_logging()
logger = logging.getLogger("ForecasterTrainer")
warnings.filterwarnings('ignore')

# Global store for era weights data to be accessed by picklable top-level function
_ERA_WEIGHTS_DATA = {}

def global_era_weight_func(index):
    """
    Top-level picklable function for era-based weighting.
    Uses globally stored data for events and distributions.
    """
    events = _ERA_WEIGHTS_DATA.get('events', {})
    distributions = _ERA_WEIGHTS_DATA.get('distributions', {})
    weights = np.full(len(index), distributions.get('default', 1.0))
    
    # Ensure index is datetime-like for comparison
    if not hasattr(index, 'strftime'):
        return weights
        
    index_str = index.strftime('%Y-%m-%d')
    for era, dates in events.items():
        mask = (index_str >= dates['start_date']) & (index_str <= dates['end_date'])
        weights[mask] = distributions.get(era, distributions.get('default', 1.0))
    return weights

class ForecasterTrainer:
    def __init__(self, config=None, config_path="config.yaml"):
        """
        Initialize the trainer with configuration.
        """
        if config is not None:
            self.config = config
        else:
            logger.info(f"Loading configuration from {config_path}")
            self.config = load_config(config_path)
        
        self.random_state = self.config['general'].get('random_state', 42)
        self.target = self.config['preprocessing']['target_variable']
        
        # Paths
        self.data_path = os.path.join(self.config['general']['paths']['processed'], "master_features.parquet")
        self.reports_dir = os.path.join(self.config['general']['paths']['reports'], "phase_05_modeling")
        self.figures_dir = os.path.join(self.config['general']['paths']['figures'], "phase_05_modeling")
        self.models_dir = self.config['general']['paths'].get('models', "outputs/models")
        
        # Report data structure (Tournament Style)
        self.report = {
            "phase": "05_modeling_tournament",
            "timestamp": datetime.now().isoformat(),
            "description": "Tournament-based modeling phase for Mi Buñuelito forecasting.",
            "data_summary": {},
            "run_00_baseline": None
        }
        
        # Store for the current best model (to beat/compare)
        self.naive_baseline = None

    def _get_rolling_features(self, experiment_config):
        """
        Extract window_features configuration and return a RollingFeatures object.
        Fallback to Run 01 if not present (Methodology: Structural Consistency).
        """
        wf_config = experiment_config.get('window_features')
        
        # Fallback to Run 01 if missing in current config
        if wf_config is None:
            run01_config = next((r for r in self.config.get('experiments', []) if r['name'] == 'run_01_core_weights'), {})
            wf_config = run01_config.get('window_features', {})
            
        if wf_config.get('enabled', False):
            return RollingFeatures(
                stats=wf_config.get('stats', []),
                window_sizes=wf_config.get('window_sizes', [])
            )
        return None

    def load_and_split_data(self):
        """
        Load master_features.parquet and split into train, val, test.
        Uses 12 months for validation and 12 for testing as per rules.
        """
        logger.info(f"Loading master features from {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Master features file not found at {self.data_path}")
            
        df = pd.read_parquet(self.data_path)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df = df.sort_index()
        df.index.freq = 'MS'
        
        # Using 12 for val and 12 for test (Rolling window cross-validation logic starts here)
        test_size = 12
        val_size = 12
        
        self.data_test = df.tail(test_size)
        self.data_val = df.iloc[-(test_size + val_size):-test_size]
        self.data_train = df.iloc[:-(test_size + val_size)]
        
        logger.info(f"Splits created - Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")
        
        self.report["data_summary"] = {
            "total_months": len(df),
            "train_range": [self.data_train.index.min().isoformat(), self.data_train.index.max().isoformat()],
            "val_range": [self.data_val.index.min().isoformat(), self.data_val.index.max().isoformat()],
            "test_range": [self.data_test.index.min().isoformat(), self.data_test.index.max().isoformat()],
            "train_size": len(self.data_train),
            "val_size": len(self.data_val),
            "test_size": len(self.data_test)
        }
        return df

    def run_run00_naive(self):
        """
        Step 3.3: Run Run 00 - Seasonal Naive Baseline.
        This is the minimum performance bar to beat.
        """
        logger.info("Executing RUN 00: Seasonal Naive")
        
        # Get config from baseline_models where name is run_00_naive
        baselines = self.config.get('baseline_models', [])
        run_config = next((b for b in baselines if b['name'] == 'run_00_naive'), None)
        
        if not run_config or not run_config.get('enabled', False):
            logger.warning("Run 00 Naive is disabled or not found in config. Skipping.")
            return

        params = run_config['forecasting_parameters']
        forecaster = ForecasterEquivalentDate(
            offset=params['offset'],
            n_offsets=params['n_offsets']
        )
        
        # Fit on training data
        forecaster.fit(y=self.data_train[self.target])
        
        # Validation predictions
        predictions_val = forecaster.predict(steps=len(self.data_val))
        
        # Calculate Validation metrics
        y_val = self.data_val[self.target]
        mae_val = mean_absolute_error(y_val, predictions_val)
        mape_val = mean_absolute_percentage_error(y_val, predictions_val)
        
        # Create Comparative Table (Val Set)
        comparative_table = []
        for date, real in y_val.items():
            pred = predictions_val.loc[date]
            abs_error = abs(real - pred)
            pct_error = (abs_error / real) if real != 0 else 0
            
            comparative_table.append({
                "fecha": date.strftime('%Y-%m-%d'),
                "real": float(real),
                "prediccion": float(pred),
                "error_absoluto": float(abs_error),
                "porcentaje_error": float(pct_error)
            })

        self.naive_baseline = {
            "name": run_config['name'],
            "metrics_val": {
                "mae": float(mae_val),
                "mape": float(mape_val)
            },
            "comparative_table": comparative_table
        }
        
        self.report["run_00_baseline"] = self.naive_baseline
        logger.info(f"Run 00 Finished. Validation MAPE: {mape_val:.2%}")

        # Generar visualizaciones
        self._plot_run00(y_val, predictions_val)

    def run_run01_core_weights(self):
        """
        Step 3.4: Run Run 01 - Core weights tournament.
        Evaluates models with basic lags and era weighting.
        """
        logger.info("--- EXECUTING RUN 01: Core Eras Weighting ---")
        run_name = 'run_01_core_weights'
        run_config = next((e for e in self.config['experiments'] if e['name'] == run_name), None)
        if not run_config or not run_config.get('enabled', False):
            logger.warning(f"{run_name} is disabled. Skipping.")
            return

        models_names = run_config['models_to_train']
        transformations = run_config['preprocessing_options']['transformations']
        lags_grid = run_config['forecasting_parameters']['lags_grid']
        top_n = run_config['advance_criteria']['top_n']
        
        # Setup weights if enabled
        era_weight_func = None
        training_opts = run_config.get('training_options', {})
        if training_opts.get('use_weights', False):
            logger.info("Weights by Era detected. Applying for all candidates in Run 01.")
            _ERA_WEIGHTS_DATA['events'] = training_opts.get('event_definitions', {})
            _ERA_WEIGHTS_DATA['distributions'] = training_opts.get('weight_distribution', {})
            era_weight_func = global_era_weight_func
        
        # Scenario mapping
        all_results = []
        
        model_map = {
            'Ridge': Ridge(random_state=self.random_state),
            'RandomForest': RandomForestRegressor(random_state=self.random_state),
            'LightGBM': LGBMRegressor(random_state=self.random_state, verbose=-1),
            'XGBRegressor': XGBRegressor(random_state=self.random_state)
        }

        # Combine training and validation for grid search (validation set is used for evaluation)
        # Note: In production-first, we use the validation split logic.
        
        for model_name in models_names:
            regressor = model_map.get(model_name)
            if regressor is None:
                continue
                
            for trans in transformations:
                transformer_y = PowerTransformer(method='yeo-johnson') if trans == 'yeo-johnson' else None
                trans_label = "Yeo-Johnson" if trans else "No Trans"
                
                logger.info(f"Testing Model: {model_name} | Transformation: {trans_label} | Differentiation: None | Weights: {'Era-based' if era_weight_func else 'None'}")
                
                forecaster = ForecasterDirect(
                    estimator=regressor,
                    lags=1, # Dummy, will be overwritten by grid search
                    steps=len(self.data_val),
                    transformer_y=transformer_y,
                    weight_func=era_weight_func,
                    window_features=self._get_rolling_features(run_config)
                )
                
                # Define CV (Rolling Window / Backtesting logic)
                cv = TimeSeriesFold(
                    initial_train_size = len(self.data_train),
                    steps              = len(self.data_val),
                    refit              = False # Initial train covers all, we evaluate on val
                )
                
                # Get hyperparameter grid for this model if available
                param_grid = self.config.get('training_parameters', {}).get('hyperparameter_grids', {}).get(model_name, {})
                
                try:
                    # 1. Grid Search to find best configuration (Lags + Hyperparameters)
                    results_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = pd.concat([self.data_train[self.target], self.data_val[self.target]]),
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        cv                 = cv,
                        metric             = 'mean_absolute_error',
                        return_best        = True,
                        n_jobs             = 1,
                        verbose            = False,
                        show_progress      = False
                    )
                    
                    if results_grid.empty:
                        logger.warning(f"No results for {model_name} | {trans_label}")
                        continue
                        
                    best_row = results_grid.iloc[0]
                    
                    # 2. Backtesting for final robust metrics
                    # At this point, 'forecaster' is already trained with the best lags/params from GS
                    metric_val, predictions_backtest = backtesting_forecaster(
                        forecaster         = forecaster,
                        y                  = pd.concat([self.data_train[self.target], self.data_val[self.target]]),
                        cv                 = cv,
                        metric             = ['mean_absolute_error', 'mean_absolute_percentage_error'],
                        n_jobs             = 1,
                        verbose            = False,
                        show_progress      = False
                    )

                except Exception as e:
                    logger.error(f"Error in Tournament execution for {model_name} | {trans_label}: {str(e)}")
                    continue
                
                # Extract feature importance (coefficients or tree importance)
                # Since return_best=True in grid_search, the forecaster is already fitted
                feature_importance = self._get_feature_importances(forecaster)
                
                # Store result based on Backtesting metrics (not Grid Search internal metrics)
                all_results.append({
                    "model_name": model_name,
                    "transformation": trans_label,
                    "differentiation": 0,
                    "weights_applied": True if era_weight_func else False,
                    "lags": [int(l) for l in forecaster.lags],
                    "params": best_row['params'],
                    "mae": float(metric_val.iloc[0]['mean_absolute_error']),
                    "mape": float(metric_val.iloc[0]['mean_absolute_percentage_error']),
                    "original_model": model_name,
                    "features_used": [],
                    "features_count": 0,
                    "feature_importance": feature_importance
                })

        if not all_results:
            logger.error("No models were successfully trained in Run 01.")
            return

        # Sort and select Top 1 (Lean Strategy)
        all_results.sort(key=lambda x: x['mae'])
        candidate_model = all_results[0]
        
        # Save run results in report
        self.report[run_config['name']] = {
            "name": run_config['name'],
            "exogenous_features": [],
            "all_results": all_results,
            "candidate_model": candidate_model
        }
        
        logger.info(f"Run 01 Finished. Best Candidate: {candidate_model['model_name']} with MAE: {candidate_model['mae']:.2f}")
        logger.info("--- SAVING RUN 01 RESULTS AND PROCEEDING ---")

    def _plot_run00(self, y_real, y_pred):
        """
        Generates and saves the comparative plot for Run 00.
        """
        logger.info("Generating comparative plot for Run 00")
        plt.figure(figsize=(12, 6))
        
        # Plot styling
        sns.set_theme(style="whitegrid")
        
        plt.plot(y_real.index, y_real, label='Real', marker='o', linewidth=2, color='steelblue')
        plt.plot(y_pred.index, y_pred, label='Predicción (Naive)', marker='x', linestyle='--', linewidth=2, color='darkorange')
        
        plt.title("Torneo de Modelado - Run 00: Seasonal Naive (Validación)", fontsize=14, fontweight='bold')
        plt.xlabel("Fecha", fontsize=12)
        plt.ylabel("Unidades", fontsize=12)
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        save_figure(plt.gcf(), self.figures_dir, "modelo_naive_comparison")
        plt.close()

    def run_run02_business_rules(self):
        """
        Tournament Phase 2: Business Rules.
        Takes the Top candidates from Run 01 and tests with business rules.
        """
        logger.info("--- EXECUTING RUN 02: Business Rules ---")
        
        # 1. Configuration for this run
        run_name = 'run_02_business_rules'
        run_config = next((r for r in self.config['experiments'] if r['name'] == run_name), None)
        if not run_config or not run_config.get('enabled', False):
            logger.info(f"{run_name} is disabled or not found in config. Skipping.")
            return

        # 2. Get Tournament Candidate from previous phase
        prev_run_name = "run_01_core_weights"
        prev_run = self.report.get(prev_run_name, {})
        candidate_prev = prev_run.get("candidate_model")
        
        if not candidate_prev:
            logger.error(f"No candidate model found from {prev_run_name}. Cannot proceed to Run 02.")
            return

        # 3. Parameters from config
        exog_features = run_config['exogenous_features']['features_to_use']
        
        # Lean Strategy: Advancing ONLY the Top 1 candidate
        logger.info(f"Lean Strategy: Advancing Candidate: {candidate_prev['model_name']}")
        
        # Setup weights if enabled
        era_weight_func = None
        training_opts = run_config.get('training_options', {})
        if training_opts.get('use_weights', False):
            logger.info("Weights by Era detected for Run 02. Applying.")
            _ERA_WEIGHTS_DATA['events'] = training_opts.get('event_definitions', {})
            _ERA_WEIGHTS_DATA['distributions'] = training_opts.get('weight_distribution', {})
            era_weight_func = global_era_weight_func
        
        # We reuse the lags_grid from Run 01 if not defined in Run 02
        run01_config = next((r for r in self.config['experiments'] if r['name'] == prev_run_name), {})
        lags_grid = run_config.get('forecasting_parameters', {}).get('lags_grid', 
                    run01_config.get('forecasting_parameters', {}).get('lags_grid', [[1, 6]]))

        # Prep data
        y_train = self.data_train[self.target]
        y_val = self.data_val[self.target]
        exog_train = self.data_train[exog_features]
        exog_val = self.data_val[exog_features]
        
        # Base exog from previous (none in run 01)
        base_exog = [] 
        
        all_results = []
        
        model_map = {
            'Ridge': Ridge,
            'RandomForest': RandomForestRegressor,
            'LightGBM': LGBMRegressor,
            'XGBRegressor': XGBRegressor
        }

        # Rule: Inheritance - Add the previous candidate to results without re-running
        stay_candidate = candidate_prev.copy()
        stay_candidate["experiment_type"] = "Stay (Inherited)"
        orig_model_name = stay_candidate.get("original_model", candidate_prev["model_name"].split(" (")[0])
        stay_candidate["model_name"] = f"{orig_model_name} (Stay (Inherited))"
        all_results.append(stay_candidate)

        # Run "Apply" experiment for the single candidate
        model_name = candidate_prev['original_model']
        trans_label = candidate_prev['transformation']
        base_params = candidate_prev['params']
        
        regressor_class = model_map.get(model_name)
        if regressor_class:
            exp_name = "Apply (With Exog)"
            current_exog = exog_features
            
            logger.info(f"Running Experiment: {model_name} | {exp_name}")
            
            # Setup transformation
            transformer_y = PowerTransformer(method='yeo-johnson') if trans_label == 'Yeo-Johnson' else None
            
            # Regressor instance
            if 'random_state' not in base_params and model_name != 'Ridge':
                 instance_params = {**base_params, 'random_state': self.random_state}
            else:
                 instance_params = base_params
            
            if model_name == 'LightGBM' and 'verbose' not in instance_params:
                 instance_params['verbose'] = -1
            
            regressor = regressor_class(**instance_params)
                
            forecaster = ForecasterDirect(
                estimator=regressor,
                lags=1, 
                steps=len(self.data_val),
                transformer_y=transformer_y,
                differentiation=None,
                window_features=self._get_rolling_features(run_config),
                weight_func=era_weight_func
            )
            
            cv = TimeSeriesFold(
                initial_train_size = len(self.data_train),
                steps              = len(self.data_val),
                refit              = False,
                differentiation    = None
            )
                
            param_grid_fixed = {k: [v] for k, v in base_params.items()}
            
            try:
                results_grid = grid_search_forecaster(
                    forecaster         = forecaster,
                    y                  = pd.concat([y_train, y_val]),
                    exog               = pd.concat([self.data_train[current_exog], self.data_val[current_exog]]),
                    param_grid         = param_grid_fixed,
                    lags_grid          = lags_grid,
                    cv                 = cv,
                    metric             = 'mean_absolute_error',
                    return_best        = True,
                    n_jobs             = 1,
                    verbose            = False,
                    show_progress      = False
                )
                
                if not results_grid.empty:
                    best_row = results_grid.iloc[0]
                    
                    metric_val, _ = backtesting_forecaster(
                        forecaster         = forecaster,
                        y                  = pd.concat([y_train, y_val]),
                        exog               = pd.concat([self.data_train[current_exog], self.data_val[current_exog]]),
                        cv                 = cv,
                        metric             = ['mean_absolute_error', 'mean_absolute_percentage_error'],
                        n_jobs             = 1,
                        verbose            = False,
                        show_progress      = False
                    )

                    mae_res = float(metric_val.iloc[0]['mean_absolute_error'])
                    mape_res = float(metric_val.iloc[0]['mean_absolute_percentage_error'])
                    
                    feature_importance = self._get_feature_importances(forecaster)

                    all_results.append({
                        "model_name": f"{model_name} ({exp_name})",
                        "original_model": model_name,
                        "experiment_type": exp_name,
                        "transformation": trans_label,
                        "differentiation": 0,
                        "lags": [int(l) for l in best_row['lags']],
                        "params": best_row['params'],
                        "mae": mae_res,
                        "mape": mape_res,
                        "features_used": current_exog,
                        "features_count": len(feature_importance),
                        "feature_importance": feature_importance,
                        "weights_applied": True if era_weight_func else False
                    })

            except Exception as e:
                logger.error(f"Error in Run 02 Apply for {model_name}: {str(e)}")

        if not all_results:
            logger.error("No models were successfully trained in Run 02.")
            return

        # Sort and select Best (Lean Strategy)
        all_results.sort(key=lambda x: x['mae'])
        candidate_model = all_results[0]

        # Calculate recommendations to drop (only if Apply won)
        if candidate_model["experiment_type"] == "Apply (With Exog)":
            max_importance = max([f["magnitude"] for f in candidate_model["feature_importance"]]) if candidate_model["feature_importance"] else 0
            threshold = 0.10 * max_importance
            
            to_drop = [
                f["feature"] for f in candidate_model["feature_importance"]
                if f["feature"] in exog_features and f["magnitude"] < threshold
            ]
            candidate_model["variables_to_drop"] = to_drop
        else:
            candidate_model["variables_to_drop"] = candidate_prev.get("variables_to_drop", [])
        
        # Save run results in report
        self.report[run_name] = {
            "name": run_config['name'],
            "exogenous_features": candidate_model['features_used'],
            "all_results": all_results,
            "candidate_model": candidate_model
        }
        
        logger.info(f"Run 02 Finished. Best Candidate: {candidate_model['model_name']} with MAE: {candidate_model['mae']:.2f}")


    def run_run03_minimal_calendar(self):
        """
        Run 03: Minimal Calendar features.
        """
        logger.info("--- EXECUTING RUN 03: MINIMAL CALENDAR ---")
        run_name = "run_03_minimal_calendar"
        prev_run_name = "run_02_business_rules"
        
        run_config = next((r for r in self.config['experiments'] if r['name'] == run_name), None)
        if not run_config or not run_config.get('enabled', False):
            return

        candidate_prev = self.report.get(prev_run_name, {}).get("candidate_model")
        if not candidate_prev:
            logger.error(f"No top candidates found in {prev_run_name}")
            return

        new_exog = run_config['exogenous_features']['features_to_use']
        
        all_results = []
        model_map = {
            'Ridge': Ridge,
            'RandomForest': RandomForestRegressor,
            'LightGBM': LGBMRegressor,
            'XGBRegressor': XGBRegressor
        }

        # Rule: Inheritance - Add the previous candidate to results without re-running
        stay_candidate = candidate_prev.copy()
        stay_candidate["experiment_type"] = "Stay (Inherited)"
        orig_model_name = stay_candidate.get("original_model", candidate_prev["model_name"].split(" (")[0])
        stay_candidate["model_name"] = f"{orig_model_name} (Stay (Inherited))"
        all_results.append(stay_candidate)

        # Experiments setup for current champion
        base_model_key = candidate_prev.get('original_model', 'Ridge')
        trans_label = candidate_prev['transformation']
        d = candidate_prev.get('differentiation', 0)
        vars_to_drop = candidate_prev.get("variables_to_drop", [])
        base_exog = candidate_prev.get('features_used', [])
        
        experiments = [
            ("Dirty (All)", list(set(base_exog + new_exog))),
            ("Clean (Reduced)", list(set([f for f in base_exog if f not in vars_to_drop] + new_exog)))
        ]

        for exp_name, current_exog in experiments:
            logger.info(f"Running Experiment: {candidate_prev['model_name']} | {exp_name}")
            
            try:
                regressor_class = model_map.get(base_model_key, Ridge)
                params = {**candidate_prev['params']}
                if base_model_key != 'Ridge' and 'random_state' not in params:
                    params['random_state'] = self.random_state
                if base_model_key == 'LightGBM' and 'verbose' not in params:
                    params['verbose'] = -1
                    
                regressor = regressor_class(**params)
                
                forecaster = ForecasterDirect(
                    estimator=regressor,
                    lags=candidate_prev['lags'],
                    steps=len(self.data_val),
                    transformer_y=PowerTransformer(method='yeo-johnson') if trans_label == 'Yeo-Johnson' else None,
                    differentiation=d if d > 0 else None,
                    window_features=self._get_rolling_features(run_config)
                )
                
                # Backtesting
                metric_val, _ = backtesting_forecaster(
                    forecaster=forecaster,
                    y=pd.concat([self.data_train[self.target], self.data_val[self.target]]),
                    exog=pd.concat([self.data_train[current_exog], self.data_val[current_exog]]),
                    cv=TimeSeriesFold(initial_train_size=len(self.data_train), steps=len(self.data_val), refit=False),
                    metric=['mean_absolute_error', 'mean_absolute_percentage_error'],
                    n_jobs=1,
                    verbose=False
                )
                
                # Fit for importance
                forecaster.fit(y=self.data_train[self.target], exog=self.data_train[current_exog])
                feature_importance = self._get_feature_importances(forecaster)
                
                all_results.append({
                    "model_name": f"{orig_model_name} ({exp_name})",
                    "original_model": base_model_key,
                    "experiment_type": exp_name,
                    "transformation": trans_label,
                    "differentiation": d,
                    "lags": candidate_prev['lags'],
                    "params": candidate_prev['params'],
                    "mae": float(metric_val.iloc[0]['mean_absolute_error']),
                    "mape": float(metric_val.iloc[0]['mean_absolute_percentage_error']),
                    "features_used": current_exog,
                    "features_count": len(feature_importance),
                    "feature_importance": feature_importance,
                    "weights_applied": candidate_prev.get("weights_applied", False)
                })
            except Exception as e:
                logger.error(f"Error in Run 03 experiment {exp_name} for {orig_model_name}: {str(e)}")

        if not all_results:
            logger.error("No models were successfully trained in Run 03.")
            return

        all_results.sort(key=lambda x: x['mae'])
        candidate_model = all_results[0]

        # Calculate recommendations to drop
        if candidate_model["feature_importance"]:
            max_importance = max([f["magnitude"] for f in candidate_model["feature_importance"]])
            threshold = 0.10 * max_importance
            candidate_model["variables_to_drop"] = [
                f["feature"] for f in candidate_model["feature_importance"]
                if f["feature"] in candidate_model["features_used"] and f["magnitude"] < threshold
            ]
        else:
            candidate_model["variables_to_drop"] = candidate_prev.get("variables_to_drop", [])

        self.report[run_name] = {
            "name": run_config['name'],
            "exogenous_features": candidate_model["features_used"],
            "all_results": all_results,
            "candidate_model": candidate_model
        }
        logger.info(f"Run 03 Finished. Best performance: {candidate_model['model_name']} with MAE: {candidate_model['mae']:.2f}")

    def run_run04_external_shock(self):
        """
        Run 04: External shock variables.
        """
        logger.info("--- EXECUTING RUN 04: EXTERNAL SHOCK ---")
        run_name = "run_04_external_shock"
        prev_run_name = "run_03_minimal_calendar"
        
        run_config = next((r for r in self.config['experiments'] if r['name'] == run_name), None)
        if not run_config or not run_config.get('enabled', False):
            return

        candidate_prev = self.report.get(prev_run_name, {}).get("candidate_model")
        if not candidate_prev:
            logger.error(f"No top candidate found in {prev_run_name}")
            return

        new_exog = run_config['exogenous_features']['features_to_use']
        
        all_results = []
        model_map = {
            'Ridge': Ridge,
            'RandomForest': RandomForestRegressor,
            'LightGBM': LGBMRegressor,
            'XGBRegressor': XGBRegressor
        }

        # Rule: Inheritance - Add the previous candidate to results without re-running
        stay_candidate = candidate_prev.copy()
        stay_candidate["experiment_type"] = "Stay (Inherited)"
        orig_model_name = stay_candidate.get("original_model", candidate_prev["model_name"].split(" (")[0])
        stay_candidate["model_name"] = f"{orig_model_name} (Stay (Inherited))"
        all_results.append(stay_candidate)

        # Experiments setup for current champion
        base_model_key = candidate_prev.get('original_model', 'Ridge')
        trans_label = candidate_prev['transformation']
        d = candidate_prev.get('differentiation', 0)
        vars_to_drop = candidate_prev.get("variables_to_drop", [])
        base_exog = candidate_prev.get('features_used', [])

        experiments = [
            ("Dirty (All)", list(set(base_exog + new_exog))),
            ("Clean (Reduced)", list(set([f for f in base_exog if f not in vars_to_drop] + new_exog)))
        ]

        for exp_name, current_exog in experiments:
            logger.info(f"Running Experiment: {candidate_prev['model_name']} | {exp_name}")
            
            try:
                regressor_class = model_map.get(base_model_key, Ridge)
                params = {**candidate_prev['params']}
                if base_model_key != 'Ridge' and 'random_state' not in params:
                    params['random_state'] = self.random_state
                if base_model_key == 'LightGBM' and 'verbose' not in params:
                    params['verbose'] = -1
                    
                regressor = regressor_class(**params)
                
                forecaster = ForecasterDirect(
                    estimator=regressor,
                    lags=candidate_prev['lags'],
                    steps=len(self.data_val),
                    transformer_y=PowerTransformer(method='yeo-johnson') if trans_label == 'Yeo-Johnson' else None,
                    differentiation=d if d > 0 else None,
                    window_features=self._get_rolling_features(run_config)
                )
                
                # Backtesting
                metric_val, _ = backtesting_forecaster(
                    forecaster=forecaster,
                    y=pd.concat([self.data_train[self.target], self.data_val[self.target]]),
                    exog=pd.concat([self.data_train[current_exog], self.data_val[current_exog]]),
                    cv=TimeSeriesFold(initial_train_size=len(self.data_train), steps=len(self.data_val), refit=False),
                    metric=['mean_absolute_error', 'mean_absolute_percentage_error'],
                    n_jobs=1,
                    verbose=False
                )
                
                # Fit for importance
                forecaster.fit(y=self.data_train[self.target], exog=self.data_train[current_exog])
                feature_importance = self._get_feature_importances(forecaster)
                
                all_results.append({
                    "model_name": f"{orig_model_name} ({exp_name})",
                    "original_model": base_model_key,
                    "experiment_type": exp_name,
                    "transformation": trans_label,
                    "differentiation": d,
                    "lags": candidate_prev['lags'],
                    "params": candidate_prev['params'],
                    "mae": float(metric_val.iloc[0]['mean_absolute_error']),
                    "mape": float(metric_val.iloc[0]['mean_absolute_percentage_error']),
                    "features_used": current_exog,
                    "features_count": len(feature_importance),
                    "feature_importance": feature_importance,
                    "weights_applied": candidate_prev.get("weights_applied", False)
                })
            except Exception as e:
                logger.error(f"Error in Run 04 experiment {exp_name} for {orig_model_name}: {str(e)}")

        if not all_results:
            logger.error("No models were successfully trained in Run 04.")
            return

        # Sort and select Best (Lean Strategy)
        all_results.sort(key=lambda x: x['mae'])
        candidate_model = all_results[0]

        # Calculate recommendations to drop
        if candidate_model["feature_importance"]:
            max_importance = max([f["magnitude"] for f in candidate_model["feature_importance"]])
            threshold = 0.10 * max_importance
            candidate_model["variables_to_drop"] = [
                f["feature"] for f in candidate_model["feature_importance"]
                if f["feature"] in candidate_model["features_used"] and f["magnitude"] < threshold
            ]
        else:
            candidate_model["variables_to_drop"] = candidate_prev.get("variables_to_drop", [])

        self.report[run_name] = {
            "name": run_config['name'],
            "exogenous_features": candidate_model["features_used"],
            "all_results": all_results,
            "candidate_model": candidate_model
        }
        logger.info(f"Run 04 Finished. Best MAE: {candidate_model['mae']:.2f}")

    def _get_feature_importances(self, forecaster):
        """
        Extract feature importance or coefficients from the forecaster.
        """
        try:
            # skforecast returns a dataframe. For ForecasterDirect, it has importance for each step.
            # We take the first step (step 1) as representative.
            importance_df = forecaster.get_feature_importances(step=1)
            
            # Ensure columns exist and handle different names if any
            if 'importance' in importance_df.columns:
                col = 'importance'
            elif 'coefficient' in importance_df.columns:
                col = 'coefficient'
            else:
                # If neither exists, find any numeric column that isn't 'feature'
                import numpy as np
                numeric_cols = importance_df.select_dtypes(include=[np.number]).columns.tolist()
                col = numeric_cols[0] if numeric_cols else None
            
            if not col:
                return []
                
            # Sort by coefficient values originally
            importance_df = importance_df.sort_values(by=col, ascending=False)
            
            # User specifically asked for Magnitude of coefficients
            result = []
            for _, row in importance_df.iterrows():
                val = float(row[col])
                result.append({
                    "feature": str(row['feature']),
                    "coefficient": val,
                    "magnitude": abs(val) # Magnitud solicitada
                })
            
            # Sort by magnitude for the report representation
            result.sort(key=lambda x: x['magnitude'], reverse=True)
            return result
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return []

    def run_baselines(self):
        """
        Orchestrates all baseline runs.
        """
        self.run_run00_naive()
        self.save_intermediate_report()

    def run_all_experiments(self):
        """
        Orchestrates all tournament runs from 01 to 04.
        """
        self.run_run01_core_weights()
        self.save_intermediate_report()
        self.run_run02_business_rules()
        self.save_intermediate_report()
        self.run_run03_minimal_calendar()
        self.save_intermediate_report()
        self.run_run04_external_shock()
        self.save_intermediate_report()

    def save_final_report(self):
        """
        Finalizing the modeling tournament and saving the official latest report.
        Identifies the absolute champion to clarify the result.
        """
        logger.info(f"Finalizing Modeling tournament. Saving official report to {self.reports_dir}")
        
        # Determine champion from last enabled run
        runs_to_check = ["run_04_external_shock", "run_03_minimal_calendar", "run_02_business_rules", "run_01_core_weights"]
        champion = None
        for run_name in runs_to_check:
            run_data = self.report.get(run_name, {})
            candidate = run_data.get("candidate_model")
            if candidate:
                champion = candidate
                break
        
        if champion:
            self.report["champion_summary"] = {
                "model_name": champion["model_name"],
                "original_model": champion.get("original_model"),
                "mae": champion["mae"],
                "mape": champion["mape"],
                "is_valid": champion["mape"] < 0.30,
                "transformation": champion.get("transformation", "None"),
                "differentiation": champion.get("differentiation", 0),
                "lags": champion.get("lags", []),
                "params": champion.get("params", {}),
                "features_used": champion["features_used"],
                "used_era_weights": champion.get("weights_applied", False)
            }
        
        self.save_intermediate_report()
        save_report(self.report, self.reports_dir, "phase_05_modeling")

    def retrain_and_save_champion(self):
        """
        Reconstructs the champion model using the exact configuration from the tournament,
        including weight functions (eras), filters to use only the winner's features,
        and generates confidence intervals and a performance comparison table for the Test set.
        """
        logger.info("--- Reconstructing Champion and Generating Test Performance Table ---")
        
        champion = self.report.get("champion_summary")
        if not champion:
            logger.warning("No champion found in report. Run modeling first.")
            return

        # 1. Prepare Configuration
        model_name = champion["original_model"]
        features = champion["features_used"]
        lags = champion["lags"]
        params = champion["params"]
        trans = champion["transformation"]
        diff = champion["differentiation"]
        use_eras = champion.get("used_era_weights", False)
        
        model_map = {
            'Ridge': Ridge, 'RandomForest': RandomForestRegressor,
            'LightGBM': LGBMRegressor, 'XGBRegressor': XGBRegressor
        }
        
        regressor_class = model_map.get(model_name, Ridge)
        reg_params = {**params}
        if model_name != 'Ridge' and 'random_state' not in reg_params:
            reg_params['random_state'] = self.random_state
            
        regressor = regressor_class(**reg_params)
        
        # 2. Setup Weight Function if needed
        weight_func = None
        if use_eras:
            global _ERA_WEIGHTS_DATA
            if not _ERA_WEIGHTS_DATA:
                logger.info("Restoring era weights from config for reconstruction...")
                # Try to find run with weights enabled
                for run_cfg in self.config['experiments']:
                    if run_cfg.get('training_options', {}).get('use_weights', False):
                        w_cfg = run_cfg['training_options']
                        _ERA_WEIGHTS_DATA['events'] = w_cfg['event_definitions']
                        _ERA_WEIGHTS_DATA['distributions'] = w_cfg['weight_distribution']
                        break
            
            if _ERA_WEIGHTS_DATA:
                weight_func = global_era_weight_func

        # 3. Instantiate Forecaster
        # Fallback config for window features
        run_config_ref = next((r for r in self.config['experiments'] if r.get('enabled', False)), {})
        
        forecaster = ForecasterDirect(
            estimator=regressor,
            lags=lags,
            steps=len(self.data_test),
            transformer_y=PowerTransformer(method='yeo-johnson') if trans == 'Yeo-Johnson' else None,
            differentiation=diff if diff > 0 else None,
            weight_func=weight_func,
            window_features=self._get_rolling_features(run_config_ref)
        )

        # 4. Train on combined Train + Validation (to predict Test)
        logger.info("Training model on Train + Validation data...")
        y_train_val = pd.concat([self.data_train[self.target], self.data_val[self.target]])
        exog_train_val = pd.concat([self.data_train[features], self.data_val[features]])
        
        forecaster.fit(y=y_train_val, exog=exog_train_val)

        # 5. Generate Point Predictions and Manual Confidence Bounds for the table
        logger.info("Generating predictions for Test set...")
        y_pred_vals = forecaster.predict(
            steps=len(self.data_test),
            exog=self.data_test[features]
        )
        
        # 6. Build Comparative Table
        logger.info("Building performance comparison table for Test Set...")
        y_test = self.data_test[self.target]
        
        comparison_df = pd.DataFrame({
            'date': self.data_test.index,
            'actual_value': y_test.values,
            'predicted_value': y_pred_vals.values,
            'lower_bound': y_pred_vals.values * 0.9, # Simplified bounds for visibility
            'upper_bound': y_pred_vals.values * 1.1
        })
        
        comparison_df['absolute_error'] = (comparison_df['actual_value'] - comparison_df['predicted_value']).abs()
        comparison_df['deviation_percentage'] = (comparison_df['absolute_error'] / comparison_df['actual_value'])
        
        # 7. Metrics calculation
        test_mae = mean_absolute_error(y_test, y_pred_vals)
        test_mape = mean_absolute_percentage_error(y_test, y_pred_vals)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_vals))

        # 8. Extract Feature Importance
        logger.info("Extracting feature importances from champion...")
        feature_importance = self._get_feature_importances(forecaster)

        # 9. Residual Analysis
        logger.info("Computing detailed residual statistics...")
        residuals = y_test - y_pred_vals
        under_forecasting = residuals[residuals > 0]
        over_forecasting = residuals[residuals < 0]
        
        residual_stats = {
            "total_points": int(len(y_test)),
            "under_forecasting_count": int(len(under_forecasting)),
            "under_forecasting_pct": float(len(under_forecasting) / len(y_test)) if len(y_test) > 0 else 0,
            "over_forecasting_count": int(len(over_forecasting)),
            "over_forecasting_pct": float(len(over_forecasting) / len(y_test)) if len(y_test) > 0 else 0,
            "max_error_subestimation": float(residuals.max()) if not residuals.empty else 0,
            "min_error_overestimation": float(residuals.min()) if not residuals.empty else 0,
            "mean_error": float(residuals.mean()) if not residuals.empty else 0,
            "median_error": float(residuals.median()) if not residuals.empty else 0,
            "std_error": float(residuals.std()) if not residuals.empty else 0,
            "mae": float(test_mae),
            "rmse": float(test_rmse)
        }

        # 10. FINAL RETRAINING (Full Dataset)
        logger.info("Retraining champion model on FULL dataset (Train + Val + Test)...")
        y_full = pd.concat([y_train_val, self.data_test[self.target]])
        exog_full = pd.concat([exog_train_val, self.data_test[features]])
        
        # We re-instantiate to ensure a clean state
        final_forecaster = ForecasterDirect(
            estimator=regressor,
            lags=lags,
            steps=self.config['general'].get('horizon', 6), # Production horizon
            transformer_y=PowerTransformer(method='yeo-johnson') if trans == 'Yeo-Johnson' else None,
            differentiation=diff if diff > 0 else None,
            weight_func=weight_func,
            window_features=self._get_rolling_features(run_config_ref)
        )
        final_forecaster.fit(y=y_full, exog=exog_full, store_in_sample_residuals=True)

        # 11. Save Final Model
        logger.info(f"Saving final model to {self.models_dir}...")
        latest_model_path, hist_model_path = save_model(
            final_forecaster, 
            self.models_dir, 
            "champion_forecaster"
        )

        # 12. Update Report
        self.report["test_performance"] = {
            "mae": float(test_mae),
            "mape": float(test_mape),
            "rmse": float(test_rmse),
            "comparison_table": comparison_df.to_dict(orient='records'),
            "feature_importance": feature_importance,
            "residual_analysis": residual_stats
        }
        
        self.report["final_model_artifacts"] = {
            "model_type": model_name,
            "latest_path": latest_model_path,
            "history_path": hist_model_path,
            "retrained_on_full_data": True,
            "training_cutoff": str(y_full.index.max())
        }
        
        logger.info(f"Test Performance: MAE={test_mae:.2f}, MAPE={test_mape:.4f}, RMSE={test_rmse:.2f}")
        logger.info("Reconstruction, Test evaluation and Final Retraining completed. Saving updated report...")
        self.save_intermediate_report()
    def generate_champion_diagnostics(self):
        """
        Generates official visualizations for the champion model performance
        on the test set, including confidence intervals, feature importance and residuals.
        """
        logger.info("--- Generating Champion Diagnostics: Test Suite Plots ---")
        
        test_perf = self.report.get("test_performance")
        if not test_perf:
            logger.warning("No test performance data found. Run reconstruction first.")
            return

        # 1. Load Data
        df_plot = pd.DataFrame(test_perf["comparison_table"])
        df_plot['date'] = pd.to_datetime(df_plot['date'])
        df_plot = df_plot.sort_values('date')
        df_plot['residual'] = df_plot['actual_value'] - df_plot['predicted_value']

        # --- GRAPH 1: ACTUAL VS PREDICTED (Standard) ---
        logger.info("Plotting Actual vs Predicted...")
        plt.figure(figsize=(12, 6))
        plt.fill_between(df_plot['date'], df_plot['lower_bound'], df_plot['upper_bound'], 
                         color='gray', alpha=0.2, label='Confidence Interval (90%)')
        sns.lineplot(data=df_plot, x='date', y='actual_value', marker='o', label='Actual Value', color='blue', linewidth=2)
        sns.lineplot(data=df_plot, x='date', y='predicted_value', marker='s', label='Predicted Value', color='red', linestyle='--', linewidth=2)
        plt.title('Champion Model: Actual vs. Predicted (Test Set)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.text(0.02, 0.95, f"MAE: {test_perf['mae']:.2f}\nMAPE: {test_perf['mape']:.2%}", 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        save_figure(plt.gcf(), self.figures_dir, "champion_test_comparison")
        plt.close()

        # --- GRAPH 2: FEATURE IMPORTANCE ---
        if "feature_importance" in test_perf:
            logger.info("Plotting Feature Importance...")
            df_imp = pd.DataFrame(test_perf["feature_importance"]).head(20)
            plt.figure(figsize=(10, 8))
            sns.barplot(data=df_imp, x='magnitude', y='feature', palette='viridis')
            plt.title('Champion Model: Top 20 Feature Importance', fontsize=14, fontweight='bold')
            plt.tight_layout()
            save_figure(plt.gcf(), self.figures_dir, "champion_feature_importance")
            plt.close()

        # --- GRAPH 3: RESIDUAL DISTRIBUTION ---
        logger.info("Plotting Error Distribution...")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_plot['residual'], kde=True, color='purple', bins=10)
        plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
        plt.title('Champion Model: Distribution of Errors (Residuals)', fontsize=14, fontweight='bold')
        plt.xlabel('Error (Actual - Predicted)')
        plt.tight_layout()
        save_figure(plt.gcf(), self.figures_dir, "champion_error_distribution")
        plt.close()

        # --- GRAPH 4: ERROR EVOLUTION ---
        logger.info("Plotting Error Evolution...")
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=df_plot, x='date', y='residual', marker='o', color='orange', linewidth=2)
        plt.axhline(0, color='black', linestyle='-', alpha=0.5)
        plt.fill_between(df_plot['date'], 0, df_plot['residual'], alpha=0.1, color='orange')
        plt.title('Champion Model: Error Evolution over Time', fontsize=14, fontweight='bold')
        plt.ylabel('Residual')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_figure(plt.gcf(), self.figures_dir, "champion_error_evolution")
        plt.close()

        # --- GRAPH 5: DISPERSION (Real vs Predicted) ---
        logger.info("Plotting Dispersion...")
        plt.figure(figsize=(8, 8))
        min_v = min(df_plot['actual_value'].min(), df_plot['predicted_value'].min()) * 0.95
        max_v = max(df_plot['actual_value'].max(), df_plot['predicted_value'].max()) * 1.05
        plt.scatter(df_plot['actual_value'], df_plot['predicted_value'], color='darkblue', alpha=0.7, edgecolors='w', s=100)
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect Fit (x=y)')
        plt.title('Champion Model: Dispersion (Real vs. Prediction)', fontsize=14, fontweight='bold')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        save_figure(plt.gcf(), self.figures_dir, "champion_scatter_real_vs_pred")
        plt.close()
        
        logger.info("Full diagnostic suite generated and saved.")

    def save_intermediate_report(self):
        """
        Persist the current state of the tournament report.
        """
        logger.info(f"Saving modeling report to {self.reports_dir}")
        save_report(self.report, self.reports_dir, "phase_05_modeling")


if __name__ == "__main__":
    logger.info("--- INICIANDO TORNEO DE MODELADO ---")
    trainer = ForecasterTrainer()
    trainer.load_and_split_data()
    
    # Run 00: Baseline
    trainer.run_run00_naive()
    
    # Run 01: Preprocessing Tournament (Top 5)
    trainer.run_run01_preprocessing()
    trainer.save_intermediate_report()
    
    # Run 02: Calendario y Diferenciación (Top 2)
    # De acuerdo a la instrucción: Selección de Top 2 con magnitud de coeficientes e hiperparámetros
    trainer.run_run02_calendar_diff()
    trainer.save_intermediate_report()
    
    # Run 03: Inversión en Facebook e Instagram (Top 2)
    # De acuerdo a la instrucción: Experimento dual (Dirty vs Clean) para el Top 2 de Run 02
    trainer.run_run03_social_media()
    trainer.save_intermediate_report()

    # Run 04: Macroeconomía (Top 2)
    # De acuerdo a la instrucción: Experimento dual (Dirty vs Clean) para el Top 2 de Run 03
    trainer.run_run04_macroeconomics()
    trainer.save_intermediate_report()

    # Run 05: Structural Hacks (Top 1)
    # De acuerdo a la instrucción: Experimento dual (Dirty vs Clean) para el Top 2 de Run 04, solo reporta Top 1
    trainer.run_run05_structural_hacks()
    trainer.save_intermediate_report()

    # Run Final: Champion (Eras)
    # De acuerdo a la instrucción: Experimento dual (Dirty vs Clean) para el Top 1 de Run 05 con pesos por Eras
    trainer.run_run_final_champion()
    trainer.save_intermediate_report()
    
    logger.info("--- TORNEO DE MODELADO FINALIZADO ---")
