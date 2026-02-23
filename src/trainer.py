import matplotlib
matplotlib.use('Agg') # Evita RuntimeError: main thread is not in main loop
import os
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from skforecast.recursive import ForecasterEquivalentDate
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import grid_search_forecaster, backtesting_forecaster, TimeSeriesFold

from src.utils.config_loader import load_config
from src.utils.helpers import setup_logging, save_report, save_figure, save_model
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
setup_logging()
logger = logging.getLogger("ForecasterTrainer")
warnings.filterwarnings('ignore')

class ForecasterTrainer:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the trainer with configuration.
        """
        logger.info(f"Loading configuration from {config_path}")
        self.config = load_config(config_path)
        self.random_state = self.config['general'].get('random_state', 42)
        self.target = self.config['preprocessing']['target_variable']
        
        # Paths
        self.data_path = os.path.join(self.config['general']['paths']['processed'], "master_features.parquet")
        self.reports_dir = os.path.join(self.config['general']['paths']['reports'], "phase_05_modeling")
        
        # Modeling parameters from Phase 5 config
        self.modeling_params = self.config.get('training_parameters', {})
        self.horizon = self.modeling_params.get('forecast_horizon', 6)
        
        # Preprocessing Inherited (to be set after Run 01)
        self.best_differentiation = None
        self.best_transformer_name = None
        
        # Report data structure
        self.report = {
            "phase": "05_modeling_and_training",
            "timestamp": datetime.now().isoformat(),
            "description": "Training and modeling phase for Mi Buñuelito forecasting.",
            "data_summary": {},
            "baselines": [],
            "experiments": [],
            "candidate_models": [],
            "champion_model": None,
            "best_preprocessing": {
                "differentiation": None,
                "transformation": None
            }
        }
        
        # Internal list to track all candidates across baselines and experiments
        self.all_candidates = []

    def _get_regressor(self, name):
        """
        Returns an instance of the requested regressor by name.
        """
        models_map = {
            "Ridge": Ridge(random_state=self.random_state),
            "RandomForest": RandomForestRegressor(random_state=self.random_state),
            "LightGBM": LGBMRegressor(random_state=self.random_state, verbose=-1),
            "XGBRegressor": XGBRegressor(random_state=self.random_state),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=self.random_state),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=self.random_state)
        }
        return models_map.get(name)

    def load_and_split_data(self):
        """
        Step 3.2: Load master_features.parquet and split into train, val, test.
        """
        logger.info(f"Loading master features from {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Master features file not found at {self.data_path}")
            
        df = pd.read_parquet(self.data_path)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df = df.sort_index()
        df.index.freq = 'MS'
        
        logger.info(f"Dataset loaded with {len(df)} months. Range: {df.index.min()} to {df.index.max()}")
        
        # Using 12 for val and 12 for test as agreed
        test_size = 12
        val_size = 12
        
        self.data_test = df.tail(test_size)
        self.data_val = df.iloc[-(test_size + val_size):-test_size]
        self.data_train = df.iloc[:-(test_size + val_size)]
        self.data_train_val_test = df
        
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

    def run_baselines(self):
        """
        Step 3.3: Run seasonal naive baseline.
        """
        logger.info("Running Baselines (Step 3.3)")
        baselines_config = self.config.get('baseline_models', [])
        
        for baseline in baselines_config:
            if not baseline.get('enabled', False):
                continue
                
            name = baseline['name']
            logger.info(f"Executing baseline: {name}")
            
            params = baseline['forecasting_parameters']
            forecaster = ForecasterEquivalentDate(
                offset=params['offset'],
                n_offsets=params['n_offsets']
            )
            
            forecaster.fit(y=self.data_train[self.target])
            
            predictions_val = forecaster.predict(steps=len(self.data_val))
            # VAL METRICS ONLY (Anti-Data Leakage)
            mae_val = mean_absolute_error(self.data_val[self.target], predictions_val)
            mape_val = mean_absolute_percentage_error(self.data_val[self.target], predictions_val)
            
            baseline_result = {
                "name": name,
                "description": baseline['description'],
                "metrics_val": {
                    "mae": float(mae_val),
                    "mape": float(mape_val)
                }
            }
            self.report["baselines"].append(baseline_result)
            
            # Store in candidates list
            self.all_candidates.append({
                "name": name,
                "metrics_val": baseline_result["metrics_val"],
                "forecaster": forecaster
            })
            logger.info(f"Baseline {name} finished. Val MAPE: {mape_val:.2%}")

    def _run_generic_experiment(self, exp_config, is_endogenous=False):
        """
        Generic method to run an experiment based on configuration.
        """
        exp_name = exp_config['name']
        logger.info(f"--- Starting Experiment: {exp_name} ---")
        
        # 1. Setup Preprocessing
        transformer = None
        differentiation = None
        
        if is_endogenous:
            # Run 01 tests multiple scenarios
            scenarios = [
                {"name": "No Trans, No Diff", "transformer": None, "differentiation": None},
                {"name": "Yeo-Johnson, No Diff", "transformer": PowerTransformer(method='yeo-johnson'), "differentiation": None},
                {"name": "No Trans, Diff 1", "transformer": None, "differentiation": 1},
                {"name": "Yeo-Johnson, Diff 1", "transformer": PowerTransformer(method='yeo-johnson'), "differentiation": 1}
            ]
        else:
            # Exogenous runs inherit best from Run 01
            if self.best_transformer_name == 'yeo-johnson':
                transformer = PowerTransformer(method='yeo-johnson')
            differentiation = self.best_differentiation
            scenarios = [{"name": "Inherited Preprocessing", "transformer": transformer, "differentiation": differentiation}]

        # 2. Features to use
        exog_cols = []
        if not is_endogenous:
            exog_cfg = exp_config.get('exogenous_features', {}).get('features_to_use', [])
            if exog_cfg == 'all':
                exog_cols = [c for c in self.data_train.columns if c != self.target]
            else:
                exog_cols = exog_cfg
            logger.info(f"Using {len(exog_cols)} exogenous features.")

        models_to_train = exp_config['models_to_train']
        lags_grid = exp_config['forecasting_parameters']['lags_grid']
        cv_params = self.modeling_params.get('grid_search_cv_params', {})
        
        run_results = []
        
        for model_name in models_to_train:
            regressor = self._get_regressor(model_name)
            if regressor is None: continue
            
            param_grid = self.config['training_parameters']['hyperparameter_grids'].get(model_name, {})
            
            for scenario in scenarios:
                logger.info(f"Model: {model_name} | Scenario: {scenario['name']}")
                
                # Ensure differentiation is int or None
                diff_val = int(scenario['differentiation']) if pd.notnull(scenario['differentiation']) else None

                forecaster = ForecasterDirect(
                    regressor=regressor,
                    lags=1, 
                    steps=self.horizon,
                    transformer_y=scenario['transformer'],
                    differentiation=diff_val
                )
                
                data_train_val = pd.concat([self.data_train, self.data_val])
                
                cv = TimeSeriesFold(
                    initial_train_size=len(self.data_train),
                    steps=self.horizon,
                    refit=cv_params.get('refit', False),
                    fixed_train_size=cv_params.get('fixed_train_size', False),
                    allow_incomplete_fold=cv_params.get('allow_incomplete_folds', True),
                    differentiation=diff_val
                )
                
                # Exog data if available
                exog_train_val = data_train_val[exog_cols] if exog_cols else None
                
                # Grid Search
                gs_results = grid_search_forecaster(
                    forecaster=forecaster,
                    y=data_train_val[self.target],
                    exog=exog_train_val,
                    param_grid=param_grid,
                    lags_grid=lags_grid,
                    cv=cv,
                    metric=self.modeling_params['metric_for_tuning'],
                    return_best=True,
                    verbose=False
                )
                
                best_params = gs_results.iloc[0]['params']
                best_lags = [int(l) for l in gs_results.iloc[0]['lags']]
                
                # Backtesting for Validation metrics
                metrics_val, _ = backtesting_forecaster(
                    forecaster=forecaster,
                    y=data_train_val[self.target],
                    exog=exog_train_val,
                    cv=cv,
                    metric=[self.modeling_params['metric_for_tuning'], 'mean_absolute_percentage_error'],
                    verbose=False
                )
                
                run_results.append({
                    "model": model_name,
                    "scenario": scenario['name'] if is_endogenous else None,
                    "differentiation": scenario['differentiation'],
                    "transformation": "yeo-johnson" if scenario['transformer'] else None,
                    "best_params": best_params,
                    "best_lags": best_lags,
                    "mae_val": float(metrics_val.iloc[0, 0]),
                    "mape_val": float(metrics_val.iloc[0, 1]),
                    "forecaster": forecaster
                })

        # 3. Determine Winner and Final Test
        run_results_df = pd.DataFrame(run_results)
        winner_row = run_results_df.loc[run_results_df['mae_val'].idxmin()]
        
        if is_endogenous:
            self.best_differentiation = int(winner_row['differentiation']) if pd.notnull(winner_row['differentiation']) else None
            self.best_transformer_name = winner_row['transformation']
            self.report["best_preprocessing"] = {
                "differentiation": self.best_differentiation,
                "transformation": self.best_transformer_name
            }

        # 4. Report (Validation Only)

        # 4. Report
        combinations_list = []
        for _, row in run_results_df.iterrows():
            combinations_list.append({
                "model": row['model'],
                "scenario": row['scenario'],
                "differentiation": int(row['differentiation']) if pd.notnull(row['differentiation']) else None,
                "transformation": row['transformation'],
                "best_lags": row['best_lags'],
                "best_params": row['best_params'],
                "mae_val": row['mae_val'],
                "mape_val": row['mape_val']
            })

        experiment_result = {
            "name": exp_name,
            "description": exp_config['description'],
            "winner_model": winner_row['model'],
            "best_params": winner_row['best_params'],
            "best_lags": winner_row['best_lags'],
            "metrics_val": {"mae": float(winner_row['mae_val']), "mape": float(winner_row['mape_val'])},
            "all_combinations": combinations_list
        }
        self.report["experiments"].append(experiment_result)

        # Update Candidates List (Validation only)
        self.all_candidates.append({
            "name": f"{exp_name}_{winner_row['model']}",
            "metrics_val": experiment_result["metrics_val"],
            "best_params": winner_row['best_params'],
            "best_lags": winner_row['best_lags'],
            "preprocessing": {
                "differentiation": self.best_differentiation,
                "transformation": self.best_transformer_name
            },
            "forecaster": winner_row['forecaster'],
            "exog_cols": exog_cols
        })

    def run_all_experiments(self):
        """
        Iterates through all experiments in config.
        """
        for exp_config in self.config.get('experiments', []):
            if not exp_config.get('enabled', False):
                continue
            
            is_endogenous = (exp_config['name'] == 'run01_endogenous')
            self._run_generic_experiment(exp_config, is_endogenous=is_endogenous)

    def save_final_report(self):
        """
        Finalize candidates, select champion, and save the JSON report.
        """
        # 1. Finalize Candidates selection
        if not self.all_candidates:
            logger.warning("No candidates found to report.")
        else:
            # Sort by MAE validation (ascending)
            sorted_candidates = sorted(self.all_candidates, key=lambda x: x['metrics_val']['mae'])
            
            # Champion Logic: Seleccionar el campeón (objeto completo)
            best = sorted_candidates[0]
            if "naive" in best['name'].lower() and len(sorted_candidates) > 1:
                self.champion_obj = sorted_candidates[1]
                logger.info(f"Champion Logic: Baseline is #1, selecting #2 as Champion: {self.champion_obj['name']}")
            else:
                self.champion_obj = best
                logger.info(f"Champion Logic: Selecting #1 as Champion: {self.champion_obj['name']}")
            
            # 1.1 Create a clean Top 3 for JSON (no non-serializable objects and NO metrics_test)
            self.report["candidate_models"] = []
            for cand in sorted_candidates[:3]:
                clean_cand = {k: v for k, v in cand.items() if k not in ['forecaster', 'metrics_test']}
                self.report["candidate_models"].append(clean_cand)
            
            # 1.2 Calculate Test Metrics ONLY for Champion (Post-Selection)
            logger.info(f"Calculating Test metrics for Champion: {self.champion_obj['name']}")
            
            data_train_val = pd.concat([self.data_train, self.data_val])
            exog_cols = self.champion_obj.get('exog_cols', [])
            exog_all = self.data_train_val_test[exog_cols] if exog_cols else None
            
            diff_val = self.champion_obj.get('preprocessing', {}).get('differentiation')
            cv_test = TimeSeriesFold(
                initial_train_size=len(data_train_val),
                steps=self.horizon,
                refit=False,
                allow_incomplete_fold=True,
                differentiation=diff_val
            )
            
            metrics_test, preds_df = backtesting_forecaster(
                forecaster=self.champion_obj['forecaster'],
                y=self.data_train_val_test[self.target],
                exog=exog_all,
                cv=cv_test,
                metric=['mean_absolute_error', 'mean_absolute_percentage_error'],
                interval=[5, 95],
                n_boot=250,
                verbose=False
            )
            
            # 1.3 Create comparative table (Real vs Prediction)
            y_test = self.data_train_val_test[self.target].iloc[len(data_train_val):]
            table_rows = []
            for date, real in y_test.items():
                if date in preds_df.index:
                    pred = preds_df.loc[date, 'pred']
                    lower = preds_df.loc[date, 'lower_bound'] if 'lower_bound' in preds_df.columns else None
                    upper = preds_df.loc[date, 'upper_bound'] if 'upper_bound' in preds_df.columns else None
                    
                    abs_error = abs(real - pred)
                    deviation = (abs_error / real) * 100 if real != 0 else 0
                    
                    row = {
                        "Fecha": date.strftime('%Y-%m-%d'),
                        "Real": float(real),
                        "Prediccion": float(pred),
                        "Error absoluto": float(abs_error),
                        "Desviacion": float(deviation)
                    }
                    if lower is not None:
                        row["Limite Inferior (5%)"] = float(lower)
                        row["Limite Superior (95%)"] = float(upper)
                    
                    table_rows.append(row)

            # 1.4 Create a clean Champion for JSON with metrics_test and comparative table
            self.report["champion_model"] = {k: v for k, v in self.champion_obj.items() if k != 'forecaster'}
            self.report["champion_model"]["metrics_test"] = {
                "mae": float(metrics_test.iloc[0, 0]),
                "mape": float(metrics_test.iloc[0, 1])
            }
            self.report["champion_model"]["test_comparative_table"] = table_rows

        # 2. Save Report
        logger.info(f"Saving final modeling report to {self.reports_dir}")
        save_report(self.report, self.reports_dir, "phase_05_modeling")

    def generate_champion_diagnostics(self):
        """
        Step 3.8 & 3.9: Visual comparison and advanced diagnostics.
        """
        if not hasattr(self, 'champion_obj'):
            logger.error("No champion model available for diagnostics.")
            return
            
        champion = self.champion_obj
        forecaster = champion['forecaster']
        exog_cols = champion.get('exog_cols', [])
        name = champion['name']
        
        logger.info(f"--- Generating Diagnostics for Champion: {name} ---")
        
        # Paths for figures
        fig_dir = self.config['general']['paths'].get('figures', 'outputs/figures')
        fig_phase_dir = os.path.join(fig_dir, "phase_05_modeling")
        os.makedirs(fig_phase_dir, exist_ok=True)
        
        # 3.8 Visual Comparison (Test Set)
        data_train_val = pd.concat([self.data_train, self.data_val])
        exog_train_val_test = self.data_train_val_test[exog_cols] if exog_cols else None
        
        # Use backtesting to get predictions over the whole test set (it handles the horizon limits)
        diff_val = champion.get('preprocessing', {}).get('differentiation')
        cv_test = TimeSeriesFold(
            initial_train_size=len(data_train_val),
            steps=self.horizon,
            refit=False,
            allow_incomplete_fold=True,
            differentiation=diff_val
        )
        
        try:
            _, preds_df = backtesting_forecaster(
                forecaster=forecaster,
                y=self.data_train_val_test[self.target],
                exog=exog_train_val_test,
                cv=cv_test,
                metric='mean_absolute_error',
                interval=[5, 95],
                n_boot=250,
                verbose=False
            )
            # predictions in skforecast backtesting have a 'pred' column
            preds = preds_df['pred']
            lower_bound = preds_df['lower_bound'] if 'lower_bound' in preds_df.columns else None
            upper_bound = preds_df['upper_bound'] if 'upper_bound' in preds_df.columns else None
        except Exception as e:
            logger.warning(f"Backtesting failed for visual comparison: {e}. Attempting direct predict.")
            # Fallback for simple models like Naive if they don't support backtesting or have different signatures
            exog_test = self.data_test[exog_cols] if exog_cols else None
            try:
                # Direct predict often doesn't give intervals unless specifically requested,
                # but for diagnostics we prefer the backtesting result.
                preds = forecaster.predict(steps=len(self.data_test), exog=exog_test)
                lower_bound, upper_bound = None, None
            except:
                logger.error("Could not generate predictions for visual comparison.")
                return
        
        # Create Comparative Table for Report
        y_test = self.data_test[self.target]
        table_rows = []
        for date, real in y_test.items():
            if date in preds.index:
                pred = preds.loc[date]
                abs_error = abs(float(real) - float(pred))
                deviation = (abs_error / float(real)) * 100 if float(real) != 0 else 0
                
                row = {
                    "Fecha": date.strftime('%Y-%m-%d'),
                    "Real": float(real),
                    "Prediccion": float(pred),
                    "Error absoluto": float(abs_error),
                    "Desviacion": float(deviation)
                }
                
                if lower_bound is not None and date in lower_bound.index:
                    row["Limite Inferior (5%)"] = float(lower_bound.loc[date])
                    row["Limite Superior (95%)"] = float(upper_bound.loc[date])
                
                table_rows.append(row)
        
        if "champion_model" in self.report:
            self.report["champion_model"]["test_comparative_table"] = table_rows
            logger.info("Updating final JSON report with comparative table.")
            save_report(self.report, self.reports_dir, "phase_05_modeling")

        # 3.8.1 Plot Visual Comparison (Original Style)
        plt.figure(figsize=(12, 6))
        plt.plot(self.data_test.index, self.data_test[self.target], label='Real', marker='o', color='steelblue')
        plt.plot(preds.index, preds, label='Predicciones', marker='x', linestyle='--', color='darkorange')
        plt.title(f"Comparativa Real vs Predicciones - {name}", fontsize=14)
        plt.xlabel("Fecha")
        plt.ylabel("Unidades")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_figure(plt.gcf(), fig_phase_dir, "champion_real_vs_pred")
        plt.close()

        # 3.8.2 Plot Confidence Intervals (Dedicated Plot)
        if lower_bound is not None and upper_bound is not None:
            plt.figure(figsize=(12, 6))
            
            # Predictions
            plt.plot(preds.index, preds, 
                     label='Predicción Champion', marker='x', 
                     linestyle='--', linewidth=2, color='#e67e22', zorder=4)
            
            # Confidence Intervals
            plt.fill_between(
                preds.index, 
                lower_bound, 
                upper_bound, 
                color='#f39c12', 
                alpha=0.2, 
                label='Zona de Incertidumbre (95%)',
                zorder=2
            )
            # Boundary lines
            plt.plot(preds.index, lower_bound, color='#f39c12', linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)
            plt.plot(preds.index, upper_bound, color='#f39c12', linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)
            
            plt.title(f"Intervalos de Confianza del Pronóstico (95%)\nModelo: {name}", 
                      fontsize=15, fontweight='bold', pad=15)
            plt.xlabel("Fecha")
            plt.ylabel("Unidades")
            plt.legend(loc='upper right', frameon=True, shadow=True)
            plt.grid(True, linestyle='--', alpha=0.4)
            
            plt.tight_layout()
            save_figure(plt.gcf(), fig_phase_dir, "champion_confidence_intervals")
            plt.close()

        # 3.9 Advanced Diagnostics
        # A. Feature Importance / Magnitud de Coeficientes
        try:
            # ForecasterDirect requires the 'step' argument
            importance_df = forecaster.get_feature_importances(step=1)
            if importance_df is not None:
                # Sort and get Top 25
                importance_df = importance_df.sort_values(by='importance', ascending=False)
                top_25 = importance_df.head(25)
                
                # Plot
                plt.figure(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=top_25, palette='viridis')
                plt.title(f"Importancia de Variables (Top 25) - {name}")
                plt.xlabel("Importancia")
                plt.ylabel("Variable")
                plt.tight_layout()
                save_figure(plt.gcf(), fig_phase_dir, "champion_feature_importance")
                plt.close()
                
                # Table for JSON
                coeff_magnitude = []
                for _, row in top_25.iterrows():
                    coeff_magnitude.append({
                        "Variable": row['feature'],
                        "Magnitud": float(row['importance'])
                    })
                
                if "champion_model" in self.report:
                    self.report["champion_model"]["magnitud_coeficientes"] = coeff_magnitude
                    logger.info("Updating final JSON report with feature importance table.")
                    save_report(self.report, self.reports_dir, "phase_05_modeling")
            else:
                logger.warning("Forecaster returned None for feature importance.")
        except Exception as e:
            logger.warning(f"Could not generate feature importance for this model type: {e}")

        # B. Residual Analysis (Advanced)
        y_test = self.data_test[self.target]
        residuals = y_test - preds
        
        # Calculate Detailed Statistics
        n_points = len(residuals)
        under_f = residuals[residuals > 0]
        over_f = residuals[residuals < 0]
        
        mae_val = mean_absolute_error(y_test, preds)
        rmse_val = np.sqrt(mean_squared_error(y_test, preds))
        
        res_stats = {
            "total_puntos": int(n_points),
            "under_forecasting": {
                "puntos": int(len(under_f)),
                "porcentaje": float(len(under_f) / n_points * 100)
            },
            "over_forecasting": {
                "puntos": int(len(over_f)),
                "porcentaje": float(len(over_f) / n_points * 100)
            },
            "error_max_subestimacion": float(residuals.max()),
            "error_min_sobreestimacion": float(residuals.min()),
            "media_error": float(residuals.mean()),
            "mediana_error": float(residuals.median()),
            "desviacion_estandar_error": float(residuals.std()),
            "mae_periodo": float(mae_val),
            "rmse_periodo": float(rmse_val)
        }
        
        if "champion_model" in self.report:
            self.report["champion_model"]["analisis_residuos"] = res_stats
            logger.info("Updating final JSON report with residual analysis statistics.")
            save_report(self.report, self.reports_dir, "phase_05_modeling")
            
        # Logging pretty stats
        logger.info("\n" + "="*40 + "\n" +
                    "📋 ESTADÍSTICAS DETALLADAS DE RESIDUOS\n" +
                    "-"*40 + "\n" +
                    f"🔹 Total de puntos analizados: {res_stats['total_puntos']}\n" +
                    f"🔹 Under-forecasting: {res_stats['under_forecasting']['puntos']} puntos ({res_stats['under_forecasting']['porcentaje']:.1f}%)\n" +
                    f"🔹 Over-forecasting: {res_stats['over_forecasting']['puntos']} puntos ({res_stats['over_forecasting']['porcentaje']:.1f}%)\n" +
                    f"🔹 Error Máximo: {res_stats['error_max_subestimacion']:,.2f}\n" +
                    f"🔹 Error Mínimo: {res_stats['error_min_sobreestimacion']:,.2f}\n" +
                    "-"*40 + "\n" +
                    f"📈 Media del error: {res_stats['media_error']:,.2f}\n" +
                    f"📉 Mediana del error: {res_stats['mediana_error']:,.2f}\n" +
                    f"📏 Desviación estándar: {res_stats['desviacion_estandar_error']:,.2f}\n" +
                    f"🎯 MAE del periodo: {res_stats['mae_periodo']:,.2f}\n" +
                    f"🧪 RMSE del periodo: {res_stats['rmse_periodo']:,.2f}\n" +
                    "="*40)

        # Plot Residual Analysis
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Distribution
        sns.histplot(residuals, kde=True, ax=ax[0], color='#76b5b5')
        ax[0].axvline(0, color='red', linestyle='--', linewidth=3)
        ax[0].set_title("Distribución de Errores (Residuos)", fontsize=14)
        ax[0].set_xlabel("Residual (Real - Predicción)")
        
        # Right: Evolution in time
        ax[1].plot(y_test.index, residuals, marker='o', color='indigo', linewidth=3)
        ax[1].axhline(0, color='red', linestyle='--', linewidth=3)
        ax[1].set_title("Evolución del Error en el Tiempo", fontsize=14)
        ax[1].set_xlabel("Fecha")
        ax[1].set_ylabel("Residual")
        plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        save_figure(fig, fig_phase_dir, "champion_residuals")
        plt.close()

    def retrain_and_save_champion(self):
        """
        Step 3.10 & 3.11: Retrain with full data and persist.
        """
        if not hasattr(self, 'champion_obj'):
            return
            
        champion = self.champion_obj
        forecaster = champion['forecaster']
        exog_cols = champion.get('exog_cols', [])
        name = champion['name']
        
        logger.info(f"--- Step 3.10: Final Retraining on Full Dataset ---")
        
        # Combine all data
        full_data = pd.concat([self.data_train, self.data_val, self.data_test])
        exog_full = full_data[exog_cols] if exog_cols else None
        
        # Fit on full data
        forecaster.fit(y=full_data[self.target], exog=exog_full)
        
        # Step 3.11: Persistence
        model_dir = self.config['general']['paths'].get('models', 'outputs/models')
        logger.info(f"Saving retrained Champion model to {model_dir}")
        save_model(forecaster, model_dir, "champion_forecaster")
        logger.info("Modeling Phase 05 Successfully Completed.")

if __name__ == "__main__":
    print("DEBUG: STARTING TRAINER SCRIPT VERSION 2.0")
    trainer = ForecasterTrainer()
    trainer.load_and_split_data()
    trainer.run_baselines()
    trainer.run_all_experiments()
    trainer.save_final_report()
    trainer.generate_champion_diagnostics()
    trainer.retrain_and_save_champion()
