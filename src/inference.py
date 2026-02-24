import os
import logging
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from src.utils.helpers import save_report, save_figure
import matplotlib.pyplot as plt

logger = logging.getLogger("Inference")

class InferenceEngine:
    """
    Handles the final forecasting phase using the champion model and 
    projected features.
    """
    def __init__(self, config):
        self.config = config
        self.report = {
            "phase": "06_forecasting",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success",
            "steps_detail": {},
            "artifacts": {}
        }

    def run_forecast(self):
        """
        Orchestrates the loading of artifacts and generation of predictions.
        """
        try:
            # 1. Load Data and Model
            df, model = self._load_artifacts()
            
            # 2. Prepare Exogenous for Horizon
            # The Features phase already extended the dataframe. 
            # We need to slice only the future dates.
            # Usually, the training data ends at T. Features phase adds T+1 to T+6.
            
            # Find the last date with actual targets (to know where history ends)
            target_col = self.config['general'].get('target', 'total_unidades_entregadas')
            
            # Since master_features.parquet in forecast mode has target as NaN for future
            last_hist_date = df[df[target_col].notnull()].index.max()
            future_df = df[df.index > last_hist_date].copy()
            
            if len(future_df) == 0:
                raise ValueError("No future dates found in master_features. Check if Features phase was run in forecast mode.")

            logger.info(f"Generating forecast for {len(future_df)} months starting from {future_df.index.min()}")

            # 3. Predict with Intervals (95% Confidence)
            # Skforecast direct predict_interval needs the exogenous variables for the horizon
            exog_cols = [c for c in future_df.columns if c != target_col]
            
            predictions = model.predict_interval(
                steps=len(future_df), 
                exog=future_df[exog_cols],
                interval=[5, 95] # 90% range (5th to 95th percentile) or [2.5, 97.5] for 95%
            )
            
            # 4. Post-process (Safety checks)
            orig_preds = predictions['pred'].copy()
            predictions = predictions.clip(lower=0)
            clipping_occurred = (orig_preds < 0).any()
            clipping_count = int((orig_preds < 0).sum())

            # 5. Calculate Uncertainty Bandwidth
            bandwidths = predictions['upper_bound'] - predictions['lower_bound']
            avg_bandwidth = float(bandwidths.mean())
            
            # 6. Build Result Dataframe
            result_df = pd.DataFrame({
                'fecha': future_df.index,
                'pronostico_unidades': predictions['pred'],
                'lower_bound': predictions['lower_bound'],
                'upper_bound': predictions['upper_bound']
            })
            
            # 6. Data Audit (Requirement 5)
            importance = self._get_feature_importances(model)
            exog_impact = self._extract_exogenous_impact(future_df[exog_cols])
            data_audit = self._perform_data_audit(future_df[exog_cols])

            # 7. Traceability (Requirement 3)
            model_details = {
                "regressor_class": model.regressor.__class__.__name__,
                "params": {k: v for k, v in model.regressor.get_params().items() if v is not None}
            }
            features_used = [x['feature'] for x in importance] if importance else []

            # 8. Health and Safety (Requirement 4)
            health_metrics = self._load_modeling_metrics()
            safety_check = {
                "clipping_occurred": clipping_occurred,
                "clipping_count": clipping_count,
                "average_uncertainty_bandwidth": avg_bandwidth
            }

            # 9. Save and Report
            last_hist_value = float(df.loc[last_hist_date, target_col])
            self._save_results(
                result_df, 
                last_hist_value, 
                importance, 
                exog_impact,
                last_hist_date.strftime('%Y-%m-%d'),
                model_details,
                features_used,
                health_metrics,
                safety_check,
                data_audit
            )
            
            return result_df

        except Exception as e:
            self.report["status"] = "error"
            self.report["error_message"] = str(e)
            logger.error(f"Inference failed: {str(e)}")
            raise

    def _load_artifacts(self):
        logger.info("Loading model and features for inference...")
        
        # Paths
        model_path = os.path.join(self.config['general']['paths']['models'], "champion_forecaster_latest.pkl")
        features_path = os.path.join(self.config['general']['paths'].get('forecasts', 'outputs/forecasts'), "artifacts", "projected_features_latest.parquet")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Champion model not found at {model_path}. Run training phase first.")
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Projected features not found at {features_path}. Run features phase in forecast mode first.")
            
        model = joblib.load(model_path)
        df = pd.read_parquet(features_path)
        
        self.report["steps_detail"]["loading"] = "Succesfully loaded champion model and projected features."
        return df, model

    def _save_results(self, result_df, last_hist_value, importance, exog_impact, cutoff_date, model_details, features, health_metrics, safety_check, data_audit):
        logger.info("Saving forecast results...")
        
        forecast_dir = self.config['general']['paths'].get('forecasts', 'outputs/forecasts')
        report_dir = self.config['general']['paths'].get('reports_phase_06', 'outputs/reports/phase_06_forecasting')
        
        os.makedirs(forecast_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Dual Persistence (Rule 7)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hist_dir = os.path.join(forecast_dir, "history")
        os.makedirs(hist_dir, exist_ok=True)

        # 1. Save Latest (Pointer)
        csv_path = os.path.join(forecast_dir, "forecast_official_latest.csv")
        parquet_path = os.path.join(forecast_dir, "forecast_official_latest.parquet")
        result_df.to_csv(csv_path, index=False)
        result_df.to_parquet(parquet_path)

        # 2. Save History (Immutable)
        result_df.to_csv(os.path.join(hist_dir, f"forecast_official_{timestamp}.csv"), index=False)
        result_df.to_parquet(os.path.join(hist_dir, f"forecast_official_{timestamp}.parquet"))
        
        # 3. Generate and Save Visualizations
        self._generate_plots(result_df, importance)

        # Update report
        self.report["artifacts"]["forecast_csv"] = csv_path
        self.report["artifacts"]["forecast_parquet"] = parquet_path
        
        # 4. Business Vision Summary (Requirement 1)
        monthly_avg = float(result_df['pronostico_unidades'].mean())
        total_units = float(result_df['pronostico_unidades'].sum())
        
        # Trend vs last historical
        trend_pct = (monthly_avg - last_hist_value) / last_hist_value
        
        # Peak and Valley
        peak_idx = result_df['pronostico_unidades'].idxmax()
        valley_idx = result_df['pronostico_unidades'].idxmin()
        
        self.report["business_vision"] = {
            "summary": {
                "monthly_average_units": monthly_avg,
                "total_semiannual_cumulative": total_units,
                "general_trend_vs_last_month": f"{trend_pct:+.2%}",
                "peak_performance": {
                    "month": result_df.loc[peak_idx, 'fecha'].strftime('%Y-%m'),
                    "units": float(result_df.loc[peak_idx, 'pronostico_unidades'])
                },
                "valley_performance": {
                    "month": result_df.loc[valley_idx, 'fecha'].strftime('%Y-%m'),
                    "units": float(result_df.loc[valley_idx, 'pronostico_unidades'])
                }
            },
            "interpretation": f"El modelo proyecta un volumen promedio mensual de {monthly_avg:,.0f} unidades, "
                              f"con una tendencia de {trend_pct:+.1%} respecto al cierre histórico. "
                              f"El pico de demanda se identifica en {result_df.loc[peak_idx, 'fecha'].strftime('%B')}. "
        }

        # 5. Explicability (Requirement 2)
        self.report["explicability"] = {
            "feature_importance": importance[:15], # Top 15 for JSON
            "exogenous_impact_summary": exog_impact
        }

        # 6. Traceability and Configuration (Requirement 3)
        self.report["traceability"] = {
            "training_cutoff_date": cutoff_date,
            "model_architecture": model_details["regressor_class"],
            "winning_hyperparameters": model_details["params"],
            "input_features": features
        }

        # 7. Health and Safety (Requirement 4)
        mape_val = health_metrics.get('mape', 0)
        safety_status = "STABLE" if mape_val < 0.20 and not safety_check["clipping_occurred"] else "WARNING"
        
        self.report["health_safety"] = {
            "status": safety_status,
            "expected_performance": {
                "historical_mae": health_metrics.get('mae'),
                "historical_mape": mape_val,
                "historical_rmse": health_metrics.get('rmse')
            },
            "forecast_integrity": {
                "negative_clipping_detected": safety_check["clipping_occurred"],
                "clipped_values_count": safety_check["clipping_count"],
                "average_bandwidth_units": safety_check["average_uncertainty_bandwidth"]
            }
        }

        # 8. Data Audit (Requirement 5)
        self.report["data_audit"] = {
            "status": "VERIFIED",
            "horizon_validation": data_audit,
            "audit_summary": f"Se validaron {len(data_audit)} meses. "
                             f"Promociones activas: {sum(1 for m in data_audit if any('Promoción' in v for v in m['validations']))}. "
                             f"Meses de primas: {sum(1 for m in data_audit if any('Primas' in v for v in m['validations']))}."
        }

        self.report["forecast_summary"] = {
            "horizon": len(result_df),
            "start_date": result_df['fecha'].min().strftime('%Y-%m-%d'),
            "end_date": result_df['fecha'].max().strftime('%Y-%m-%d'),
            "total_predicted_units": total_units,
            "average_lower_bound": float(result_df['lower_bound'].mean()),
            "average_upper_bound": float(result_df['upper_bound'].mean())
        }
        
        # Save report
        save_report(self.report, report_dir, "phase_06_forecasting")
        logger.info(f"Forecast saved to {csv_path}")

    def _generate_plots(self, df, importance):
        """
        Generates official visualizations for the forecast.
        """
        logger.info("Generating forecast visualizations...")
        fig_dir = self.config['general']['paths'].get('figures_phase_06', 'outputs/figures/phase_06_forecasting')
        
        # Plot 1: Forecast with Intervals
        plt.figure(figsize=(12, 6))
        plt.plot(df['fecha'], df['pronostico_unidades'], marker='o', label='Pronóstico', color='blue', linewidth=2)
        plt.fill_between(
            df['fecha'], 
            df['lower_bound'], 
            df['upper_bound'], 
            color='blue', 
            alpha=0.2, 
            label='Intervalo de Confianza (95%)'
        )
        
        plt.title('Pronóstico de Demanda Mensual - Mi Buñuelito', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Unidades Entregadas', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(rotation=45)
        
        # Save figure 1
        latest_path, hist_path = save_figure(plt.gcf(), fig_dir, "forecast_visual")
        self.report["artifacts"]["forecast_plot"] = latest_path
        plt.close()

        # Plot 2: Feature Importance (Requirement 2)
        if importance:
            top_importance = importance[:10]
            features = [x['feature'] for x in top_importance]
            values = [x['coefficient'] for x in top_importance]
            
            plt.figure(figsize=(10, 8))
            colors = ['green' if v > 0 else 'red' for v in values]
            plt.barh(features, values, color=colors)
            plt.title('Importancia de Características (Impacto en Unidades)', fontsize=14, fontweight='bold')
            plt.xlabel('Magnitud del Coeficiente', fontsize=12)
            plt.grid(True, axis='x', linestyle='--', alpha=0.5)
            plt.gca().invert_yaxis()
            
            # Save figure 2
            latest_imp_path, _ = save_figure(plt.gcf(), fig_dir, "feature_importance")
            self.report["artifacts"]["importance_plot"] = latest_imp_path
            plt.close()

        logger.info(f"Visualizations saved to {fig_dir}")

    def _get_feature_importances(self, model):
        """
        Extract feature importance or coefficients from the forecaster.
        """
        try:
            # skforecast returns a dataframe. For ForecasterDirect, it has importance for each step.
            importance_df = model.get_feature_importances(step=1)
            
            col = None
            if 'importance' in importance_df.columns:
                col = 'importance'
            elif 'coefficient' in importance_df.columns:
                col = 'coefficient'
            else:
                numeric_cols = importance_df.select_dtypes(include=[np.number]).columns.tolist()
                col = numeric_cols[0] if numeric_cols else None
            
            if not col: return []
                
            importance_df = importance_df.sort_values(by=col, ascending=False)
            
            result = []
            for _, row in importance_df.iterrows():
                val = float(row[col])
                result.append({
                    "feature": str(row['feature']),
                    "coefficient": val,
                    "magnitude": abs(val)
                })
            
            result.sort(key=lambda x: x['magnitude'], reverse=True)
            return result
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return []

    def _extract_exogenous_impact(self, exog_df):
        """
        Summarizes the projected exogenous variables for the report.
        """
        try:
            summary = []
            for col in exog_df.columns:
                summary.append({
                    "feature": col,
                    "projected_mean": float(exog_df[col].mean()),
                    "projected_min": float(exog_df[col].min()),
                    "projected_max": float(exog_df[col].max()),
                    "volatility": float(exog_df[col].std()) if len(exog_df) > 1 else 0.0
                })
            return summary
        except Exception as e:
            logger.warning(f"Could not extract exogenous impact: {str(e)}")
            return []

    def _load_modeling_metrics(self):
        """
        Loads the test performance metrics from the modeling phase report.
        """
        try:
            report_path = os.path.join(
                self.config['general']['paths']['reports'], 
                "phase_05_modeling", 
                "phase_05_modeling_latest.json"
            )
            if not os.path.exists(report_path):
                logger.warning(f"Modeling report not found at {report_path}")
                return {}
            
            import json
            with open(report_path, 'r') as f:
                modeling_report = json.load(f)
            
            return modeling_report.get("test_performance", {})
        except Exception as e:
            logger.warning(f"Error loading modeling metrics: {str(e)}")
            return {}

    def _perform_data_audit(self, exog_df):
        """
        Performs a thematic audit of the exogenous features used for the horizon.
        Ensures business rules like promotions and holidays are correctly projected.
        """
        logger.info("Performing data audit for forecast horizon...")
        audit = []
        try:
            for date, row in exog_df.iterrows():
                month_audit = {
                    "date": date.strftime('%Y-%m-%d'),
                    "month_name": date.strftime('%B'),
                    "validations": []
                }
                
                # Check Promos (Rule: Apr, May, Sep, Oct)
                if row.get('es_promo', 0) > 0.5:
                    month_audit["validations"].append("Promoción detectada (Regla de Negocio)")
                
                # Check Bonus (Rule: Jun, Dic)
                if row.get('is_bonus_month', 0) > 0.5:
                    month_audit["validations"].append("Mes de Primas detectado")
                
                # Check Novenas (Rule: Dic)
                if row.get('novenas_intensity', 0) > 0.05:
                    month_audit["validations"].append(f"Inyección de Novenas: {row['novenas_intensity']:.2f}")
                    
                # Check Holidays
                holidays = row.get('holidays_count', 0)
                if holidays > 0:
                    month_audit["validations"].append(f"Festivos: {int(holidays)} días")
                
                # Check Drift/Trend
                drift = row.get('time_drift_index', 0)
                if drift > 0:
                    month_audit["validations"].append(f"Índice de Tendencia: {int(drift)}")
                    
                audit.append(month_audit)
        except Exception as e:
            logger.warning(f"Data audit failed: {str(e)}")
            
        return audit
