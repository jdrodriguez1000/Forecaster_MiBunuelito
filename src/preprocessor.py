import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from src.utils.helpers import save_report

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Data cleaning, aggregation, and null handling following the 11-step protocol.
    Adheres to Project Rules for Traceability and Deliverables.
    """
    def __init__(self, config):
        self.config = config
        self.report = {
            "phase": "02_preprocessing",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "steps_detail": {},
            "data_profile": {
                "initial": {},
                "intermediate": {},
                "final_master": {}
            },
            "artifacts": {}
        }

    def process(self, data_dict):
        """
        Execute the 11-step preprocessing pipeline.
        Args:
            data_dict (dict): Dictionary of pandas DataFrames {table_name: df}
        Returns:
            pd.DataFrame: Merged and cleansed master dataset
        """
        logger.info("Starting preprocessing pipeline...")
        processed_data = data_dict.copy()

        try:
            # Step 1: Contract Standardization & Validation
            self._step_01_validate_contracts(processed_data)

            # Step 2: Immediate Structural Cleaning
            self._step_02_drop_immediate(processed_data)

            # Step 3: Temporal Shielding (Anti-Leakage)
            self._step_03_anti_leakage(processed_data)

            # Step 4: Duplicate Management
            self._step_04_handle_duplicates(processed_data)

            # Step 5: Sentinel to Null Transformation
            self._step_05_sentinels_to_null(processed_data)

            # Step 6: Reindexing and Temporal Integrity (Gaps)
            self._step_06_reindex_gaps(processed_data)

            # Step 7: Domain Imputation Protocols
            self._step_07_apply_imputation(processed_data)

            # Step 8: Truth Recalculation (Technical Phase)
            self._step_08_recalculate_target(processed_data)

            # Step 9: Support Column Deletion (Anti-Laziness)
            self._step_09_drop_support_columns(processed_data)

            # Step 10: Aggregation and Merging
            master_df = self._step_10_aggregate_and_merge(processed_data)

            # Step 11: Artifact and Report Generation
            master_df = self._step_11_generate_artifacts(master_df)

            return master_df

        except Exception as e:
            self.report["status"] = "error"
            self.report["error_message"] = str(e)
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def _step_01_validate_contracts(self, data):
        logger.info("Step 1: Validating contracts...")
        details = []
        for table, df in data.items():
            contract = self.config['extractions']['tables'][table]['contract']
            missing = [col for col in contract.keys() if col not in df.columns]
            if missing:
                error_msg = f"CRITICAL: Missing columns in {table}: {missing}"
                raise ValueError(error_msg)
            
            date_col = self.config['extractions']['tables'][table]['date_column']
            data[table][date_col] = pd.to_datetime(data[table][date_col])
            
            self.report["data_profile"]["initial"][table] = {
                "shape": df.shape,
                "columns": list(df.columns)
            }
            details.append(f"Validated {table} contract.")
        
        self.report["steps_detail"]["01_contract_validation"] = details

    def _step_02_drop_immediate(self, data):
        logger.info("Step 2: Dropping immediate columns...")
        drops = self.config['preprocessing']['cleaning']['drop_immediate']
        details = []
        for table, df in data.items():
            if table in drops:
                to_drop = [c for c in drops[table] if c in df.columns]
                data[table] = df.drop(columns=to_drop)
                details.append(f"Dropped {to_drop} from {table}")
        
        self.report["steps_detail"]["02_structural_cleaning"] = details

    def _step_03_anti_leakage(self, data):
        logger.info("Step 3: Applying anti-leakage...")
        anti_leakage = self.config['preprocessing']['anti_leakage']
        if not anti_leakage['active']:
            self.report["steps_detail"]["03_anti_leakage"] = "Inactive"
            return

        today = pd.Timestamp.now()
        limit_date = (today.replace(day=1) - pd.Timedelta(days=1)).normalize()
        details = []

        for table, df in data.items():
            date_col = self.config['extractions']['tables'][table]['date_column']
            initial_count = len(df)
            data[table] = df[df[date_col] <= limit_date]
            removed = initial_count - len(data[table])
            details.append(f"Removed {removed} rows from {table} (Date > {limit_date})")
        
        self.report["steps_detail"]["03_anti_leakage"] = details

    def _step_04_handle_duplicates(self, data):
        logger.info("Step 4: Handling duplicates...")
        strategy = self.config['preprocessing']['duplicates']['strategy']
        details = []
        for table, df in data.items():
            date_col = self.config['extractions']['tables'][table]['date_column']
            initial_count = len(df)
            
            df = df.drop_duplicates(keep='last' if strategy == 'keep_last' else 'first')
            df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep='last')
            
            data[table] = df
            removed = initial_count - len(df)
            details.append(f"Removed {removed} duplicate entries/dates from {table}")
        
        self.report["steps_detail"]["04_duplicate_management"] = details

    def _step_05_sentinels_to_null(self, data):
        logger.info("Step 5: Converting sentinels up to nulls...")
        sentinels = self.config['extractions']['sentinels']
        all_sentinels = (
            sentinels.get('numeric', []) + 
            sentinels.get('categorical', []) + 
            sentinels.get('datetime', [])
        )
        details = []
        for table, df in data.items():
            initial_nulls = df.isnull().sum().sum()
            data[table] = df.replace(all_sentinels, np.nan)
            final_nulls = data[table].isnull().sum().sum()
            details.append(f"Converted {int(final_nulls - initial_nulls)} sentinels to NaN in {table}")
        
        self.report["steps_detail"]["05_sentinel_transformation"] = details

    def _step_06_reindex_gaps(self, data):
        logger.info("Step 6: Reindexing gaps...")
        freqs = self.config['preprocessing']['reindexing']['frequencies']
        details = []
        for table, freq in freqs.items():
            if table not in data:
                continue
            
            if freq is None:
                details.append(f"Skipped {table} (native frequency)")
                continue
            
            df = data[table]
            date_col = self.config['extractions']['tables'][table]['date_column']
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            full_range = pd.date_range(start=min_date, end=max_date, freq=freq)
            
            df = df.set_index(date_col).reindex(full_range).rename_axis(date_col).reset_index()
            data[table] = df
            details.append(f"Reindexed {table} with frequency {freq}")
        
        self.report["steps_detail"]["06_reindexing"] = details

    def _step_07_apply_imputation(self, data):
        logger.info("Step 7: Applying imputation protocols...")
        rules = self.config['preprocessing']['imputation']
        details = []
        
        if 'macro_economia' in data:
            data['macro_economia'] = data['macro_economia'].ffill().bfill()
            details.append("Macro: Applied ffill/bfill.")

        if 'promocion_dia' in data:
            df = data['promocion_dia']
            date_col = self.config['extractions']['tables']['promocion_dia']['date_column']
            df['es_promo'] = df['es_promo'].fillna(rules['promocion_dia']['default_value'])
            df.loc[df[date_col].dt.year < rules['promocion_dia']['relevance_year'], 'es_promo'] = 0
            data['promocion_dia'] = df
            details.append("Promociones: Applied default and relevance year filter.")

        if 'redes_sociales' in data:
            data['redes_sociales'] = data['redes_sociales'].fillna(rules['redes_sociales']['default_investment'])
            details.append("Redes Sociales: Applied default 0 to investment.")

        if 'ventas_diarias' in data:
            df = data['ventas_diarias']
            unit_cols = self.config['preprocessing']['cleaning']['target_build_components']
            df[unit_cols] = df[unit_cols].ffill().fillna(0)
            
            financial_cols = [rules['ventas_diarias']['financials']['price_col'], 
                            rules['ventas_diarias']['financials']['cost_col']]
            df[financial_cols] = df[financial_cols].ffill().bfill()
            data['ventas_diarias'] = df
            details.append("Ventas: Imputed units (ffill/0) and financials (ffill/bfill).")

        self.report["steps_detail"]["07_imputation"] = details

    def _step_08_recalculate_target(self, data):
        logger.info("Step 8: Recalculating target variable...")
        vd_rules = self.config['business_rules']['ventas_diarias']['total_units_recalculation']
        
        if 'ventas_diarias' not in data:
            return

        df = data['ventas_diarias']
        target = vd_rules['target']
        components = vd_rules['components']
        
        df[target] = df[components].sum(axis=1)
        data['ventas_diarias'] = df
        self.report["steps_detail"]["08_truth_recalculation"] = f"Recalculated {target} from components."

    def _step_09_drop_support_columns(self, data):
        logger.info("Step 9: Dropping support columns...")
        components = self.config['preprocessing']['cleaning']['target_build_components']
        if 'ventas_diarias' in data:
            data['ventas_diarias'] = data['ventas_diarias'].drop(columns=components)
            self.report["steps_detail"]["09_anti_laziness_cleanup"] = f"Dropped support columns {components}."

    def _step_10_aggregate_and_merge(self, data):
        logger.info("Step 10: Aggregating and merging...")
        agg_level = self.config['preprocessing']['aggregation_level']
        processed_dfs = {}
        for table, df in data.items():
            date_col = self.config['extractions']['tables'][table]['date_column']
            df = df.set_index(date_col)
            
            if agg_level == "monthly":
                if table == 'ventas_diarias':
                    resampled = df.resample('MS').agg({
                        self.config['preprocessing']['target_variable']: 'sum',
                        self.config['preprocessing']['imputation']['ventas_diarias']['financials']['price_col']: 'mean',
                        self.config['preprocessing']['imputation']['ventas_diarias']['financials']['cost_col']: 'mean'
                    })
                elif table == 'promocion_dia':
                    resampled = df.resample('MS').agg({'es_promo': 'max'})
                elif table == 'redes_sociales':
                    resampled = df.resample('MS').sum()
                elif table == 'macro_economia':
                    resampled = df.resample('MS').mean()
                
                processed_dfs[table] = resampled.reset_index()

        master_df = None
        for table, df in processed_dfs.items():
            date_col = self.config['extractions']['tables'][table]['date_column']
            if master_df is None:
                master_df = df
            else:
                master_df = pd.merge(master_df, df, on=date_col, how='outer')

        date_col_name = master_df.columns[0]
        master_df = master_df.sort_values(date_col_name).reset_index(drop=True)
        master_df = master_df.ffill().fillna(0)
        
        self.report["steps_detail"]["10_aggregation_merge"] = f"Consolidated into monthly {master_df.shape} dataset."
        return master_df

    def _step_11_generate_artifacts(self, master_df):
        logger.info("Step 11: Generating artifacts...")
        output_dir = self.config['general']['paths']['cleansed']
        os.makedirs(output_dir, exist_ok=True)
        
        # Establacer el índice como serie de tiempo (columna fecha)
        date_col = master_df.columns[0]
        master_df = master_df.set_index(date_col)
        
        # Save Parquet
        output_name = "master_cleansed.parquet"
        output_path = os.path.join(output_dir, output_name)
        master_df.to_parquet(output_path) # Parquet mantendrá el índice
        
        # Profile Master Data for JSON Report
        self.report["data_profile"]["final_master"] = {
            "name": output_name,
            "path": output_path,
            "shape": master_df.shape,
            "index_name": master_df.index.name,
            "index_dtype": str(master_df.index.dtype),
            "column_dtypes": {col: str(dtype) for col, dtype in master_df.dtypes.items()},
            "null_count": int(master_df.isnull().sum().sum()),
            "duplicate_rows": int(master_df.duplicated().sum()),
            "duplicate_dates": int(master_df.index.duplicated().sum()),
            "samples": {
                "head_3": master_df.head(3).reset_index().to_dict(orient="records"),
                "tail_3": master_df.tail(3).reset_index().to_dict(orient="records"),
                "random_3": master_df.sample(min(3, len(master_df))).reset_index().to_dict(orient="records")
            }
        }
        
        # Save Report
        report_dir = os.path.join(self.config['general']['paths']['reports'], "phase_02_preprocessing")
        save_report(self.report, report_dir, "phase_01_preprocessing" if "01A" in str(self.config.get("general",{}).get("paths",{}).get("reports","")) else "phase_02_preprocessing")
        
        self.report["artifacts"]["master_dataset"] = output_path
        self.report["steps_detail"]["11_artifact_generation"] = "Exported Parquet with Time Index and generated traceability report."
        return master_df
