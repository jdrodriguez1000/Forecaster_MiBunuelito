import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
from src.utils.config_loader import load_config
from src.connectors.db_connector import DBConnector

# Logging configuration (English to avoid encoding issues)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data Extraction and Audit Engine.
    Downloads data from Supabase API and performs exhaustive diagnostic
    based on a data contract defined in config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes loader using strict definitions from config.yaml.
        """
        # Search for config in project root relative to src/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(current_dir, ".."))
        
        # Load config utilizing official utility (No custom logic here)
        config_abs_path = os.path.join(self.project_root, config_path)
        self.config = load_config(config_abs_path)
        
        self.db_connector = DBConnector()
        
        # DYNAMIC PATH RESOLUTION FROM CONFIG.YAML
        # We respect the "raw" path defined in general:paths
        raw_rel_path = self.config["general"]["paths"]["raw"]
        self.raw_path = os.path.join(self.project_root, raw_rel_path)
        os.makedirs(self.raw_path, exist_ok=True)

    def load_and_audit(self, force_full_load: bool = False) -> Dict[str, Any]:
        """
        Orchestrates the Phase 01: Discovery & Extraction.
        Rule 2.35: Strictly incremental extraction with recursive pagination.
        """
        tables_to_load = self.config["extractions"]["tables"]
        full_report = {
            "phase": "Phase 01: Data Discovery & Extraction",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tables": {}
        }

        # Initialize Supabase client
        supabase = self.db_connector.get_client()

        for table_key, table_info in tables_to_load.items():
            table_name = table_info["table_name"]
            date_col = table_info.get("date_column")
            output_file = os.path.join(self.raw_path, f"{table_name}.parquet")
            
            df_local = pd.DataFrame()
            if os.path.exists(output_file):
                df_local = pd.read_parquet(output_file)

            # --- RULE 2.35: INCREMENTAL EXTRACTION WITH PAGINATION ---
            if force_full_load or df_local.empty:
                logger.info(f"FULL LOAD for {table_name} (Recursive Pagination)...")
                df_new = self._fetch_all_with_pagination(supabase, table_name)
                df_total = df_new
            else:
                max_date = str(df_local[date_col].max())
                logger.info(f"INCREMENTAL LOAD for {table_name} since {max_date} (Recursive Pagination)")
                
                # Fetch all newer records using pagination
                df_incremental = self._fetch_all_with_pagination(supabase, table_name, date_col, max_date)
                
                if not df_incremental.empty:
                    logger.info(f"Adding {len(df_incremental)} new records to {table_name}.")
                    df_total = pd.concat([df_local, df_incremental], ignore_index=True)
                    # Respect Rules 4.54/4.55 (Keep last instance if duplicates arise)
                    df_total = df_total.drop_duplicates(subset=[date_col] if date_col else None, keep='last')
                else:
                    logger.info(f"No new records found for {table_name}.")
                    df_total = df_local

            # Persistence in local Parquet
            if not df_total.empty:
                df_total.to_parquet(output_file, index=False)
            
            # Audit execution (Ensures the 13 points are checked)
            report = self._audit_table(df_total, table_info)
            full_report["tables"][table_name] = report

        return full_report

    def _fetch_all_with_pagination(self, supabase, table_name: str, date_col: str = None, after_date: str = None) -> pd.DataFrame:
        """
        Solves the 1,000 record limit by fetching data in chunks recursively.
        """
        all_data = []
        start = 0
        page_size = 1000
        
        while True:
            query = supabase.table(table_name).select("*")
            
            # Application of incremental filter if provided
            if date_col and after_date:
                query = query.gt(date_col, after_date)
            
            # Apply range for pagination
            response = query.range(start, start + page_size - 1).execute()
            data_chunk = response.data
            
            if not data_chunk:
                break
                
            all_data.extend(data_chunk)
            
            # If we got fewer records than a full page, it means there are no more
            if len(data_chunk) < page_size:
                break
                
            start += page_size
            
        return pd.DataFrame(all_data)

    def _audit_table(self, df: pd.DataFrame, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes 13 audit points grouped by sections.
        """
        if df.empty:
            return {"error": "DataFrame is empty, cannot perform audit."}

        contract = table_info.get("contract", {})
        date_col = table_info.get("date_column")
        report = {}

        # --- SECTION 1: DATA CONTRACT & STRUCTURAL INTEGRITY ---
        audit_struct = {}
        expected_cols = set(contract.keys())
        current_cols = set(df.columns)
        audit_struct["contract_fulfilled"] = expected_cols.issubset(current_cols)
        audit_struct["additional_columns"] = list(current_cols - expected_cols)
        audit_struct["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
        report["structural_integrity"] = audit_struct

        # --- SECTION 2: DATA QUALITY & CLEANING ---
        audit_quality = {}
        audit_quality["null_counts"] = df.isnull().sum().to_dict()
        audit_quality["duplicate_rows_count"] = df.duplicated().sum().item()
        
        if date_col and date_col in df.columns:
            date_dupes = df.duplicated(subset=[date_col], keep=False) & ~df.duplicated(keep=False)
            audit_quality["duplicate_dates_count"] = date_dupes.sum().item()
        else:
            audit_quality["duplicate_dates_count"] = 0
        
        # Sentinel values from config
        sentinel_config = self.config.get("extractions", {}).get("sentinels", {})
        all_sentinels = []
        for v in sentinel_config.values():
            if isinstance(v, list):
                all_sentinels.extend(v)
        
        audit_quality["sentinel_counts"] = {col: df[col].isin(all_sentinels).sum().item() for col in df.columns}
        audit_quality["zero_variance_cols"] = [col for col in df.columns if df[col].nunique() <= 1]
        audit_quality["high_cardinality_cols"] = [col for col in df.select_dtypes(include=['object']).columns 
                                                 if len(df) > 0 and (df[col].nunique() / len(df)) > 0.5]
        report["data_quality"] = audit_quality

        # --- SECTION 3: TIME SERIES HEALTH ---
        audit_ts = {}
        if date_col and date_col in df.columns:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            audit_ts["date_range"] = {
                "min": df_temp[date_col].min().strftime("%Y-%m-%d") if not df_temp[date_col].isnull().all() else None,
                "max": df_temp[date_col].max().strftime("%Y-%m-%d") if not df_temp[date_col].isnull().all() else None
            }
            
            if not df_temp[date_col].isnull().all():
                full_range = pd.date_range(start=df_temp[date_col].min(), end=df_temp[date_col].max(), freq='D')
                missing_dates = full_range.difference(df_temp[date_col])
                audit_ts["gaps_detected_count"] = len(missing_dates)
                audit_ts["has_gaps"] = len(missing_dates) > 0
        
        report["time_series_health"] = audit_ts

        # --- SECTION 4: STATISTICAL PROFILING & ANOMALIES ---
        audit_stats = {}
        num_cols = df.select_dtypes(include=[np.number]).columns
        obj_cols = df.select_dtypes(include=['object', 'category']).columns

        stats_numeric = {}
        for col in num_cols:
            desc = df[col].describe().to_dict()
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum().item()
            
            stats_numeric[col] = {
                "descriptive": desc,
                "outliers": {
                    "count": outliers_count,
                    "lower_limit": float(lower_bound),
                    "upper_limit": float(upper_bound)
                }
            }
        audit_stats["numeric_profile"] = stats_numeric

        stats_categorical = {}
        for col in obj_cols:
            counts = df[col].value_counts()
            percents = df[col].value_counts(normalize=True) * 100
            stats_categorical[col] = {
                str(val): {"count": counts[val].item(), "percentage": round(percents[val], 2)}
                for val in counts.index[:10]
            }
        audit_stats["categorical_profile"] = stats_categorical
        report["statistical_profiling"] = audit_stats

        return report
