import pandas as pd
import numpy as np
import os
import yaml
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BusinessValidator:
    """
    Clase encargada de validar las reglas de negocio y financieras definidas en config.yaml.
    Funciona de forma independiente al DataLoader.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # Determinar raíz del proyecto
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(current_dir, ".."))
        
        abs_config_path = os.path.join(self.project_root, config_path)
        with open(abs_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.rules = self.config.get("business_rules", {})
        self.reports_path = self.config["general"]["paths"]["experiments"].get("phase_02")

    def validate_all(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Ejecuta todas las validaciones definidas para los DataFrames proporcionados.
        """
        audit_report = {
            "phase": "Phase 02: Financial & Business Logic Audit",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tables": {}
        }
        
        for table_name, df in data_dict.items():
            if table_name in self.rules:
                logger.info(f"Auditing business rules for {table_name}...")
                audit_report["tables"][table_name] = self._audit_table_rules(table_name, df)
                
        return audit_report

    def _audit_table_rules(self, table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        table_rules = self.rules[table_name]
        results = {}
        
        # 1. Validación de Ventas Diarias
        if table_name == "ventas_diarias":
            results.update(self._validate_ventas(df, table_rules))
        
        # 2. Validación de Redes Sociales
        if table_name == "redes_sociales":
            results.update(self._validate_redes(df, table_rules))
            
        return results

    def _validate_ventas(self, df: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        v = {}
        
        # R1: total_unidades = precio_normal + promo_pagadas + promo_bonificadas
        r1_cols = rules["total_units_sum"]
        diff_units = df[r1_cols["target"]] - df[r1_cols["components"]].sum(axis=1)
        v["total_units_consistency"] = {
            "success": bool((diff_units == 0).all()),
            "errors_count": int((diff_units != 0).sum()),
            "max_diff": float(diff_units.abs().max())
        }
        
        # R2: promo_pagadas == promo_bonificadas
        r2_cols = rules["promo_balanced"]
        diff_promo = df[r2_cols["col_pagadas"]] - df[r2_cols["col_bonificadas"]]
        v["promo_balance_consistency"] = {
            "success": bool((diff_promo == 0).all()),
            "errors_count": int((diff_promo != 0).sum())
        }
        
        # R3: precio >= costo
        r3_cols = rules["price_viability"]
        violation_price = df[r3_cols["price"]] < df[r3_cols["cost"]]
        v["price_viability"] = {
            "success": bool((~violation_price).all()),
            "violations_count": int(violation_price.sum())
        }
        
        # R4: ingresos = (normal + pagadas) * precio_full
        r4_cols = rules["income_calculation"]
        expected_income = df[r4_cols["base_units"]].sum(axis=1) * df[r4_cols["unit_price"]]
        diff_income = (df[r4_cols["target"]] - expected_income).round(2)
        v["income_logic"] = {
            "success": bool((diff_income == 0).all()),
            "errors_count": int((diff_income != 0).sum()),
            "max_diff": float(diff_income.abs().max())
        }
        
        # R5: costo_total = unidades * costo_unitario
        r5_cols = rules["cost_calculation"]
        expected_cost = df[r5_cols["total_units"]] * df[r5_cols["unit_cost"]]
        diff_cost = (df[r5_cols["target"]] - expected_cost).round(2)
        v["cost_logic"] = {
            "success": bool((diff_cost == 0).all()),
            "errors_count": int((diff_cost != 0).sum())
        }
        
        # R6: ingresos >= costo_total
        r6_cols = rules["profitability"]
        loss_days = df[r6_cols["income"]] < df[r6_cols["cost"]]
        v["positive_gross_margin"] = {
            "success": bool((~loss_days).all()),
            "negative_days_count": int(loss_days.sum())
        }
        
        # R7: utilidad = ingresos - costo
        r7_cols = rules["utility_calculation"]
        expected_utility = df[r7_cols["income"]] - df[r7_cols["cost"]]
        diff_utility = (df[r7_cols["target"]] - expected_utility).round(2)
        v["utility_logic"] = {
            "success": bool((diff_utility == 0).all()),
            "errors_count": int((diff_utility != 0).sum())
        }
        
        return v

    def _validate_redes(self, df: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        v = {}
        r_cols = rules["investment_sum"]
        expected_inv = df[r_cols["components"]].sum(axis=1)
        diff_inv = (df[r_cols["target"]] - expected_inv).round(2)
        
        v["investment_consistency"] = {
            "success": bool((diff_inv == 0).all()),
            "errors_count": int((diff_inv != 0).sum()),
            "max_diff": float(diff_inv.abs().max())
        }
        return v
