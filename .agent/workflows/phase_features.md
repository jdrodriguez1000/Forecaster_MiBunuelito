---
description: Generates and validates the feature engineering artifacts and experimental notebook.
---

# Workflow: Phase 04 - Feature Engineering

This workflow automates the generation of the engineered dataset and the corresponding laboratory notebook.

## Steps

1. **Parameters Verification**
   Ensure `config.yaml` has the `features` section configured with engineering settings (cyclical, events, and technical).

2. **Production Execution**
   Run the orchestrator to generate the production artifacts:
   ```bash
   python main.py --phase features
   ```

3. **Notebook Generation**
   Generate the laboratory notebook to visualize the results:
   // turbo
   ```bash
   python scripts/gen_features.py
   ```

4. **Validation**
   - Open `notebooks/04_feature_engineering.ipynb`.
   - Execute all cells and verify that the dataset length is exactly **97 rows** (No future extension).
   - Verify feature distributions and correlation with the target.
   - Check the `outputs/reports/phase_04_feature_engineering/phase_04_feature_engineering_latest.json` for traceability and data previews.

5. **Artifacts Produced**
   - `data/04_processed/master_features.parquet`
   - `notebooks/04_feature_engineering.ipynb`
   - `outputs/reports/phase_04_feature_engineering/`
