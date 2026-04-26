# Model Reports

This folder contains generated **metrics, charts, and visualizations** for the trained dual-engine system:

- Crop recommendation (classification): **BayesNet** and **RandomForest**
- Yield prediction (regression): **tabular model** (XGBoost or sklearn fallback)
- Optional: yield deep model (**CNN→LSTM**) if `yield_cnn_lstm.keras` is present

## Generate a new report

From the repository root:

```bash
python Models/generate_model_reports.py
```

Outputs are written to `reports/run_<timestamp>/`.

## Output contents

Each run folder contains:

- `run_info.json` (environment + run configuration)
- `crop_engine/`
  - `random_forest_metrics.json`
  - `random_forest_confusion_matrix.png`
  - `random_forest_f1_by_class.png`
  - `bayesnet_metrics.json`
  - `bayesnet_confusion_matrix.png`
  - `bayesnet_f1_by_class.png`
- `yield_tabular/`
  - `metrics.json`
  - `actual_vs_predicted.png`
  - `residuals.png`
  - `mae_by_crop_top.png`
- `yield_deep/`
  - `metrics.json`
  - `actual_vs_predicted.png` (only when deep model is available)
  - `residuals.png` (only when deep model is available)
