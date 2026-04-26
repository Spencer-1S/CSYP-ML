# Dual-Engine Crop Recommendation and Yield Prediction

This repository implements a **dual-engine machine learning system** for precision agriculture:

- **Engine 1 (Classification)**: crop recommendation from soil + weather features.
- **Engine 2 (Regression)**: yield prediction from production data (tabular model) with an optional deep model for next-yield forecasting from past yield history.

The primary implementation is in [Models/dual_engine_system.py](Models/dual_engine_system.py) and includes training, evaluation, persistence (save/load), and a small Flask web app for inference.

---

## Models

### Engine 1 — Crop recommendation (classification)

Trained on [Datasets/Crop_recommendation.csv](Datasets/Crop_recommendation.csv).

- **Bayesian Network classifier** (`SimpleDiscreteBayesNetClassifier`)
   - Discretizes numeric features into quantile bins.
   - Models a dependency (rainfall/temperature → humidity) rather than assuming independence.
- **RandomForestClassifier (tuned)**
   - Uses a preprocessing pipeline (imputation + scaling for numeric, one-hot for categoricals).
   - Hyperparameters are tuned with **GridSearchCV** and **Stratified K-Fold** CV.
   - Explicitly compares `gini` vs `entropy` split criteria.

Numeric inputs: `N, P, K, temperature, humidity, ph, rainfall`.

Note: the crop dataset does not contain `City`/`Season`, but the system accepts them; missing values are imputed to `"Unknown"`.

### Engine 2 — Yield prediction (regression)

Trained on [Datasets/IndiaAgricultureCropProduction.csv](Datasets/IndiaAgricultureCropProduction.csv).

1) **Tabular model** (default)
- Preprocessing: numeric impute + scale; categorical impute + one-hot.
- Regressor:
   - **XGBoost** (`XGBRegressor`) if available.
   - Fallback: `HistGradientBoostingRegressor` if XGBoost is not installed.

2) **Optional deep model** (CNN→LSTM)
- Learns a next-step yield forecast from **sliding windows** of past yields.
- Enabled via `--train-dl`.
- Inference requires past yield history of length `--dl-window`.

---

## Repository Structure

```
CSYP-ML/
   Datasets/
      Crop_recommendation.csv
      DistrictWiseRainfallNormal.csv
      IndiaAgricultureCropProduction.csv
   Models/
      dual_engine_system.py
      model.ipynb
      artifacts/
         dual_engine_system/         # generated after training
   webapp/
      app.py
      templates/
         index.html
   README.md
```

Dataset note: the dual-engine training script currently uses `Crop_recommendation.csv` and `IndiaAgricultureCropProduction.csv`. `DistrictWiseRainfallNormal.csv` is included for exploration/extension.

---

## Setup

### Prerequisites

- Python **3.10+**

### Install dependencies

Minimum (train + run tabular models):

```bash
pip install pandas numpy scikit-learn joblib
```

For the web app:

```bash
pip install flask
```

For better yield performance (tabular):

```bash
pip install xgboost
```

For the optional deep yield model (CNN→LSTM):

```bash
pip install tensorflow
```

---

## Train / Evaluate / Save Artifacts

All training is driven by the script [Models/dual_engine_system.py](Models/dual_engine_system.py).

Run commands from the repository root. On macOS/Linux, you can typically drop the leading `.` (e.g., `python Models/dual_engine_system.py ...`).

Train crop + tabular yield models and save artifacts:

```bash
python .\Models\dual_engine_system.py --retrain --skip-eval
```

Train + run K-Fold evaluation (slower):

```bash
python .\Models\dual_engine_system.py --retrain
```

Also train the optional CNN→LSTM yield model:

```bash
python .\Models\dual_engine_system.py --retrain --skip-eval --train-dl --dl-window 4 --dl-epochs 2
```

Control training size for the large production dataset:

- Default: `--production-max-rows 80000`
- Use full dataset: `--production-max-rows 0`

### Saved artifacts

Artifacts are written to:

`Models/artifacts/dual_engine_system/`

Expected files after training:

- `crop_engine.joblib`
- `yield_engine_tabular.joblib`
- `yield_cnn_lstm.keras` (only if `--train-dl` was used)

---

## Web App (Inference)

The Flask app loads the saved artifacts and serves a single page for inference.

Start the server:

```bash
python .\webapp\app.py
```

Open:

- http://127.0.0.1:5000/

### Inputs (how to use the form)

- **Crop recommendation (Engine 1)**: fill in N/P/K + temperature/humidity/pH/rainfall.
- **Yield prediction (Engine 2 tabular)**: fill in State/District/Crop/Season + Area (+ optional Year).
- **Optional deep yield**: paste exactly `dl_window` past yields (comma or newline separated).

Stop the server: press `Ctrl+C` in the terminal running Flask.

---

## Notes

- [Models/model.ipynb](Models/model.ipynb) is an exploratory notebook; the production training/inference pipeline is the dual-engine script.

---

## Authors

- Vishal Anand
- Aneesh Jain

---

Last updated: April 2026
