# Dual-Engine Crop Recommendation and Yield Prediction

A minimal, end-to-end ML project for precision agriculture:

- **Crop recommendation (classification)** from soil + weather inputs.
- **Yield prediction (regression)** from agricultural production data (tabular model), with an optional CNN→LSTM model that forecasts next yield from past yield history.

Core implementation: [Models/dual_engine_system.py](Models/dual_engine_system.py). A simple inference UI is provided via Flask in `webapp/`.

---

## Quickstart

From the repository root:

```bash
pip install pandas numpy scikit-learn joblib flask
python Models/dual_engine_system.py --retrain --skip-eval
python webapp/app.py
```

Open: http://127.0.0.1:5000/

Stop the server: `Ctrl+C`.

---

## Models (High level)

- **Crop engine**: Bayesian Network classifier + tuned RandomForestClassifier.
- **Yield engine**: tabular regressor (XGBoost if installed; otherwise HistGradientBoostingRegressor) + optional CNN→LSTM (`--train-dl`).

Crop numeric inputs: `N, P, K, temperature, humidity, ph, rainfall`.

---

## Datasets

Used by the training script:

- [Datasets/Crop_recommendation.csv](Datasets/Crop_recommendation.csv)
- [Datasets/IndiaAgricultureCropProduction.csv](Datasets/IndiaAgricultureCropProduction.csv)

Included for extension/exploration:

- [Datasets/DistrictWiseRainfallNormal.csv](Datasets/DistrictWiseRainfallNormal.csv)

---

## Setup

### Prerequisites

- Python **3.10+**

### Optional dependencies

- Better tabular yield performance: `pip install xgboost`
- Enable deep yield model (CNN→LSTM): `pip install tensorflow`

---

## Training (save/load artifacts)

Train crop + tabular yield and save artifacts:

```bash
python Models/dual_engine_system.py --retrain --skip-eval
```

Run K-Fold evaluation (slower):

```bash
python Models/dual_engine_system.py --retrain
```

Also train the optional CNN→LSTM model:

```bash
python Models/dual_engine_system.py --retrain --skip-eval --train-dl --dl-window 4 --dl-epochs 2
```

Production dataset size control:

- Default: `--production-max-rows 80000`
- Full dataset: `--production-max-rows 0`

---

## Artifacts

Artifacts are written to `Models/artifacts/dual_engine_system/`.

- `crop_engine.joblib`
- `yield_engine_tabular.joblib`
- `yield_cnn_lstm.keras` (only if trained with `--train-dl`)

---

## Web App (Inference)

Start:

```bash
python webapp/app.py
```

Inputs:

- **Crop**: N/P/K + temperature/humidity/pH/rainfall.
- **Yield (tabular)**: State, District, Crop, Season, Area (Year is optional).
- **Yield (CNN→LSTM, optional)**: paste exactly `dl_window` past yields (comma or newline separated).

---

## Project Structure

```
CSYP-ML/
  Datasets/
  Models/
    dual_engine_system.py
    artifacts/
      dual_engine_system/
  webapp/
    app.py
    templates/
      index.html
  README.md
```

---

## Notes

- [Models/data_preprocessing_and_analysis.ipynb](Models/data_preprocessing_and_analysis.ipynb) is exploratory; the supported training/inference path is the dual-engine script + Flask app.

---

## Authors

- Vishal Anand
- Aneesh Jain

---

Last updated: April 2026
