# Research Project Description — Dual-Engine Crop Recommendation and Yield Prediction

## Abstract
Precision agriculture increasingly relies on machine learning (ML) to support data-driven decisions such as **crop selection** and **yield estimation**. This project presents an end-to-end, reproducible **dual-engine ML system** that addresses two complementary tasks: (i) **crop recommendation** (multi-class classification) using soil and weather attributes and (ii) **yield prediction** (regression) using historical agricultural production records. The system includes training pipelines, cross-validation evaluation, artifact persistence for reliable inference, a lightweight Flask web interface for real-time predictions, and an automated reporting workflow that produces metrics and visualizations.

The crop engine combines (a) a dependency-aware **Bayesian Network classifier** that discretizes continuous agronomic variables and (b) a tuned **Random Forest** model trained using scikit-learn pipelines and cross-validation. The yield engine provides a robust tabular regression pipeline using **XGBoost** when available (with a scikit-learn fallback), and an optional **CNN→LSTM** deep learning model that forecasts next yield from a fixed window of past yields. A practical safeguard is included for negative yield outputs: predictions below zero trigger a median-based fallback derived from training data. Together, these components form a deployable pipeline suitable for a research prototype and extensible for further study.

---

## 1. Problem Statement and Motivation
Agricultural decision-making is affected by multiple interacting variables: soil nutrients, climate conditions, seasonal patterns, and regional differences in cultivation practices. Farmers and planners require tools that:

- Recommend crops likely to perform well under given soil and environmental conditions.
- Estimate expected yield to support planning for inputs, storage, supply chains, and risk.

This work targets both decisions through a single project that unifies data ingestion, preprocessing, model training, evaluation, and deployment.

---

## 2. Project Scope and Objectives

### 2.1 Objectives
1. Build a **crop recommendation model** using soil and weather features.
2. Build a **yield prediction model** using historical production records.
3. Provide **reproducible training** with saved artifacts for inference.
4. Deliver a **simple web UI** for interactive prediction.
5. Generate a **metrics + visualization report** to support analysis and paper writing.

### 2.2 Outputs
- Trained model artifacts stored under `Models/artifacts/dual_engine_system/`.
- A Flask inference application under `webapp/`.
- A report generator producing JSON metrics and PNG plots under `reports/`.

---

## 3. System Architecture (Dual-Engine)
The implemented pipeline follows the conceptual flow:

**Raw Data → Preprocessing → Feature Handling → Model Training → Evaluation → Persistence → Inference/UI**

Two engines are trained and used in inference:

1. **Engine 1 (Crop Recommendation / Classification)**
   - Bayesian Network classifier (discrete, dependency-aware)
   - RandomForest classifier (tuned)

2. **Engine 2 (Yield Prediction / Regression)**
   - Tabular regressor (XGBoost if available; sklearn fallback)
   - Optional deep regressor (CNN→LSTM) for sequence-based forecasting

The primary implementation is in `Models/dual_engine_system.py`.

---

## 4. Datasets

### 4.1 Crop Recommendation Dataset
Source file: `Datasets/Crop_recommendation.csv`

- Target: `label` (crop name)
- Numeric features used by the system:
  - `N`, `P`, `K` (soil nutrients)
  - `temperature`, `humidity`, `ph`, `rainfall`

Important note (design alignment): the dataset does **not** contain `City` and `Season`. The implementation supports these as optional categorical inputs for completeness against the project specification; when absent they are imputed as `"Unknown"`.

### 4.2 Production Dataset for Yield
Source file: `Datasets/IndiaAgricultureCropProduction.csv`

- Typical columns: `State`, `District`, `Crop`, `Year`, `Season`, `Area`, `Production`, `Yield`.
- The yield engine uses a tabular representation; and optionally a time-series representation via sliding windows.

### 4.3 Rainfall Dataset (Extension)
Source file: `Datasets/DistrictWiseRainfallNormal.csv`

This dataset is included for extension/exploration. The current training pipeline does not directly integrate it into the shipped models, but it can be used to engineer additional climate features.

---

## 5. Data Preprocessing and Quality Handling

### 5.1 Header Normalization and String Cleanup
Real-world CSVs often contain inconsistent formatting. The production dataset can include padded column headers (e.g., trailing spaces). The loader normalizes headers by stripping whitespace and trims key categorical columns (`State`, `District`, `Crop`, `Year`, `Season`, unit columns). This stabilizes downstream column selection and one-hot encoding.

### 5.2 Missing Values
- Numeric features: imputed via `SimpleImputer(strategy="mean")`.
- Categorical features: imputed via `SimpleImputer(strategy="most_frequent")`.

### 5.3 Unit Consistency (Yield)
The yield pipeline performs a best-effort standardization step and can filter rows to align with expected units:

- `Production Units = Tonnes`
- `Area Units = Hectare`

This reduces unit-mixing risk when estimating yield in `t/ha`.

### 5.4 Feature Specifications
A `FeatureSpec` defines the canonical input feature sets:

- Crop engine: numeric soil/weather + optional categorical (`City`, `Season`)
- Yield engine: numeric (`Area` and optionally derived `Year_start`) + categorical (`State`, `District`, `Crop`, `Season`)

### 5.5 Multicollinearity Control (Crop Engine)
To reduce redundant numeric predictors, the crop engine includes **Variance Inflation Factor (VIF)** filtering applied to numeric features.

- For each numeric feature $x_i$, regress it on other features to compute $R^2_i$.
- Compute $\text{VIF}_i = \frac{1}{1 - R^2_i}$.
- Iteratively drop the feature with highest VIF until VIF ≤ threshold (default 10) or a small set remains.

This is a pragmatic step to improve model stability and interpretability.

---

## 6. Engine 1 — Crop Recommendation (Classification)

### 6.1 Bayesian Network Classifier (Discrete)
The project implements a lightweight, dependency-aware Bayesian Network classifier (`SimpleDiscreteBayesNetClassifier`).

Key design choices:

- **Discretization**: continuous variables are binned using quantile edges. This is robust to skew and makes conditional probability tables feasible.
- **Explicit dependency modeling**: unlike Naive Bayes (which assumes conditional independence), the classifier models a dependency where humidity depends on rainfall and temperature in addition to crop label.
- **Smoothing**: Laplace/Dirichlet-style smoothing via an `alpha` parameter avoids zero probabilities.
- **Inference**: predicts the crop label using maximum a-posteriori (MAP) estimation by summing log-probabilities.

This model is included both as an interpretable baseline and as a method aligned with domain intuition (humidity is not independent of rainfall/temperature).

### 6.2 RandomForest Classifier (Tuned)
A RandomForest pipeline is trained using scikit-learn’s `Pipeline` and `ColumnTransformer`.

- Numeric preprocessing: mean-impute → standardize.
- Categorical preprocessing: most-frequent impute → one-hot encode.

Hyperparameter tuning:

- Grid search across number of trees, depth, and sample split/leaf parameters.
- Explicit comparison between split criteria:
  - Gini impurity: $G = 1 - \sum_k p_k^2$
  - Entropy: $H = -\sum_k p_k \log(p_k)$

Selection criterion uses macro-F1 under Stratified K-Fold CV.

### 6.3 Classification Metrics
The evaluation reports:

- Accuracy
- Precision / Recall / F1 (macro and weighted)
- Confusion matrix
- Per-class metrics

Automated charts:

- Confusion matrix heatmap
- F1-score by class

---

## 7. Engine 2 — Yield Prediction (Regression)

### 7.1 Tabular Regression Model
The default yield model is a tabular regressor trained with preprocessing and one-hot encoding.

- Categorical features: `State`, `District`, `Crop`, `Season`
- Numeric features:
  - `Area`
  - optional derived `Year_start` (parsed from `Year` like `2001-02` → `2001`)

Regressor:

- Primary: `XGBRegressor` (XGBoost) if installed.
- Fallback: scikit-learn `HistGradientBoostingRegressor` if XGBoost is unavailable.

### 7.2 Negative Yield Safeguard (Fallback Statistics)
In practice, tree/boosting regressors can output negative values for rare or unseen category combinations. Since yield in `t/ha` is physically non-negative, the inference path includes a rule:

- If the model predicts `raw_pred < 0`, return a robust median fallback computed from training data.

Fallback tables include:

1. Global median yield
2. Median by `(Crop, Season)`
3. Median by `(State, Crop, Season)`

This reduces user-facing failures and improves realism when extrapolation would otherwise produce invalid values.

### 7.3 Optional Deep Learning Model (CNN→LSTM)
An optional sequence-based model forecasts the next yield value from a fixed-size history window.

Sequence construction:

- Parse `Year_start` and sort within groups.
- Grouping keys (default): `(State, Crop)`.
- For each group, create sliding windows of length `window`:
  - input: past yields $[y_{t-window}, ..., y_{t-1}]$
  - target: $y_t$

Neural network architecture:

- `Conv1D` → `MaxPooling1D` → `LSTM` → `Dense(1)`
- Optimizer: Adam
- Loss: MAE

The deep model is saved to `yield_cnn_lstm.keras` when trained.

Important evaluation note: if the deep model is trained earlier on sequences derived from the same dataset, then a random train/test split on the same derived sequence set is not a strict temporal holdout. For research-grade claims, a time-aware split (e.g., training on years ≤ Y and testing on later years) is recommended.

---

## 8. Evaluation Protocol

### 8.1 Cross-Validation
- Crop classification: Stratified K-Fold CV
- Yield regression (tabular): K-Fold CV

### 8.2 Metrics Used
- **Classification**: Accuracy, macro/weighted Precision-Recall-F1, confusion matrix.
- **Regression**: MAE, Median Absolute Error, RMSE, $R^2$.

### 8.3 Generated Reports and Visualizations
The repository includes a report generator: `Models/generate_model_reports.py`.

It creates run folders under `reports/run_<timestamp>/` containing:

- Crop engine:
  - JSON metrics for BayesNet and RandomForest
  - Confusion matrix heatmaps
  - F1-score by class
- Yield (tabular):
  - metrics JSON
  - actual vs predicted scatter plot
  - residual distribution plot
  - MAE by crop plot
- Yield (deep, optional):
  - metrics JSON and plots if the deep model artifact exists

---

## 9. Model Persistence and Reproducibility

### 9.1 Artifact Strategy
To support deployment and reproducible inference:

- scikit-learn pipelines and fitted objects are persisted using `joblib`.
- custom Bayesian Network components are stored as a **plain state dictionary** (rather than pickling a `__main__` class), improving portability across scripts and environments.
- deep learning model (if trained) is saved as a `.keras` artifact.

Artifacts location:

- `Models/artifacts/dual_engine_system/`

### 9.2 Reproducible Runs
- Random state is fixed (e.g., `RANDOM_STATE = 42`) for model components.
- The report generator stores a `run_info.json` capturing:
  - Python version, OS, dataset row counts used
  - CV configuration
  - whether the deep model was enabled

---

## 10. Deployment: Flask Web Application

A minimal Flask app provides an interactive UI for inference.

- Route `GET /`: renders a single-page form with defaults.
- Route `POST /predict`: validates inputs and returns predictions.

Inputs:

- Crop engine: N/P/K + temperature/humidity/pH/rainfall (+ optional City/Season).
- Yield engine (tabular): State/District/Crop/Season + Area (+ optional Year/Year_start).
- Yield engine (deep): a pasted history of exactly `dl_window` yields.

This interface demonstrates practical end-user interaction and verifies that saved artifacts can be loaded and used consistently.

---

## 11. Limitations
This project is designed as a research prototype; limitations include:

1. **Dataset constraints**: crop recommendation dataset is clean and may not reflect field noise; production dataset can have reporting inconsistencies.
2. **Feature coverage**: yield modeling uses limited agronomic features (primarily area + categorical region/crop/season); weather and soil factors are not yet fused into the yield model.
3. **Generalization**: regional differences and unseen category combinations can challenge yield estimation.
4. **Deep model evaluation**: sequence-based evaluation should prefer temporal holdout to avoid overly optimistic estimates.
5. **Interpretability**: RandomForest and XGBoost feature effects can be analyzed further (e.g., SHAP, partial dependence) if needed for the paper.

---

## 12. Future Work (Research Extensions)
Recommended directions for a research paper extension:

- Integrate rainfall normals and seasonal climate signals into yield prediction.
- Engineer additional numeric features from production data (log area, interactions, rolling averages).
- Use time-aware validation (walk-forward / blocked CV) for yield forecasting.
- Add explainability (SHAP for tree models; feature importance stability across folds).
- Compare against baselines (linear regression, ridge/lasso, CatBoost/LightGBM, naive seasonal averages).
- Incorporate geospatial encoding (district/state embeddings, clustering) or external climate datasets.
- Add monitoring and model drift detection in deployment.

---

## 13. Reproduction Guide (Commands)

Train and save artifacts:

```bash
python Models/dual_engine_system.py --retrain --skip-eval
```

Optionally train the deep yield model:

```bash
python Models/dual_engine_system.py --retrain --skip-eval --train-dl --dl-window 4 --dl-epochs 2
```

Run the web app:

```bash
python webapp/app.py
```

Generate metrics and charts:

```bash
python Models/generate_model_reports.py
```
