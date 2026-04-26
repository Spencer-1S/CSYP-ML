"""Dual-Engine ML system for Precision Agriculture.

Implements the system architecture shown in the provided slides:

Raw Data -> Preprocessing + VIF filtering -> K-Fold CV ->
  (Engine 1) BayesNet / RandomForest -> Crop Recommendation (classification)
  (Engine 2) XGBoost / CNN+LSTM     -> Yield Estimation (regression)
-> Final Dual Predictions

Notes about the provided repository datasets:
- `Datasets/Crop_recommendation.csv` contains only numeric features
  (N, P, K, temperature, humidity, ph, rainfall) and `label`.
  It does NOT include City or Season.

- `Datasets/IndiaAgricultureCropProduction.csv` includes `State`, `District`,
  `Crop`, `Year`, `Season`, `Area`, `Production`, and `Yield`.
  This is suitable for yield modeling (tabular + time-series sequences).

Because City/Season are required inputs in the system spec, this module:
- Accepts City/Season as optional categorical features for the crop engine.
- If they are absent in the crop dataset, it will impute them to "Unknown".

This file is designed to be:
- Professional and readable
- Safe from leakage (pipelines; CV only)
- Easy to extend (clear classes/functions)

Python: 3.10+
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import math
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42


def _featurespec_to_dict(spec: "FeatureSpec") -> dict[str, list[str]]:
    return {
        "crop_numeric": list(spec.crop_numeric),
        "crop_categorical": list(spec.crop_categorical),
        "yield_numeric": list(spec.yield_numeric),
        "yield_categorical": list(spec.yield_categorical),
    }


def _featurespec_from_dict(d: dict[str, Any]) -> "FeatureSpec":
    return FeatureSpec(
        crop_numeric=tuple(d.get("crop_numeric", ())),
        crop_categorical=tuple(d.get("crop_categorical", ())),
        yield_numeric=tuple(d.get("yield_numeric", ())),
        yield_categorical=tuple(d.get("yield_categorical", ())),
    )


def _default_artifacts_dir(root: Path | None = None) -> Path:
    """Default location for persisted models.

    Stored under `Models/artifacts/dual_engine_system/` so it is easy to ignore in git.
    """

    root = root or find_project_root()
    return root / "Models" / "artifacts" / "dual_engine_system"


def _save_joblib(obj: Any, path: Path) -> None:
    try:
        import joblib
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError("Missing dependency: joblib. Install with: pip install joblib") from ex
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def _load_joblib(path: Path) -> Any:
    try:
        import joblib
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError("Missing dependency: joblib. Install with: pip install joblib") from ex
    return joblib.load(path)


@dataclass(frozen=True)
class FeatureSpec:
    """Feature specification for the dual-engine system."""

    # Classification features (crop recommendation)
    crop_numeric: tuple[str, ...] = (
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    )
    crop_categorical: tuple[str, ...] = ("City", "Season")

    # Regression features (yield estimation) - default mapping for the repo dataset
    # You can override this when training on a different yield dataset.
    yield_numeric: tuple[str, ...] = ("Area",)
    yield_categorical: tuple[str, ...] = ("State", "District", "Crop", "Season")


# -----------------------------------------------------------------------------
# Project helpers
# -----------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    """Finds the project root by locating `Datasets/` and `Models/` folders."""

    start = start or Path.cwd()
    for p in [start, *start.parents]:
        if (p / "Datasets").exists() and (p / "Models").exists():
            return p
    return start


def load_crop_recommendation_csv(path: Path) -> pd.DataFrame:
    """Load the crop recommendation dataset."""

    df = pd.read_csv(path)
    # Standardize column casing to match spec if needed.
    # (The repo file already matches.)
    return df


def load_production_csv(path: Path) -> pd.DataFrame:
    """Load the crop production dataset used for yield estimation."""

    # The repo dataset has padded column headers (e.g. 'State      ').
    # Normalize headers + common string columns so training/inference is stable.
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    for col in [
        "State",
        "District",
        "Crop",
        "Year",
        "Season",
        "Area Units",
        "Production Units",
    ]:
        if col in df.columns:
            # Preserve missing values (StringDtype keeps <NA>).
            df[col] = df[col].astype("string").str.strip()

    return df


# -----------------------------------------------------------------------------
# VIF filtering (multicollinearity control)
# -----------------------------------------------------------------------------


def _r2_for_feature(df: pd.DataFrame, target_col: str) -> float:
    """Compute R^2 for predicting one feature from the others.

    VIF is defined as: VIF_i = 1 / (1 - R^2_i)

    Implementation uses a closed-form least squares fit via numpy.
    """

    y = df[target_col].to_numpy(dtype=float)
    X = df.drop(columns=[target_col]).to_numpy(dtype=float)

    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])

    # Solve least squares: beta = argmin ||Xb - y||
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 1.0

    r2 = 1.0 - ss_res / ss_tot
    return float(np.clip(r2, 0.0, 1.0))


def compute_vif(df_numeric: pd.DataFrame) -> pd.Series:
    """Compute VIF for each column in a numeric dataframe.

    - Expects no missing values (impute first)
    - Expects at least 2 columns
    """

    if df_numeric.shape[1] < 2:
        return pd.Series({c: 1.0 for c in df_numeric.columns})

    vifs: dict[str, float] = {}
    for col in df_numeric.columns:
        r2 = _r2_for_feature(df_numeric, col)
        if r2 >= 1.0 - 1e-12:
            vifs[col] = float("inf")
        else:
            vifs[col] = 1.0 / (1.0 - r2)

    return pd.Series(vifs).sort_values(ascending=False)


def vif_filter(
    df_numeric: pd.DataFrame,
    threshold: float = 10.0,
    max_iter: int = 50,
) -> tuple[pd.DataFrame, pd.Series]:
    """Iteratively drop features with VIF > threshold.

    Returns:
    - filtered dataframe
    - final VIF series

    Important: VIF is strictly defined for numeric features. Categorical features
    should be handled by one-hot encoding (and VIF would then apply to the encoded
    columns). In this project we apply VIF to numeric features only.
    """

    filtered = df_numeric.copy()
    for _ in range(max_iter):
        vifs = compute_vif(filtered)
        worst_col = vifs.index[0]
        worst_vif = float(vifs.iloc[0])
        if worst_vif <= threshold or filtered.shape[1] <= 2:
            return filtered, vifs
        filtered = filtered.drop(columns=[worst_col])

    vifs = compute_vif(filtered)
    return filtered, vifs


# -----------------------------------------------------------------------------
# Engine 1: Bayes Net classifier (not Naive Bayes)
# -----------------------------------------------------------------------------


class SimpleDiscreteBayesNetClassifier(BaseEstimator, ClassifierMixin):
    """A lightweight Bayesian Network classifier with explicit feature dependencies.

    Why this exists:
    - A true Bayes Net (Bayesian Network) differs from Naive Bayes because it
      can represent dependencies between features.
    - The spec calls out a realistic dependency: rainfall -> humidity.

    Approach:
    - We discretize continuous variables into quantile bins.
    - We use the following DAG (directed acyclic graph):

        label -> {N,P,K,temperature,rainfall,ph,City,Season}
        humidity -> {label, rainfall, temperature}

      This captures rainfall/temperature influence on humidity while still
      allowing crop label to influence all observed features.

    Inference:
    - For each class label y, we compute:

        log P(y) + sum log P(feature | parents)

      and select the maximum a-posteriori (MAP) label.

    This is a practical, dependency-aware classifier suitable for tabular data.
    """

    def __init__(
        self,
        numeric_features: Iterable[str],
        categorical_features: Iterable[str] = (),
        n_bins: int = 6,
        alpha: float = 1.0,
    ) -> None:
        self.numeric_features = tuple(numeric_features)
        self.categorical_features = tuple(categorical_features)
        self.n_bins = int(n_bins)
        self.alpha = float(alpha)

        # Learned parameters
        self.classes_: np.ndarray | None = None
        self.bin_edges_: dict[str, np.ndarray] = {}

        # CPDs stored as nested dictionaries with Laplace smoothing.
        self.prior_: dict[Any, float] = {}
        self.cpd_label_to_feature_: dict[str, dict[Any, dict[Any, float]]] = {}
        self.cpd_humidity_: dict[tuple[Any, Any, Any], dict[Any, float]] = {}

    def _fit_discretizers(self, X: pd.DataFrame) -> None:
        self.bin_edges_.clear()
        for col in self.numeric_features:
            # Quantile-based bins are robust to skew.
            values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            qs = np.linspace(0.0, 1.0, self.n_bins + 1)
            edges = np.unique(np.quantile(values[~np.isnan(values)], qs))
            if len(edges) < 3:
                # Fallback: min/max edges
                edges = np.unique(np.array([np.nanmin(values), np.nanmax(values)]))
            self.bin_edges_[col] = edges

    def _discretize_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        Xd = X.copy()
        for col in self.numeric_features:
            edges = self.bin_edges_[col]
            # pd.cut yields categories; convert to integer bin index.
            bins = pd.cut(
                pd.to_numeric(Xd[col], errors="coerce"),
                bins=edges,
                include_lowest=True,
                duplicates="drop",
            )
            Xd[col] = bins.cat.codes.astype(int)
        return Xd

    @staticmethod
    def _safe_str_series(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("Unknown")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleDiscreteBayesNetClassifier":
        # Align indices to avoid boolean-indexer alignment issues during slicing.
        X = X.copy().reset_index(drop=True)
        y = y.astype(str).reset_index(drop=True)

        # Ensure categorical features exist
        for col in self.categorical_features:
            if col not in X.columns:
                X[col] = "Unknown"

        # Mean-impute numeric features for stability before discretization.
        for col in self.numeric_features:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].mean())

        self._fit_discretizers(X)
        Xd = self._discretize_numeric(X).reset_index(drop=True)

        # Standardize categorical to strings
        for col in self.categorical_features:
            Xd[col] = self._safe_str_series(Xd[col])

        self.classes_ = np.sort(y.unique())

        # Priors P(label)
        counts = y.value_counts().to_dict()
        total = float(len(y))
        k = float(len(self.classes_))
        self.prior_ = {
            c: (counts.get(c, 0.0) + self.alpha) / (total + self.alpha * k)
            for c in self.classes_
        }

        # CPDs P(feature | label) for most features
        self.cpd_label_to_feature_.clear()
        for feat in [*self.numeric_features, *self.categorical_features]:
            table: dict[Any, dict[Any, float]] = {}
            for cls in self.classes_:
                subset = Xd[y == cls]
                vc = subset[feat].value_counts().to_dict()
                # Determine support (unique values) for smoothing
                support = sorted(Xd[feat].unique().tolist())
                denom = float(len(subset)) + self.alpha * float(len(support))
                table[cls] = {
                    v: (vc.get(v, 0.0) + self.alpha) / denom for v in support
                }
            self.cpd_label_to_feature_[feat] = table

        # CPD for humidity with parents (label, rainfall, temperature)
        # P(humidity | label, rainfall, temperature)
        if "humidity" in self.numeric_features:
            humidity_vals = sorted(Xd["humidity"].unique().tolist())
            self.cpd_humidity_.clear()
            grouped = Xd.assign(label=y).groupby(["label", "rainfall", "temperature"])
            for (cls, rain, temp), group in grouped:
                vc = group["humidity"].value_counts().to_dict()
                denom = float(len(group)) + self.alpha * float(len(humidity_vals))
                self.cpd_humidity_[(cls, rain, temp)] = {
                    h: (vc.get(h, 0.0) + self.alpha) / denom for h in humidity_vals
                }

        return self

    def _log_prob_row(self, row: pd.Series, cls: str) -> float:
        # log P(label)
        lp = math.log(self.prior_.get(cls, 1e-12))

        # For all features except humidity: P(feat | label)
        for feat in [*self.numeric_features, *self.categorical_features]:
            if feat == "humidity":
                continue
            val = row[feat]
            prob = self.cpd_label_to_feature_[feat][cls].get(val)
            if prob is None:
                # Unseen value: uniform-ish smoothing
                prob = 1e-12
            lp += math.log(prob)

        # Humidity dependency: P(humidity | label, rainfall, temperature)
        if "humidity" in self.numeric_features:
            key = (cls, row.get("rainfall"), row.get("temperature"))
            table = self.cpd_humidity_.get(key)
            if table is None:
                # Fallback: use P(humidity | label) if the full parent combo was unseen.
                table = self.cpd_label_to_feature_["humidity"][cls]
            prob = table.get(row.get("humidity"), 1e-12)
            lp += math.log(prob)

        return lp

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted")

        X = X.copy()
        for col in self.categorical_features:
            if col not in X.columns:
                X[col] = "Unknown"

        for col in self.numeric_features:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].mean())

        Xd = self._discretize_numeric(X)
        for col in self.categorical_features:
            Xd[col] = self._safe_str_series(Xd[col])

        logps = np.zeros((len(Xd), len(self.classes_)), dtype=float)
        for i, (_, row) in enumerate(Xd.iterrows()):
            for j, cls in enumerate(self.classes_):
                logps[i, j] = self._log_prob_row(row, str(cls))

        # Log-sum-exp normalization
        maxlp = np.max(logps, axis=1, keepdims=True)
        probs = np.exp(logps - maxlp)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

    def to_state(self) -> dict[str, Any]:
        """Serialize learned parameters to a plain-Python dict.

        This avoids pickling custom classes under `__main__` when the training script
        is run directly, which breaks loading in other entrypoints (e.g. Flask).
        """

        def _py_key(v: Any) -> Any:
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        humidity_table: list[dict[str, Any]] = []
        for (cls, rain, temp), probs in self.cpd_humidity_.items():
            humidity_table.append(
                {
                    "cls": str(cls),
                    "rainfall": _py_key(rain),
                    "temperature": _py_key(temp),
                    "probs": {str(_py_key(k)): float(v) for k, v in probs.items()},
                }
            )

        return {
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "n_bins": int(self.n_bins),
            "alpha": float(self.alpha),
            "classes": [str(c) for c in (self.classes_.tolist() if self.classes_ is not None else [])],
            "bin_edges": {k: [float(x) for x in v.tolist()] for k, v in self.bin_edges_.items()},
            "prior": {str(k): float(v) for k, v in self.prior_.items()},
            "cpd_label_to_feature": {
                feat: {
                    str(cls): {str(_py_key(val)): float(p) for val, p in table.items()}
                    for cls, table in per_class.items()
                }
                for feat, per_class in self.cpd_label_to_feature_.items()
            },
            "cpd_humidity": humidity_table,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SimpleDiscreteBayesNetClassifier":
        def _maybe_int(s: str) -> Any:
            try:
                if s.strip().isdigit() or (s.strip().startswith("-") and s.strip()[1:].isdigit()):
                    return int(s)
                return float(s)
            except Exception:
                return s

        model = cls(
            numeric_features=state.get("numeric_features", ()),
            categorical_features=state.get("categorical_features", ()),
            n_bins=int(state.get("n_bins", 6)),
            alpha=float(state.get("alpha", 1.0)),
        )
        classes = state.get("classes", [])
        model.classes_ = np.array([str(c) for c in classes], dtype=object)
        model.bin_edges_ = {k: np.array(v, dtype=float) for k, v in state.get("bin_edges", {}).items()}
        model.prior_ = {str(k): float(v) for k, v in state.get("prior", {}).items()}

        cpd: dict[str, dict[Any, dict[Any, float]]] = {}
        for feat, per_class in state.get("cpd_label_to_feature", {}).items():
            cpd_feat: dict[Any, dict[Any, float]] = {}
            for cls_label, table in per_class.items():
                cpd_feat[str(cls_label)] = { _maybe_int(str(val)): float(p) for val, p in table.items() }
            cpd[feat] = cpd_feat
        model.cpd_label_to_feature_ = cpd

        model.cpd_humidity_ = {}
        for rec in state.get("cpd_humidity", []):
            cls_label = str(rec.get("cls"))
            rain = _maybe_int(str(rec.get("rainfall")))
            temp = _maybe_int(str(rec.get("temperature")))
            probs = rec.get("probs", {})
            model.cpd_humidity_[(cls_label, rain, temp)] = { _maybe_int(str(k)): float(v) for k, v in probs.items() }

        return model


# -----------------------------------------------------------------------------
# Engine 1 wrapper: BayesNet + RandomForest with Gini vs Entropy tuning
# -----------------------------------------------------------------------------


class CropRecommendationEngine:
    """Dual model crop recommendation engine (Bayes Net + Random Forest)."""

    def __init__(
        self,
        feature_spec: FeatureSpec = FeatureSpec(),
        vif_threshold: float = 10.0,
    ) -> None:
        self.feature_spec = feature_spec
        self.vif_threshold = float(vif_threshold)

        self.kept_numeric_: list[str] = []
        self.vif_report_: pd.Series | None = None

        self.bayesnet_: SimpleDiscreteBayesNetClassifier | None = None
        self.rf_search_: GridSearchCV | None = None

        # Persisted best estimator (pipeline) for inference without GridSearchCV.
        self.rf_best_: Pipeline | None = None

    def _ensure_city_season(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.feature_spec.crop_categorical:
            if col not in df.columns:
                df[col] = "Unknown"
        return df

    def _build_rf_pipeline(self, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
        """Preprocessing pipeline: mean impute + scaling for numeric.

        Even though RandomForest does not *require* scaling, this follows the
        explicit spec (SimpleImputer(mean) + StandardScaler).
        """

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_features),
                ("cat", cat_pipe, categorical_features),
            ],
            remainder="drop",
        )

        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        return Pipeline(steps=[("pre", pre), ("model", rf)])

    @staticmethod
    def _print_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
        acc = accuracy_score(y_true, y_pred)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        print(f"\n{title}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec_m:.4f} (macro) | {prec_w:.4f} (weighted)")
        print(f"Recall:    {rec_m:.4f} (macro) | {rec_w:.4f} (weighted)")
        print(f"F1-score:  {f1_m:.4f} (macro) | {f1_w:.4f} (weighted)")

    def fit(self, df: pd.DataFrame, target_col: str = "label") -> "CropRecommendationEngine":
        df = self._ensure_city_season(df)

        if target_col not in df.columns:
            raise ValueError(f"Missing target column: {target_col}")

        # --- VIF filtering on numeric features ---
        numeric = list(self.feature_spec.crop_numeric)
        Xnum = df[numeric].copy()
        Xnum = Xnum.apply(pd.to_numeric, errors="coerce")
        Xnum = Xnum.fillna(Xnum.mean())

        filtered, vifs = vif_filter(Xnum, threshold=self.vif_threshold)
        self.kept_numeric_ = filtered.columns.tolist()
        self.vif_report_ = vifs

        dropped = [c for c in numeric if c not in self.kept_numeric_]
        print("\nVIF filtering (numeric features):")
        print(f"- Threshold: VIF > {self.vif_threshold:g} => drop")
        print(f"- Kept:   {self.kept_numeric_}")
        print(f"- Dropped:{dropped if dropped else 'None'}")
        # Show the final VIF values after filtering.
        if isinstance(self.vif_report_, pd.Series):
            print("Final VIFs (descending):")
            print(self.vif_report_.round(3).to_string())

        # --- BayesNet model ---
        self.bayesnet_ = SimpleDiscreteBayesNetClassifier(
            numeric_features=self.kept_numeric_,
            categorical_features=self.feature_spec.crop_categorical,
            n_bins=6,
            alpha=1.0,
        )

        X = df[[*self.kept_numeric_, *self.feature_spec.crop_categorical]].copy()
        y = df[target_col].astype(str)
        self.bayesnet_.fit(X, y)

        # --- Random Forest tuning (Gini vs Entropy) ---
        rf_pipe = self._build_rf_pipeline(self.kept_numeric_, list(self.feature_spec.crop_categorical))

        # Mathematical comparison note:
        # - Gini impurity:   G = 1 - sum_k p_k^2
        # - Entropy:         H = - sum_k p_k log(p_k)
        # Both measure node impurity; we tune `criterion` to pick the best
        # split objective under cross-validation.
        param_grid = {
            "model__criterion": ["gini", "entropy"],
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(
            estimator=rf_pipe,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        search.fit(X, y)
        self.rf_search_ = search
        self.rf_best_ = search.best_estimator_

        # Print a clear gini vs entropy comparison from CV results
        res = pd.DataFrame(search.cv_results_)
        g = res[res["param_model__criterion"] == "gini"]["mean_test_score"].max()
        e = res[res["param_model__criterion"] == "entropy"]["mean_test_score"].max()
        print("\nRandomForest criterion comparison (CV macro-F1):")
        print(f"- Best Gini macro-F1:    {g:.4f}")
        print(f"- Best Entropy macro-F1: {e:.4f}")
        print(f"Selected model: {search.best_params_.get('model__criterion')} with macro-F1={search.best_score_:.4f}")

        return self

    def save(self, artifacts_dir: Path | None = None) -> None:
        """Persist the fitted crop engine to disk."""

        if self.bayesnet_ is None or self.rf_best_ is None:
            raise RuntimeError("CropRecommendationEngine is not fitted")

        artifacts_dir = artifacts_dir or _default_artifacts_dir()
        payload = {
            "feature_spec": _featurespec_to_dict(self.feature_spec),
            "vif_threshold": self.vif_threshold,
            "kept_numeric": self.kept_numeric_,
            "vif_report": self.vif_report_,
            "bayesnet_state": self.bayesnet_.to_state(),
            "rf_best": self.rf_best_,
        }
        _save_joblib(payload, artifacts_dir / "crop_engine.joblib")

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "CropRecommendationEngine":
        """Load a previously trained crop engine from disk."""

        artifacts_dir = artifacts_dir or _default_artifacts_dir()
        payload = _load_joblib(artifacts_dir / "crop_engine.joblib")
        feature_spec = _featurespec_from_dict(payload["feature_spec"]) if isinstance(payload.get("feature_spec"), dict) else payload["feature_spec"]
        eng = cls(
            feature_spec=feature_spec,
            vif_threshold=float(payload["vif_threshold"]),
        )
        eng.kept_numeric_ = list(payload["kept_numeric"])
        eng.vif_report_ = payload.get("vif_report")
        if "bayesnet_state" in payload:
            eng.bayesnet_ = SimpleDiscreteBayesNetClassifier.from_state(payload["bayesnet_state"])
        else:
            # Back-compat for older artifacts (may fail if saved under __main__).
            eng.bayesnet_ = payload.get("bayesnet")
        eng.rf_best_ = payload["rf_best"]
        eng.rf_search_ = None
        return eng

    def evaluate_kfold(self, df: pd.DataFrame, target_col: str = "label", n_splits: int = 5) -> None:
        """K-Fold evaluation for BOTH BayesNet and RandomForest."""

        df = self._ensure_city_season(df)
        X = df[[*self.kept_numeric_, *self.feature_spec.crop_categorical]].copy()
        y = df[target_col].astype(str).to_numpy()

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        # RandomForest: cross_val_predict using the tuned estimator
        if self.rf_best_ is None:
            raise RuntimeError("RandomForest model not fitted")

        rf_pred = cross_val_predict(self.rf_best_, X, y, cv=cv, n_jobs=-1)
        self._print_classification_metrics(y, rf_pred, title="Engine 1B: RandomForest (K-Fold CV)")

        # BayesNet: manual CV loop (custom estimator)
        if self.bayesnet_ is None:
            raise RuntimeError("BayesNet model not fitted")

        bayes_preds: list[str] = []
        bayes_true: list[str] = []
        for train_idx, test_idx in cv.split(X, y):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            model = SimpleDiscreteBayesNetClassifier(
                numeric_features=self.kept_numeric_,
                categorical_features=self.feature_spec.crop_categorical,
                n_bins=6,
                alpha=1.0,
            ).fit(Xtr, pd.Series(ytr))
            pred = model.predict(Xte)
            bayes_preds.extend(pred.tolist())
            bayes_true.extend(yte.tolist())

        self._print_classification_metrics(
            np.array(bayes_true), np.array(bayes_preds), title="Engine 1A: BayesNet (K-Fold CV)"
        )

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Predict crop recommendations from a single input row.

        Returns both engines' predictions.
        """

        if self.bayesnet_ is None or self.rf_best_ is None:
            raise RuntimeError("Engine is not fitted")

        X = pd.DataFrame([row])
        X = self._ensure_city_season(X)
        X = X[[*self.kept_numeric_, *self.feature_spec.crop_categorical]]

        bayes_pred = str(self.bayesnet_.predict(X)[0])
        rf_pred = str(self.rf_best_.predict(X)[0])

        return {
            "crop_bayesnet": bayes_pred,
            "crop_random_forest": rf_pred,
        }


# -----------------------------------------------------------------------------
# Engine 2: Yield estimation (XGBoost + CNN+LSTM)
# -----------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class YieldEstimationEngine:
    """Yield estimation engine with tabular (XGBoost) and time-series (CNN+LSTM)."""

    def __init__(
        self,
        feature_spec: FeatureSpec = FeatureSpec(),
        target_col: str = "Yield",
        yield_units: Literal["t/ha", "as_is"] = "t/ha",
    ) -> None:
        self.feature_spec = feature_spec
        self.target_col = target_col
        self.yield_units = yield_units

        self.xgb_model_: Any | None = None
        self.xgb_pipeline_: Pipeline | None = None

        # Fallback statistics for rare/unseen categories where the model can
        # output negative values (we clamp yield to be non-negative).
        self.fallback_stats_: dict[str, Any] | None = None

        # Captures the exact feature lists used to fit the tabular pipeline.
        self.tabular_numeric_features_: list[str] | None = None
        self.tabular_categorical_features_: list[str] | None = None

        self.dl_model_: Any | None = None
        self.dl_history_: Any | None = None
        self.dl_window_: int | None = None
        self.dl_group_cols_: list[str] | None = None

    @staticmethod
    def _standardize_yield_t_per_ha(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Best-effort conversion to tonnes/hectare.

        The production dataset in this repo already has a `Yield` column. Its unit
        appears consistent with `Production Units = Tonnes` and `Area Units = Hectare`.

        We keep rows where those units match; otherwise we drop them to avoid mixing units.
        """

        df = df.copy()
        if "Production Units" in df.columns and "Area Units" in df.columns:
            mask = (
                df["Production Units"].astype(str).str.lower().eq("tonnes")
                & df["Area Units"].astype(str).str.lower().eq("hectare")
            )
            if int(mask.sum()) > 0:
                df = df[mask].copy()
            else:
                warnings.warn(
                    "No rows matched (Production Units=Tonnes, Area Units=Hectare). "
                    "Proceeding without unit filtering.",
                    RuntimeWarning,
                )

        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col])
        return df

    def _build_tabular_preprocess(
        self,
        numeric: list[str],
        categorical: list[str],
        *,
        onehot_sparse_output: bool,
    ) -> ColumnTransformer:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=onehot_sparse_output,
                    ),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric),
                ("cat", cat_pipe, categorical),
            ],
            remainder="drop",
            # When the downstream model requires dense input (e.g. sklearn HGBR),
            # force a dense output regardless of overall matrix density.
            sparse_threshold=0.0 if not onehot_sparse_output else 0.3,
        )

    @staticmethod
    def _print_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        _rmse = rmse(y_true, y_pred)
        print(f"\n{title}")
        print(f"MAE:               {mae:.4f}")
        print(f"Median Abs Error:  {medae:.4f}")
        print(f"RMSE:              {_rmse:.4f}")
        print(f"R^2:               {r2:.4f}")

    def fit_xgboost(self, df: pd.DataFrame) -> "YieldEstimationEngine":
        """Train XGBoost regressor on tabular features."""

        df = self._standardize_yield_t_per_ha(df, self.target_col)

        numeric = list(self.feature_spec.yield_numeric)
        categorical = list(self.feature_spec.yield_categorical)

        # Optional: add a parsed year feature for better signal.
        if "Year" in df.columns and "Year_start" not in df.columns:
            df = df.copy()
            df["Year_start"] = (
                df["Year"].astype(str).str.extract(r"(\d{4})", expand=False)
            )
            df["Year_start"] = pd.to_numeric(df["Year_start"], errors="coerce")
            numeric = [*numeric, "Year_start"]

        y = pd.to_numeric(df[self.target_col], errors="coerce")
        X = df[[*numeric, *categorical]].copy()
        X["Year_start"] = pd.to_numeric(X.get("Year_start"), errors="coerce")

        # Persist fitted feature lists for robust inference.
        self.tabular_numeric_features_ = list(numeric)
        self.tabular_categorical_features_ = list(categorical)

        # XGBoost is optional; if not installed we fall back to sklearn GradientBoosting.
        onehot_sparse_output = True
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=900,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        except Exception as ex:  # noqa: BLE001
            warnings.warn(
                "xgboost is not available; falling back to HistGradientBoostingRegressor. "
                "Install with: pip install xgboost",
                RuntimeWarning,
            )
            from sklearn.ensemble import HistGradientBoostingRegressor

            model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
            # HistGradientBoostingRegressor requires dense input.
            onehot_sparse_output = False

        pre = self._build_tabular_preprocess(
            numeric,
            categorical,
            onehot_sparse_output=onehot_sparse_output,
        )

        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X, y)
        self.xgb_pipeline_ = pipe
        self.xgb_model_ = model

        # Precompute robust fallback medians for rare category combos.
        # These get used only when the regressor predicts a negative yield.
        self.fallback_stats_ = self._compute_fallback_stats(df)
        return self

    def _compute_fallback_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute median yield lookups used when the regressor predicts < 0.

        We intentionally keep these tables small and robust:
        - global median
        - median by (Crop, Season)
        - median by (State, Crop, Season)
        """

        df2 = df.copy()
        df2[self.target_col] = pd.to_numeric(df2[self.target_col], errors="coerce")
        df2 = df2.dropna(subset=[self.target_col])

        def _median_map(group_cols: list[str]) -> dict[str, float]:
            g = df2.groupby(group_cols, dropna=True)[self.target_col].median()
            out: dict[str, float] = {}
            for key, val in g.items():
                if not isinstance(key, tuple):
                    key = (key,)
                out["|".join(map(str, key))] = float(val)
            return out

        return {
            "global_median": float(df2[self.target_col].median()),
            "crop_season": _median_map(["Crop", "Season"]) if {"Crop", "Season"}.issubset(df2.columns) else {},
            "state_crop_season": _median_map(["State", "Crop", "Season"]) if {"State", "Crop", "Season"}.issubset(df2.columns) else {},
        }

    def _fallback_yield(self, row: dict[str, Any]) -> float | None:
        stats = self.fallback_stats_ or {}
        if not stats:
            return None

        state = str(row.get("State", ""))
        crop = str(row.get("Crop", ""))
        season = str(row.get("Season", ""))

        s_map = stats.get("state_crop_season") or {}
        c_map = stats.get("crop_season") or {}

        key_s = "|".join([state, crop, season])
        if key_s in s_map:
            return float(s_map[key_s])

        key_c = "|".join([crop, season])
        if key_c in c_map:
            return float(c_map[key_c])

        if "global_median" in stats:
            return float(stats["global_median"])

        return None

    def save(self, artifacts_dir: Path | None = None) -> None:
        """Persist the fitted yield engine (tabular pipeline, and DL model if present)."""

        if self.xgb_pipeline_ is None:
            raise RuntimeError("YieldEstimationEngine is not fitted")

        artifacts_dir = artifacts_dir or _default_artifacts_dir()

        payload = {
            "feature_spec": _featurespec_to_dict(self.feature_spec),
            "target_col": self.target_col,
            "yield_units": self.yield_units,
            "xgb_pipeline": self.xgb_pipeline_,
            "tabular_numeric_features": self.tabular_numeric_features_,
            "tabular_categorical_features": self.tabular_categorical_features_,
            "fallback_stats": self.fallback_stats_,
            "dl_window": self.dl_window_,
            "dl_group_cols": self.dl_group_cols_,
        }
        _save_joblib(payload, artifacts_dir / "yield_engine_tabular.joblib")

        # Deep learning model is optional; only save if trained and TF is available.
        if self.dl_model_ is not None:
            try:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                self.dl_model_.save(artifacts_dir / "yield_cnn_lstm.keras")
            except Exception:
                # Ignore save failures for optional DL component.
                pass

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "YieldEstimationEngine":
        artifacts_dir = artifacts_dir or _default_artifacts_dir()
        payload = _load_joblib(artifacts_dir / "yield_engine_tabular.joblib")
        feature_spec = _featurespec_from_dict(payload["feature_spec"]) if isinstance(payload.get("feature_spec"), dict) else payload["feature_spec"]
        eng = cls(
            feature_spec=feature_spec,
            target_col=str(payload["target_col"]),
            yield_units=payload.get("yield_units", "t/ha"),
        )
        eng.xgb_pipeline_ = payload["xgb_pipeline"]
        eng.tabular_numeric_features_ = payload.get("tabular_numeric_features")
        eng.tabular_categorical_features_ = payload.get("tabular_categorical_features")
        eng.fallback_stats_ = payload.get("fallback_stats")
        eng.dl_window_ = payload.get("dl_window")
        eng.dl_group_cols_ = payload.get("dl_group_cols")

        # Backward compatibility: older artifacts won't have fallback stats.
        # If repo datasets are present, compute stats once on load.
        if eng.fallback_stats_ is None:
            try:
                root = find_project_root()
                prod_path = root / "Datasets" / "IndiaAgricultureCropProduction.csv"
                if prod_path.exists():
                    df_prod = load_production_csv(prod_path)
                    df_prod = eng._standardize_yield_t_per_ha(df_prod, eng.target_col)
                    eng.fallback_stats_ = eng._compute_fallback_stats(df_prod)
            except Exception:
                eng.fallback_stats_ = None

        # Try to load optional DL model if present.
        dl_path = artifacts_dir / "yield_cnn_lstm.keras"
        if dl_path.exists():
            try:
                import tensorflow as tf

                eng.dl_model_ = tf.keras.models.load_model(dl_path)
            except Exception:
                eng.dl_model_ = None

        return eng

    def evaluate_xgboost_kfold(self, df: pd.DataFrame, n_splits: int = 5) -> None:
        if self.xgb_pipeline_ is None:
            raise RuntimeError("XGBoost pipeline not fitted")

        df = self._standardize_yield_t_per_ha(df, self.target_col)

        numeric = list(self.feature_spec.yield_numeric)
        categorical = list(self.feature_spec.yield_categorical)

        if "Year" in df.columns and "Year_start" not in df.columns:
            df = df.copy()
            df["Year_start"] = (
                df["Year"].astype(str).str.extract(r"(\d{4})", expand=False)
            )
            df["Year_start"] = pd.to_numeric(df["Year_start"], errors="coerce")
            numeric = [*numeric, "Year_start"]

        y = pd.to_numeric(df[self.target_col], errors="coerce").to_numpy(dtype=float)
        X = df[[*numeric, *categorical]].copy()

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        pred = cross_val_predict(self.xgb_pipeline_, X, y, cv=cv, n_jobs=-1)
        self._print_regression_metrics(y, pred, title="Engine 2A: Tabular Regressor (K-Fold CV)")

    # --- Deep learning integration (CNN + LSTM) ---

    @staticmethod
    def _build_sequences(
        df: pd.DataFrame,
        group_cols: list[str],
        time_col: str,
        value_col: str,
        window: int = 6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert grouped time series into sliding-window sequences."""

        Xs: list[np.ndarray] = []
        ys: list[float] = []

        for _, g in df.groupby(group_cols):
            g = g.sort_values(time_col)
            values = g[value_col].to_numpy(dtype=float)
            if len(values) <= window:
                continue
            for i in range(window, len(values)):
                Xs.append(values[i - window : i])
                ys.append(values[i])

        if not Xs:
            raise ValueError(
                "Not enough time-series data to build sequences. "
                "Try lowering `window` or using broader grouping keys."
            )

        Xarr = np.stack(Xs, axis=0)
        yarr = np.array(ys, dtype=float)
        return Xarr, yarr

    @staticmethod
    def build_cnn_lstm_model(input_length: int) -> Any:
        """CNN + LSTM sequential model using ADAM optimizer."""

        try:
            import tensorflow as tf
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError(
                "TensorFlow is required for the deep learning yield model. "
                "Install with: pip install tensorflow"
            ) from ex

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_length, 1)),
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(1),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mae",
            metrics=["mae"],
        )
        return model

    def fit_cnn_lstm(
        self,
        df: pd.DataFrame,
        window: int = 6,
        epochs: int = 20,
        batch_size: int = 64,
        group_cols: list[str] | None = None,
    ) -> "YieldEstimationEngine":
        """Train the CNN+LSTM model using time-series of yield.

        Default grouping uses (State, Crop) which creates one yield time-series
        per (State, Crop). This is intentionally broad to ensure enough history
        exists for sequence construction on the provided production dataset.
        """

        df = self._standardize_yield_t_per_ha(df, self.target_col)

        df = df.copy()
        df["Year_start"] = df["Year"].astype(str).str.extract(r"(\d{4})", expand=False)
        df["Year_start"] = pd.to_numeric(df["Year_start"], errors="coerce")
        df = df.dropna(subset=["Year_start", self.target_col])
        df["Year_start"] = df["Year_start"].astype(int)

        group_cols = group_cols or ["State", "Crop"]

        Xseq, yseq = self._build_sequences(
            df=df,
            group_cols=group_cols,
            time_col="Year_start",
            value_col=self.target_col,
            window=window,
        )

        # Reshape to [samples, timesteps, channels]
        Xseq = Xseq[..., np.newaxis]

        self.dl_model_ = self.build_cnn_lstm_model(input_length=window)
        self.dl_history_ = self.dl_model_.fit(
            Xseq,
            yseq,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
        self.dl_window_ = int(window)
        self.dl_group_cols_ = list(group_cols)
        return self

    def predict_yield_deep_from_history(self, past_yields: Iterable[float]) -> float:
        """Predict next yield from a fixed-length yield history sequence.

        `past_yields` must have length equal to the trained `dl_window_`.
        """

        if self.dl_model_ is None:
            raise RuntimeError("Deep learning model is not trained/loaded")
        if self.dl_window_ is None:
            raise RuntimeError("Deep learning window metadata is missing")

        seq = np.array(list(past_yields), dtype=float)
        if seq.shape[0] != self.dl_window_:
            raise ValueError(f"Expected {self.dl_window_} past yields, got {seq.shape[0]}")

        X = seq.reshape(1, self.dl_window_, 1)
        y_pred = float(self.dl_model_.predict(X, verbose=0)[0][0])
        return max(0.0, y_pred)

    def predict_yield(self, row: dict[str, Any]) -> dict[str, Any]:
        """Predict yield from tabular features. Deep model requires sequences."""

        if self.xgb_pipeline_ is None:
            raise RuntimeError("XGBoost pipeline not fitted")

        if not self.tabular_numeric_features_ or not self.tabular_categorical_features_:
            raise RuntimeError("Yield engine is missing fitted feature metadata")

        # Allow users to provide either `Year_start` (int) or `Year` (e.g., '2001-02').
        row = dict(row)
        if "Year_start" in self.tabular_numeric_features_ and "Year_start" not in row:
            if "Year" in row and row["Year"] is not None:
                year_str = str(row["Year"])
                try:
                    row["Year_start"] = int(pd.Series([year_str]).str.extract(r"(\d{4})")[0].iloc[0])
                except Exception:  # noqa: BLE001
                    row["Year_start"] = np.nan
            else:
                row["Year_start"] = np.nan

        # Ensure all expected columns exist so the ColumnTransformer can select them.
        for c in self.tabular_numeric_features_:
            row.setdefault(c, np.nan)
        for c in self.tabular_categorical_features_:
            row.setdefault(c, "Unknown")

        X = pd.DataFrame([row])
        raw_pred = float(self.xgb_pipeline_.predict(X)[0])

        # Physical constraint: yield in tonnes/hectare cannot be negative.
        # When the model predicts < 0 (often due to rare category combos),
        # fall back to robust medians computed from the training dataset.
        if raw_pred < 0:
            fb = self._fallback_yield(row)
            y_pred = float(fb) if fb is not None else 0.0
        else:
            y_pred = raw_pred
        return {
            "yield_tabular": y_pred,
            "yield_units": self.yield_units,
        }


# -----------------------------------------------------------------------------
# Dual-engine orchestrator
# -----------------------------------------------------------------------------


class DualEngineSystem:
    """Orchestrates crop recommendation + yield estimation."""

    def __init__(self, feature_spec: FeatureSpec = FeatureSpec()) -> None:
        self.feature_spec = feature_spec
        self.crop_engine = CropRecommendationEngine(feature_spec=feature_spec)
        self.yield_engine = YieldEstimationEngine(feature_spec=feature_spec)

    def save(self, artifacts_dir: Path | None = None) -> None:
        artifacts_dir = artifacts_dir or _default_artifacts_dir()
        self.crop_engine.save(artifacts_dir)
        self.yield_engine.save(artifacts_dir)

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "DualEngineSystem":
        artifacts_dir = artifacts_dir or _default_artifacts_dir()
        crop = CropRecommendationEngine.load(artifacts_dir)
        yld = YieldEstimationEngine.load(artifacts_dir)
        sys = cls(feature_spec=crop.feature_spec)
        sys.crop_engine = crop
        sys.yield_engine = yld
        return sys

    def fit_from_repo_datasets(
        self,
        root: Path | None = None,
        *,
        production_max_rows: int | None = None,
        train_dl: bool = False,
        dl_window: int = 6,
        dl_epochs: int = 10,
    ) -> "DualEngineSystem":
        """Train engines using datasets in this repository."""

        root = root or find_project_root()

        crop_path = root / "Datasets" / "Crop_recommendation.csv"
        prod_path = root / "Datasets" / "IndiaAgricultureCropProduction.csv"

        df_crop = load_crop_recommendation_csv(crop_path)
        df_prod = load_production_csv(prod_path)
        if production_max_rows is not None and len(df_prod) > production_max_rows:
            if train_dl:
                # For sequence models, avoid random sampling which destroys time continuity.
                tmp = df_prod.copy()
                target_col = self.yield_engine.target_col

                tmp["Year_start"] = tmp["Year"].astype(str).str.extract(r"(\d{4})", expand=False)
                tmp["Year_start"] = pd.to_numeric(tmp["Year_start"], errors="coerce")
                tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")

                tmp = tmp.dropna(subset=["Year_start", target_col]).sort_values("Year_start")
                df_prod = tmp.tail(production_max_rows)
            else:
                df_prod = df_prod.sample(n=production_max_rows, random_state=RANDOM_STATE)

        print("\nFitting crop recommendation engine...")
        self.crop_engine.fit(df_crop, target_col="label")

        print("\nFitting yield estimation engine (tabular)...")
        self.yield_engine.fit_xgboost(df_prod)

        if train_dl:
            print("\nFitting yield estimation engine (CNN+LSTM)...")
            try:
                self.yield_engine.fit_cnn_lstm(df_prod, window=dl_window, epochs=dl_epochs)
            except ValueError as ex:
                # If sequences are insufficient, retry with a smaller window.
                if "Not enough time-series data" in str(ex) and dl_window > 3:
                    warnings.warn(
                        "Not enough sequences for the requested DL window; retrying with window=3.",
                        RuntimeWarning,
                    )
                    self.yield_engine.fit_cnn_lstm(df_prod, window=3, epochs=dl_epochs)
                else:
                    raise

        return self

    def evaluate_from_repo_datasets(
        self,
        root: Path | None = None,
        *,
        production_max_rows: int | None = None,
    ) -> None:
        root = root or find_project_root()

        df_crop = load_crop_recommendation_csv(root / "Datasets" / "Crop_recommendation.csv")
        df_prod = load_production_csv(root / "Datasets" / "IndiaAgricultureCropProduction.csv")
        if production_max_rows is not None and len(df_prod) > production_max_rows:
            df_prod = df_prod.sample(n=production_max_rows, random_state=RANDOM_STATE)

        print("\n=== K-Fold evaluation: Crop recommendation ===")
        self.crop_engine.evaluate_kfold(df_crop, target_col="label")

        print("\n=== K-Fold evaluation: Yield estimation (tabular) ===")
        self.yield_engine.evaluate_xgboost_kfold(df_prod)

    def predict_dual(self, crop_row: dict[str, Any], yield_row: dict[str, Any]) -> dict[str, Any]:
        """Final dual predictions: crop label(s) + yield estimate."""

        crop_pred = self.crop_engine.predict(crop_row)
        yield_pred = self.yield_engine.predict_yield(yield_row)

        return {
            **crop_pred,
            **yield_pred,
        }


# -----------------------------------------------------------------------------
# Demo entrypoint
# -----------------------------------------------------------------------------


def _demo() -> None:
    """Trains and evaluates on repo datasets, then runs one example prediction."""

    import argparse

    parser = argparse.ArgumentParser(description="Dual-engine crop + yield ML system")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if saved models exist.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip K-Fold evaluation (faster start-up).",
    )
    parser.add_argument(
        "--train-dl",
        action="store_true",
        help="Also train the optional CNN+LSTM yield model (requires tensorflow).",
    )
    parser.add_argument(
        "--dl-window",
        type=int,
        default=6,
        help="Window size for the CNN+LSTM yield model (timesteps).",
    )
    parser.add_argument(
        "--dl-epochs",
        type=int,
        default=10,
        help="Epochs for the CNN+LSTM yield model.",
    )
    parser.add_argument(
        "--production-max-rows",
        type=int,
        default=80_000,
        help="Sample size for the large production dataset (set to 0 for full).",
    )
    args = parser.parse_args()

    artifacts_dir = _default_artifacts_dir()
    crop_art = artifacts_dir / "crop_engine.joblib"
    yld_art = artifacts_dir / "yield_engine_tabular.joblib"

    production_max_rows = None if args.production_max_rows == 0 else args.production_max_rows

    if (not args.retrain) and crop_art.exists() and yld_art.exists():
        print(f"\nLoading saved models from: {artifacts_dir}")
        system = DualEngineSystem.load(artifacts_dir)
    else:
        print(f"\nTraining models (will be saved to: {artifacts_dir})")
        system = DualEngineSystem().fit_from_repo_datasets(
            production_max_rows=production_max_rows,
            train_dl=bool(args.train_dl),
            dl_window=int(args.dl_window),
            dl_epochs=int(args.dl_epochs),
        )
        system.save(artifacts_dir)

    if not args.skip_eval:
        system.evaluate_from_repo_datasets(production_max_rows=production_max_rows)

    # Example input row for the crop recommender
    crop_row = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 203.0,
        "City": "Unknown",
        "Season": "Unknown",
    }

    # Example row for yield estimator: uses production schema by default
    yield_row = {
        "State": "Andaman and Nicobar Islands",
        "District": "NICOBARS",
        "Crop": "Arecanut",
        "Season": "Kharif",
        "Area": 1254.0,
        "Year_start": 2001,
    }

    pred = system.predict_dual(crop_row=crop_row, yield_row=yield_row)
    print("\nExample dual prediction:")
    for k, v in pred.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _demo()
