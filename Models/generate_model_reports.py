from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.dual_engine_system import (  # noqa: E402
    DualEngineSystem,
    SimpleDiscreteBayesNetClassifier,
    _default_artifacts_dir,
    find_project_root,
    load_crop_recommendation_csv,
    load_production_csv,
)


@dataclass
class RunInfo:
    created_at: str
    python: str
    platform: str
    artifacts_dir: str
    crop_rows: int
    yield_rows_used: int
    crop_cv_splits: int
    yield_cv_splits: int
    deep_enabled: bool


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = max(2, len(labels))
    size = min(18, max(8, int(0.35 * n) + 4))
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix_normalized(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
    normalize: str = "true",
) -> None:
    """Plot a normalized confusion matrix.

    normalize:
      - 'true'  : rows sum to 1 (recall per class)
      - 'pred'  : columns sum to 1
      - 'all'   : global normalization
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm, dtype=float)
    denom: np.ndarray
    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
    else:
        denom = np.array([[cm.sum()]])
    denom = np.where(denom == 0, 1.0, denom)
    cmn = cm / denom

    n = max(2, len(labels))
    size = min(18, max(8, int(0.35 * n) + 4))
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    sns.heatmap(
        cmn,
        annot=False,
        cmap="Blues",
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_confusion_errors(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Plot only off-diagonal errors to avoid 'all-diagonal' charts."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm, dtype=float)
    cm_err = cm.copy()
    np.fill_diagonal(cm_err, 0.0)

    # If there are no errors, still write a plot for consistency.
    vmax = float(np.max(cm_err)) if float(np.max(cm_err)) > 0 else 1.0

    n = max(2, len(labels))
    size = min(18, max(8, int(0.35 * n) + 4))
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    sns.heatmap(
        cm_err,
        annot=False,
        cmap="Reds",
        cbar=True,
        vmin=0.0,
        vmax=vmax,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_f1_by_class(per_class: dict[str, dict[str, float]], title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = list(per_class.keys())
    f1 = [per_class[l]["f1"] for l in labels]

    fig = plt.figure(figsize=(10, max(4, int(0.35 * len(labels)))))
    ax = fig.add_subplot(111)
    order = np.argsort(f1)
    labels_sorted = [labels[i] for i in order]
    f1_sorted = [f1[i] for i in order]

    ax.barh(labels_sorted, f1_sorted)
    ax.set_title(title)
    ax.set_xlabel("F1-score")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)

    labels = sorted(list({*y_true.tolist(), *y_pred.tolist()}))
    acc = float(accuracy_score(y_true, y_pred))

    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    p_c, r_c, f1_c, s_c = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class: dict[str, dict[str, float]] = {}
    for i, lab in enumerate(labels):
        per_class[str(lab)] = {
            "precision": float(p_c[i]),
            "recall": float(r_c[i]),
            "f1": float(f1_c[i]),
            "support": float(s_c[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "labels": labels,
        "accuracy": acc,
        "macro": {"precision": float(p_m), "recall": float(r_m), "f1": float(f1_m)},
        "weighted": {"precision": float(p_w), "recall": float(r_w), "f1": float(f1_w)},
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _plot_scatter_true_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=10, alpha=0.35)

    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([mn, mx], [mn, mx], color="black", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_scatter_true_pred_zoomed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
    lo: float = 1.0,
    hi: float = 99.0,
) -> None:
    """Zoomed scatter plot using percentile-based axis limits.

    This prevents extreme outliers from flattening the plot and makes the
    typical-range relationship visible.
    """

    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    both = np.concatenate([y_true, y_pred])
    both = both[np.isfinite(both)]
    if both.size == 0:
        return

    q_lo, q_hi = np.percentile(both, [lo, hi])
    if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
        return

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=10, alpha=0.35)
    ax.plot([q_lo, q_hi], [q_lo, q_hi], color="black", linewidth=1)
    ax.set_xlim(float(q_lo), float(q_hi))
    ax.set_ylim(float(q_lo), float(q_hi))
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    sns.histplot(resid, bins=50, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Residual (pred - actual)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_residuals_zoomed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
    lo: float = 1.0,
    hi: float = 99.0,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return

    q_lo, q_hi = np.percentile(resid, [lo, hi])
    if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
        return

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    sns.histplot(resid, bins=60, kde=True, ax=ax)
    ax.set_xlim(float(q_lo), float(q_hi))
    ax.set_title(title)
    ax.set_xlabel("Residual (pred - actual)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _distribution_summary(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    q = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "min": float(q[0]),
        "p01": float(q[1]),
        "p05": float(q[2]),
        "p25": float(q[3]),
        "p50": float(q[4]),
        "p75": float(q[5]),
        "p95": float(q[6]),
        "p99": float(q[7]),
        "max": float(q[8]),
    }


def _plot_group_mae(df: pd.DataFrame, group_col: str, err_col: str, title: str, out_path: Path, top_n: int = 12) -> None:
    import matplotlib.pyplot as plt

    g = (
        df.groupby(group_col, dropna=False)[err_col]
        .agg(["mean", "count"])
        .sort_values(["count", "mean"], ascending=[False, True])
        .head(top_n)
    )
    if g.empty:
        return

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(g.index.astype(str).tolist(), g["mean"].to_numpy())
    ax.set_title(title)
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xlabel(group_col)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_reports(
    out_dir: Path,
    crop_cv_splits: int,
    yield_cv_splits: int,
    production_max_rows: int,
    deep_test_size: float,
) -> Path:
    root = find_project_root(PROJECT_ROOT)
    artifacts_dir = _default_artifacts_dir(root)
    system = DualEngineSystem.load(artifacts_dir)

    out_dir = out_dir.resolve()
    _ensure_dir(out_dir)

    # --------------------
    # Load datasets
    # --------------------
    crop_path = root / "Datasets" / "Crop_recommendation.csv"
    prod_path = root / "Datasets" / "IndiaAgricultureCropProduction.csv"

    df_crop = load_crop_recommendation_csv(crop_path)
    df_prod = load_production_csv(prod_path)

    if production_max_rows and production_max_rows > 0 and len(df_prod) > production_max_rows:
        df_prod = df_prod.sample(n=production_max_rows, random_state=42)

    # --------------------
    # Crop engine reports
    # --------------------
    crop_out = out_dir / "crop_engine"
    _ensure_dir(crop_out)

    crop_eng = system.crop_engine
    df_crop2 = crop_eng._ensure_city_season(df_crop)

    X_crop = df_crop2[[*crop_eng.kept_numeric_, *crop_eng.feature_spec.crop_categorical]].copy()
    y_crop = df_crop2["label"].astype(str).to_numpy()

    # RandomForest (pipeline) CV predictions
    rf = crop_eng.rf_best_
    if rf is None:
        raise RuntimeError("Crop RandomForest pipeline missing in artifacts")

    cv = StratifiedKFold(n_splits=crop_cv_splits, shuffle=True, random_state=42)
    rf_pred = np.asarray(cross_val_predict(rf, X_crop, y_crop, cv=cv, n_jobs=-1))
    rf_metrics = _classification_metrics(y_crop, rf_pred)
    _write_json(crop_out / "random_forest_metrics.json", rf_metrics)
    _plot_confusion_matrix(
        np.asarray(rf_metrics["confusion_matrix"], dtype=int),
        rf_metrics["labels"],
        "Crop Engine (RandomForest) - Confusion Matrix (CV)",
        crop_out / "random_forest_confusion_matrix.png",
    )
    _plot_confusion_matrix_normalized(
        np.asarray(rf_metrics["confusion_matrix"], dtype=int),
        rf_metrics["labels"],
        "Crop Engine (RandomForest) - Confusion Matrix (Row-normalized)",
        crop_out / "random_forest_confusion_matrix_normalized.png",
        normalize="true",
    )
    _plot_confusion_errors(
        np.asarray(rf_metrics["confusion_matrix"], dtype=int),
        rf_metrics["labels"],
        "Crop Engine (RandomForest) - Errors Only (Off-diagonal)",
        crop_out / "random_forest_confusion_errors.png",
    )
    _plot_f1_by_class(
        rf_metrics["per_class"],
        "Crop Engine (RandomForest) - F1 by Class (CV)",
        crop_out / "random_forest_f1_by_class.png",
    )

    # BayesNet manual CV
    bayes_preds: list[str] = []
    bayes_true: list[str] = []

    for tr, te in cv.split(X_crop, y_crop):
        Xtr, Xte = X_crop.iloc[tr], X_crop.iloc[te]
        ytr, yte = y_crop[tr], y_crop[te]

        model = SimpleDiscreteBayesNetClassifier(
            numeric_features=crop_eng.kept_numeric_,
            categorical_features=crop_eng.feature_spec.crop_categorical,
            n_bins=6,
            alpha=1.0,
        ).fit(Xtr, pd.Series(ytr))

        pred = model.predict(Xte)
        bayes_preds.extend([str(p) for p in pred])
        bayes_true.extend([str(t) for t in yte])

    bayes_metrics = _classification_metrics(np.array(bayes_true), np.array(bayes_preds))
    _write_json(crop_out / "bayesnet_metrics.json", bayes_metrics)
    _plot_confusion_matrix(
        np.asarray(bayes_metrics["confusion_matrix"], dtype=int),
        bayes_metrics["labels"],
        "Crop Engine (BayesNet) - Confusion Matrix (CV)",
        crop_out / "bayesnet_confusion_matrix.png",
    )
    _plot_confusion_matrix_normalized(
        np.asarray(bayes_metrics["confusion_matrix"], dtype=int),
        bayes_metrics["labels"],
        "Crop Engine (BayesNet) - Confusion Matrix (Row-normalized)",
        crop_out / "bayesnet_confusion_matrix_normalized.png",
        normalize="true",
    )
    _plot_confusion_errors(
        np.asarray(bayes_metrics["confusion_matrix"], dtype=int),
        bayes_metrics["labels"],
        "Crop Engine (BayesNet) - Errors Only (Off-diagonal)",
        crop_out / "bayesnet_confusion_errors.png",
    )
    _plot_f1_by_class(
        bayes_metrics["per_class"],
        "Crop Engine (BayesNet) - F1 by Class (CV)",
        crop_out / "bayesnet_f1_by_class.png",
    )

    # --------------------
    # Yield tabular reports
    # --------------------
    yield_out = out_dir / "yield_tabular"
    _ensure_dir(yield_out)

    y_eng = system.yield_engine
    df_prod2 = y_eng._standardize_yield_t_per_ha(df_prod, y_eng.target_col)

    numeric = list(y_eng.feature_spec.yield_numeric)
    categorical = list(y_eng.feature_spec.yield_categorical)

    if "Year" in df_prod2.columns and "Year_start" not in df_prod2.columns:
        df_prod2 = df_prod2.copy()
        df_prod2["Year_start"] = df_prod2["Year"].astype(str).str.extract(r"(\d{4})", expand=False)
        df_prod2["Year_start"] = pd.to_numeric(df_prod2["Year_start"], errors="coerce")
        numeric = [*numeric, "Year_start"]

    y_true = pd.to_numeric(df_prod2[y_eng.target_col], errors="coerce").to_numpy(dtype=float)
    X = df_prod2[[*numeric, *categorical]].copy()

    if y_eng.xgb_pipeline_ is None:
        raise RuntimeError("Yield tabular pipeline missing in artifacts")

    cv_y = KFold(n_splits=yield_cv_splits, shuffle=True, random_state=42)
    y_pred = np.asarray(cross_val_predict(y_eng.xgb_pipeline_, X, y_true, cv=cv_y, n_jobs=-1))

    y_metrics = {
        "rows": int(len(df_prod2)),
        "cv_splits": int(yield_cv_splits),
        "metrics": _regression_metrics(y_true, y_pred),
        "y_true_summary": _distribution_summary(y_true),
        "y_pred_summary": _distribution_summary(y_pred),
        "abs_error_summary": _distribution_summary(np.abs(y_pred - y_true)),
        "yield_units": str(y_eng.yield_units),
    }
    _write_json(yield_out / "metrics.json", y_metrics)
    _plot_scatter_true_pred(
        y_true,
        y_pred,
        f"Yield (Tabular) - Actual vs Predicted (CV) [{y_eng.yield_units}]",
        yield_out / "actual_vs_predicted.png",
    )
    _plot_scatter_true_pred_zoomed(
        y_true,
        y_pred,
        f"Yield (Tabular) - Actual vs Predicted (Zoomed p01–p99) [{y_eng.yield_units}]",
        yield_out / "actual_vs_predicted_zoomed.png",
        lo=1.0,
        hi=99.0,
    )
    _plot_residuals(
        y_true,
        y_pred,
        "Yield (Tabular) - Residuals (CV)",
        yield_out / "residuals.png",
    )
    _plot_residuals_zoomed(
        y_true,
        y_pred,
        "Yield (Tabular) - Residuals (Zoomed p01–p99)",
        yield_out / "residuals_zoomed.png",
        lo=1.0,
        hi=99.0,
    )

    df_err = pd.DataFrame({
        "Crop": df_prod2.get("Crop", pd.Series(["Unknown"] * len(df_prod2))).astype(str),
        "abs_error": np.abs(y_pred - y_true),
    })
    _plot_group_mae(
        df_err,
        group_col="Crop",
        err_col="abs_error",
        title="Yield (Tabular) - Mean Absolute Error by Crop (Top) (CV)",
        out_path=yield_out / "mae_by_crop_top.png",
        top_n=12,
    )

    # --------------------
    # Yield deep model reports (optional)
    # --------------------
    deep_out = out_dir / "yield_deep"
    _ensure_dir(deep_out)

    deep_enabled = bool(y_eng.dl_model_ is not None and y_eng.dl_window_)
    deep_report: dict[str, Any] = {"enabled": deep_enabled}

    if deep_enabled:
        df_seq = df_prod.copy()
        df_seq = y_eng._standardize_yield_t_per_ha(df_seq, y_eng.target_col)
        df_seq = df_seq.copy()
        df_seq["Year_start"] = df_seq["Year"].astype(str).str.extract(r"(\d{4})", expand=False)
        df_seq["Year_start"] = pd.to_numeric(df_seq["Year_start"], errors="coerce")
        df_seq = df_seq.dropna(subset=["Year_start", y_eng.target_col])
        df_seq["Year_start"] = df_seq["Year_start"].astype(int)

        group_cols = y_eng.dl_group_cols_ or ["State", "Crop"]
        window = int(y_eng.dl_window_ or 0)

        try:
            Xseq, yseq = y_eng._build_sequences(
                df=df_seq,
                group_cols=list(group_cols),
                time_col="Year_start",
                value_col=y_eng.target_col,
                window=window,
            )
            Xseq = Xseq[..., np.newaxis]

            if len(yseq) < 10:
                raise ValueError(f"Too few sequences to evaluate (n={len(yseq)})")

            _, Xte, _, yte = train_test_split(
                Xseq,
                yseq,
                test_size=deep_test_size,
                random_state=42,
            )
            yhat = y_eng.dl_model_.predict(Xte, verbose=0).reshape(-1)

            deep_report = {
                "enabled": True,
                "window": window,
                "group_cols": list(group_cols),
                "sequences": int(len(yseq)),
                "test_size": float(deep_test_size),
                "metrics": _regression_metrics(yte, yhat),
                "y_true_summary": _distribution_summary(yte),
                "y_pred_summary": _distribution_summary(yhat),
                "abs_error_summary": _distribution_summary(np.abs(yhat - yte)),
                "yield_units": str(y_eng.yield_units),
                "note": "Evaluation uses sequences built from the dataset and a random train/test split. If the deep model was trained earlier on overlapping sequences, this is not a strict holdout metric.",
            }

            _plot_scatter_true_pred(
                yte,
                yhat,
                f"Yield (CNN→LSTM) - Actual vs Predicted [{y_eng.yield_units}]",
                deep_out / "actual_vs_predicted.png",
            )
            _plot_scatter_true_pred_zoomed(
                yte,
                yhat,
                f"Yield (CNN→LSTM) - Actual vs Predicted (Zoomed p01–p99) [{y_eng.yield_units}]",
                deep_out / "actual_vs_predicted_zoomed.png",
                lo=1.0,
                hi=99.0,
            )
            _plot_residuals(
                yte,
                yhat,
                "Yield (CNN→LSTM) - Residuals",
                deep_out / "residuals.png",
            )
            _plot_residuals_zoomed(
                yte,
                yhat,
                "Yield (CNN→LSTM) - Residuals (Zoomed p01–p99)",
                deep_out / "residuals_zoomed.png",
                lo=1.0,
                hi=99.0,
            )
        except Exception as ex:  # noqa: BLE001
            deep_report = {
                "enabled": True,
                "window": window,
                "group_cols": list(group_cols),
                "yield_units": str(y_eng.yield_units),
                "error": str(ex),
            }

    _write_json(deep_out / "metrics.json", deep_report)

    run_info = RunInfo(
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
        artifacts_dir=str(artifacts_dir),
        crop_rows=int(len(df_crop)),
        yield_rows_used=int(len(df_prod2)),
        crop_cv_splits=int(crop_cv_splits),
        yield_cv_splits=int(yield_cv_splits),
        deep_enabled=bool(deep_enabled),
    )
    _write_json(out_dir / "run_info.json", asdict(run_info))

    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate metrics + charts for crop/yield models.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory. Default: reports/run_<timestamp>/",
    )
    parser.add_argument("--crop-cv-splits", type=int, default=5)
    parser.add_argument("--yield-cv-splits", type=int, default=3)
    parser.add_argument("--production-max-rows", type=int, default=80000)
    parser.add_argument("--deep-test-size", type=float, default=0.2)

    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_out = find_project_root(PROJECT_ROOT) / "reports" / f"run_{stamp}"
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else default_out

    out = generate_reports(
        out_dir=out_dir,
        crop_cv_splits=int(args.crop_cv_splits),
        yield_cv_splits=int(args.yield_cv_splits),
        production_max_rows=int(args.production_max_rows),
        deep_test_size=float(args.deep_test_size),
    )

    print(f"Wrote reports to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
