from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from flask import Flask, render_template, request

# Import from the project module
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.dual_engine_system import DualEngineSystem, _default_artifacts_dir


def _to_float(v: str | None) -> float | None:
    if v is None:
        return None
    v = v.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _to_int(v: str | None) -> int | None:
    if v is None:
        return None
    v = v.strip()
    if v == "":
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _parse_history(history_text: str | None) -> list[float] | None:
    if history_text is None:
        return None
    s = history_text.strip()
    if not s:
        return None

    # Accept: "1.2, 1.3, 1.1, 1.4" or newline-separated.
    parts: list[str] = []
    for chunk in s.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)

    values: list[float] = []
    for p in parts:
        values.append(float(p))
    return values


def create_app() -> Flask:
    app = Flask(__name__)

    artifacts_dir = _default_artifacts_dir()
    if not artifacts_dir.exists():
        raise RuntimeError(
            f"Artifacts not found at: {artifacts_dir}. "
            "Train first using: python .\\Models\\dual_engine_system.py --retrain --skip-eval"
        )

    system = DualEngineSystem.load(artifacts_dir)

    @app.get("/")
    def index() -> str:
        dl_window = system.yield_engine.dl_window_
        return render_template(
            "index.html",
            pred=None,
            errors=None,
            form_defaults=_default_form_defaults(),
            dl_window=dl_window,
        )

    @app.post("/predict")
    def predict() -> str:
        errors: list[str] = []

        crop_row: dict[str, Any] = {
            "N": _to_float(request.form.get("N")),
            "P": _to_float(request.form.get("P")),
            "K": _to_float(request.form.get("K")),
            "temperature": _to_float(request.form.get("temperature")),
            "humidity": _to_float(request.form.get("humidity")),
            "ph": _to_float(request.form.get("ph")),
            "rainfall": _to_float(request.form.get("rainfall")),
            "City": (request.form.get("City") or "Unknown").strip() or "Unknown",
            "Season": (request.form.get("Season") or "Unknown").strip() or "Unknown",
        }

        # Basic validation for required crop numeric inputs
        for k in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
            if crop_row[k] is None:
                errors.append(f"Missing/invalid crop input: {k}")

        yield_row: dict[str, Any] = {
            "State": (request.form.get("State") or "Unknown").strip() or "Unknown",
            "District": (request.form.get("District") or "Unknown").strip() or "Unknown",
            "Crop": (request.form.get("Crop") or "Unknown").strip() or "Unknown",
            "Season": (request.form.get("YieldSeason") or "Unknown").strip() or "Unknown",
            "Area": _to_float(request.form.get("Area")),
            "Year_start": _to_int(request.form.get("Year_start")),
        }

        if yield_row["Area"] is None:
            errors.append("Missing/invalid yield input: Area")

        # Year_start is optional for inference; the pipeline can still run.
        if yield_row["Year_start"] is None:
            yield_row.pop("Year_start", None)

        pred: dict[str, Any] | None = None
        dl_pred: float | None = None

        if not errors:
            pred = system.predict_dual(crop_row=crop_row, yield_row=yield_row)

            history_text = request.form.get("past_yields")
            try:
                history = _parse_history(history_text)
                if history is not None:
                    dl_pred = system.yield_engine.predict_yield_deep_from_history(history)
            except Exception as ex:  # noqa: BLE001
                errors.append(f"Deep model history error: {ex}")

        # Echo back the form values
        form_defaults = {k: request.form.get(k, "") for k in request.form.keys()}
        form_defaults = {**_default_form_defaults(), **form_defaults}

        return render_template(
            "index.html",
            pred=pred,
            dl_pred=dl_pred,
            errors=errors or None,
            form_defaults=form_defaults,
            dl_window=system.yield_engine.dl_window_,
        )

    return app


def _default_form_defaults() -> dict[str, str]:
    return {
        # Crop defaults
        "N": "90",
        "P": "42",
        "K": "43",
        "temperature": "20.9",
        "humidity": "82.0",
        "ph": "6.5",
        "rainfall": "203.0",
        "City": "Unknown",
        "Season": "Unknown",
        # Yield defaults
        "State": "Andaman and Nicobar Islands",
        "District": "NICOBARS",
        "Crop": "Arecanut",
        "YieldSeason": "Kharif",
        "Area": "1254",
        "Year_start": "2001",
        # Deep model history (optional)
        "past_yields": "",
    }


if __name__ == "__main__":
    app = create_app()
    # Local dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
