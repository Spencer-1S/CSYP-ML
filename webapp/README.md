# Web App (Flask)

## 1) Train & save models (one-time)

From the project root:

- Tabular (XGBoost) + crop models:
  - `python .\Models\dual_engine_system.py --retrain --skip-eval`

- Also train the optional deep model (CNN→LSTM):
  - `python .\Models\dual_engine_system.py --retrain --skip-eval --train-dl --dl-window 4 --dl-epochs 2`

Artifacts are saved under `Models/artifacts/dual_engine_system/`.

## 2) Run the web app

From the project root:

- `python .\webapp\app.py`

Then open:

- http://127.0.0.1:5000/

## Notes

- The **tabular yield** prediction runs from the saved pipeline.
- The **deep yield** prediction is optional and requires you to paste exactly `dl_window` past yields.
