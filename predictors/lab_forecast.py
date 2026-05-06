import os
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models").strip() or "models")
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (BASE_DIR / MODEL_DIR).resolve()

_models = {}
_available = None


def _load_available():
    global _available
    if _available is None:
        try:
            _available = joblib.load(MODEL_DIR / "lab_forecast_available.joblib")
        except FileNotFoundError:
            _available = []


def _load_model(test_name: str):
    safe = test_name.lower().replace(" ", "_")
    if safe not in _models:
        path = MODEL_DIR / f"lab_forecast_{safe}.joblib"
        if not path.exists():
            return None
        _models[safe] = joblib.load(path)
    return _models[safe]


def predict(test_name: str, last_values: list) -> dict:
    _load_available()
    model = _load_model(test_name)
    if model is None:
        return {"error": f"No forecast model available for '{test_name}'"}

    window = (list(last_values)[-3:] + [0.0] * 3)[:3]
    X = np.array([window])
    predicted = float(model.predict(X)[0])

    trend = "Stable"
    if last_values and predicted > last_values[-1] * 1.10:
        trend = "Rising"
    elif last_values and predicted < last_values[-1] * 0.90:
        trend = "Falling"

    return {
        "test_name": test_name,
        "predicted_value": round(predicted, 2),
        "trend": trend,
        "input_window": window,
        "available_tests": _available,
    }
