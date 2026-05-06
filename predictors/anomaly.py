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

_model = None
_features = None


def _load():
    global _model, _features
    if _model is None:
        _model = joblib.load(MODEL_DIR / "anomaly_model.joblib")
        _features = joblib.load(MODEL_DIR / "anomaly_features.joblib")


def predict(features: dict) -> dict:
    _load()
    X = np.array([[features.get(feature, 0.0) for feature in _features]])
    score = float(_model.decision_function(X)[0])
    is_anomaly = bool(_model.predict(X)[0] == -1)
    anomaly_score = round(max(0, min(100, (1 - score) * 50)), 1)

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "raw_score": round(score, 4),
        "interpretation": (
            "Outlier - vitals significantly deviate from population norms"
            if is_anomaly
            else "Within normal population range"
        ),
    }
