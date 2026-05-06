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
        _model = joblib.load(MODEL_DIR / "readmission_model.joblib")
        _features = joblib.load(MODEL_DIR / "readmission_features.joblib")


def predict(features: dict) -> dict:
    _load()
    X = np.array([[features.get(feature, 0.0) for feature in _features]])
    probability = float(_model.predict_proba(X)[0][1])
    label = "High" if probability >= 0.65 else "Moderate" if probability >= 0.35 else "Low"
    return {
        "readmission_probability": round(probability, 4),
        "readmission_risk_level": label,
        "readmission_score": round(probability * 100, 1),
        "readmission_proxy_90d": True,
    }
