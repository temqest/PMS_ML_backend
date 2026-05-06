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
_encoder = None
_features = None


def _load():
    global _model, _encoder, _features
    if _model is None:
        _model = joblib.load(MODEL_DIR / "chronic_risk_model.joblib")
        _encoder = joblib.load(MODEL_DIR / "chronic_risk_encoder.joblib")
        _features = joblib.load(MODEL_DIR / "chronic_risk_features.joblib")


def predict(features: dict) -> dict:
    _load()
    X = np.array([[features.get(feature, 0.0) for feature in _features]])
    label_encoded = _model.predict(X)[0]
    probabilities = _model.predict_proba(X)[0]
    label = _encoder.inverse_transform([label_encoded])[0]
    confidence = float(np.max(probabilities))

    importances = sorted(
        zip(_features, _model.feature_importances_),
        key=lambda x: -x[1],
    )
    top_factors = [
        {"feature": feature, "importance": round(importance, 4)}
        for feature, importance in importances[:5]
    ]

    return {
        "chronic_risk_level": label,
        "chronic_risk_score": round(confidence * 100, 1),
        "confidence": round(confidence, 4),
        "top_factors": top_factors,
    }
