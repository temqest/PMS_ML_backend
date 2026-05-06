import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from schemas.request_schemas import (
    AnomalyRequest,
    ChronicRiskRequest,
    LabForecastRequest,
    ReadmissionRequest,
    TrainRequest,
)
from predictors import anomaly, chronic_risk, lab_forecast, readmission
from training.extract_features import extract_features_for_patient

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP_NAME = "PMS Predictive Care ML Service"
APP_VERSION = "0.2.0-phase2"
MODEL_DIR_RAW = os.getenv("MODEL_DIR", "models").strip() or "models"
MODEL_DIR = Path(MODEL_DIR_RAW)
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (BASE_DIR / MODEL_DIR).resolve()

ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
    }


@app.get("/health")
def health():
    # Startup dependency checks will grow in later phases.
    return {
        "status": "ok",
        "service": "predictive-care-ml",
        "model_dir": str(MODEL_DIR),
        "model_dir_exists": MODEL_DIR.is_dir(),
        "allowed_origins": ALLOWED_ORIGINS,
    }


@app.get("/features/{patient_id}")
def get_features(patient_id: str):
    features = extract_features_for_patient(patient_id)
    if not features:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found or has no records")
    return features


@app.post("/predict/readmission")
def predict_readmission(req: ReadmissionRequest):
    try:
        result = readmission.predict(req.features.model_dump())
        return {"patient_id": req.patient_id, **result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Readmission model not trained yet. POST /train first.") from exc


@app.post("/predict/chronic-risk")
def predict_chronic_risk(req: ChronicRiskRequest):
    try:
        result = chronic_risk.predict(req.features.model_dump())
        return {"patient_id": req.patient_id, **result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Chronic risk model not trained yet. POST /train first.") from exc


@app.post("/predict/lab-forecast")
def predict_lab_forecast(req: LabForecastRequest):
    result = lab_forecast.predict(req.test_name, req.last_values)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return {"patient_id": req.patient_id, **result}


@app.post("/predict/anomaly")
def predict_anomaly(req: AnomalyRequest):
    try:
        result = anomaly.predict(req.features.model_dump())
        return {"patient_id": req.patient_id, **result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Anomaly model not trained yet. POST /train first.") from exc


@app.get("/predict/full/{patient_id}")
def predict_full(patient_id: str):
    features = extract_features_for_patient(patient_id)
    if not features:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    results: dict[str, Any] = {"patient_id": patient_id, "features": features}

    try:
        results["readmission"] = readmission.predict(features)
    except Exception as exc:  # noqa: BLE001 - keep endpoint resilient per integration contract
        results["readmission"] = {"error": str(exc)}

    try:
        results["chronic_risk"] = chronic_risk.predict(features)
    except Exception as exc:  # noqa: BLE001 - keep endpoint resilient per integration contract
        results["chronic_risk"] = {"error": str(exc)}

    try:
        results["anomaly"] = anomaly.predict(features)
    except Exception as exc:  # noqa: BLE001 - keep endpoint resilient per integration contract
        results["anomaly"] = {"error": str(exc)}

    return results


@app.post("/train")
def train_models(req: TrainRequest):
    to_train = req.models if req.models and req.models != ["all"] else [
        "readmission",
        "chronic_risk",
        "lab_forecast",
        "anomaly",
    ]

    script_map = {
        "readmission": BASE_DIR / "training" / "train_readmission.py",
        "chronic_risk": BASE_DIR / "training" / "train_chronic_risk.py",
        "lab_forecast": BASE_DIR / "training" / "train_lab_forecast.py",
        "anomaly": BASE_DIR / "training" / "train_anomaly.py",
    }

    results: dict[str, str] = {}
    for name in to_train:
        script = script_map.get(name)
        if not script:
            results[name] = "unknown model"
            continue

        try:
            subprocess.run(
                [sys.executable, str(script)],
                check=True,
                capture_output=True,
                text=True,
                cwd=BASE_DIR,
            )
            results[name] = "trained"
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            results[name] = f"failed: {stderr[:300] if stderr else 'see service logs'}"

    return {"trained": results}
