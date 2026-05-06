# Machine Learning Backend (Predictive Care)

This folder hosts the Python FastAPI service for Predictive Care ML training and inference.

## Model semantics (important)

### Utilization / “readmission” proxy (train_readmission.py)

There is **no hospitalization or claims feed** in this stack. The readmission model is trained as a **90-day visit proxy**:

- **Rows:** one training row per `(patient_id, index_visit)` where the index visit is any final `Visit` health record.
- **Features:** computed from Mongo data **strictly before** the index visit time (`extract_features_for_patient_at`), so the label is not leaked into inputs.
- **Label:** `1` if the patient has any subsequent `Visit` with `visitType` in `("Urgent", "Follow-up")` within **(index_time, index_time + 90 days]**, else `0`.
- **Evaluation:** `GroupShuffleSplit` by `patient_id` so rows from the same patient do not straddle train and test.

UI and API copy should describe this as **utilization / follow-up risk**, not literal 30-day hospital readmission.

### Chronic / composite risk (train_chronic_risk.py)

Labels are tiers (**Low / Moderate / High / Critical**) derived from **`chronic_burden_score`**, which matches the Node rule-based `computeChronicDiseaseScore` logic (diagnosis keywords, abnormal labs, elevated BP). The random forest learns an approximation; it does not replace clinical coding.

### Feature version

`training/extract_features.py` exports `FEATURE_VERSION` (currently `v2`). Training writes `*_feature_version.joblib` next to each classifier. Inference features use `urgent_visits_prior` and `adherence_risk_proxy`; **retrain** after pulling these changes so `readmission_features.joblib` / `chronic_risk_features.joblib` align with the Node app.

## Implemented scope

- Service scaffolding, configuration, and health routes
- MongoDB feature extraction (`extract_features.py`) for per-patient vectors and time-safe training rows
- Model training scripts:
  - `train_readmission.py` — logistic regression on proxy rows
  - `train_chronic_risk.py` — random forest on tier labels from chronic burden
  - `train_lab_forecast.py` — ridge regression per lab test name
  - `train_anomaly.py` — isolation forest on vitals / lab ratio features
- Predictor modules that load joblib artifacts lazily
- FastAPI prediction APIs, full prediction endpoint, and train endpoint
- Shared request schemas (`schemas/request_schemas.py`)

## Structure

- `main.py` — FastAPI app entrypoint + routes
- `requirements.txt` — Python dependencies
- `.env.example` — required environment variables
- `models/` — trained model artifacts (`.joblib`)
- `training/` — feature extraction and model training scripts
- `predictors/` — model inference modules
- `schemas/request_schemas.py` — request payload contracts

## Setup and run

```bash
cd machine_learning_backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Environment highlights:

- `MONGO_URI`: Mongo connection string
- `MONGO_DB_NAME`: database name to read training/prediction data from (default `pms`)
- `MODEL_DIR`: where `.joblib` model artifacts are written/read
- `ALLOWED_ORIGINS`: comma-separated frontend/backend origins allowed by CORS

## Train models (first run or after schema / feature changes)

Run from the `machine_learning_backend` directory (or ensure `PYTHONPATH` includes `training`):

```bash
python training/train_readmission.py
python training/train_chronic_risk.py
python training/train_lab_forecast.py
python training/train_anomaly.py
```

Retrain after bulk health-record backfill or when `FEATURE_VERSION` changes.

## Start API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Render start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Endpoints

- GET `/`
- GET `/health`
- GET `/features/{patient_id}`
- POST `/predict/readmission`
- POST `/predict/chronic-risk`
- POST `/predict/lab-forecast`
- POST `/predict/anomaly`
- GET `/predict/full/{patient_id}`
- POST `/train`
