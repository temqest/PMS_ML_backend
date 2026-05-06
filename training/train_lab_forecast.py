"""
Predicts the next numeric value for a lab test using a rolling window.
One regression model is saved per lab test name.
"""

import os
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "pms")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models").strip() or "models")
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (BASE_DIR / MODEL_DIR).resolve()
WINDOW = 3


def _parse_value(details):
    numeric = details.get("labResultNumeric")
    if numeric is not None:
        try:
            return float(numeric)
        except (TypeError, ValueError):
            pass

    try:
        raw = details.get("labResultValue", "")
        return float(str(raw).split()[0])
    except (ValueError, IndexError):
        return None


def train():
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI is required for lab forecast training.")

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]

    lab_records = list(
        db.healthrecords.find(
            {"record_type": "Lab Result", "save_state": "final"},
            {"patient_id": 1, "record_date": 1, "details": 1},
        )
    )

    series = defaultdict(list)
    for record in lab_records:
        details = record.get("details") or {}
        test_name = details.get("labTestName")
        value = _parse_value(details)
        date = record.get("record_date")
        patient_id = record.get("patient_id")
        if test_name and value is not None and date is not None and patient_id:
            series[(patient_id, test_name)].append((date, value))

    by_test = defaultdict(list)
    for (_, test_name), readings in series.items():
        readings.sort(key=lambda x: x[0])
        values = [value for _, value in readings]
        for idx in range(WINDOW, len(values)):
            window = values[idx - WINDOW : idx]
            target = values[idx]
            by_test[test_name].append(window + [target])

    os.makedirs(MODEL_DIR, exist_ok=True)
    trained = []

    skipped_tests = []
    for test_name, rows in by_test.items():
        if len(rows) < 5:
            skipped_tests.append(f"{test_name} (only {len(rows)} training rows, need >=5)")
            continue

        arr = np.array(rows)
        X, y = arr[:, :WINDOW], arr[:, WINDOW]
        if len(X) < 4:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0)),
            ]
        )
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"  {test_name}: MAE = {mae:.2f}")

        safe_name = test_name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"lab_forecast_{safe_name}.joblib")
        trained.append(test_name)

    joblib.dump(trained, MODEL_DIR / "lab_forecast_available.joblib")
    print(f"\nTrained forecasts for: {trained}")
    if skipped_tests:
        print(f"Skipped (insufficient rows): {len(skipped_tests)} test(s), e.g. {skipped_tests[:5]}")


if __name__ == "__main__":
    train()
