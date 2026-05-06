"""
Detects patient outliers based on population-level vital patterns.
"""

import os

import joblib
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from extract_features import extract_all_patients

load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR", "./models")

VITALS_FEATURES = [
    "age",
    "avg_systolic",
    "avg_diastolic",
    "avg_heart_rate",
    "avg_weight",
    "avg_temperature",
    "abnormal_ratio",
    "last_glucose",
]


def train():
    print("Extracting features for anomaly model...")
    df = extract_all_patients()

    if len(df) < 10:
        print("Not enough data.")
        return

    X = df[VITALS_FEATURES].fillna(df[VITALS_FEATURES].median())

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "iso",
                IsolationForest(
                    n_estimators=200,
                    contamination=0.1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X)

    scaled = model.named_steps["scaler"].transform(X)
    scores = model.named_steps["iso"].decision_function(scaled)
    print(f"Anomaly score range: {scores.min():.3f} -> {scores.max():.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "anomaly_model.joblib"))
    joblib.dump(VITALS_FEATURES, os.path.join(MODEL_DIR, "anomaly_features.joblib"))
    print(f"Model saved -> {MODEL_DIR}/anomaly_model.joblib")


if __name__ == "__main__":
    train()
