"""
Train a proxy model: P(Urgent or Follow-up visit within 90d after an index visit | pre-index features).

Labels are forward-looking from visit timestamps only (no hospital feed). See README.
"""

import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from extract_features import FEATURE_VERSION, build_readmission_proxy_training_df

load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR", "./models")

FEATURES = [
    "age",
    "gender_male",
    "total_visits",
    "visits_90d",
    "visits_180d",
    "urgent_visits_prior",
    "avg_systolic",
    "avg_diastolic",
    "avg_heart_rate",
    "avg_weight",
    "avg_temperature",
    "high_bp_count",
    "chronic_keyword_hits",
    "total_labs",
    "abnormal_labs",
    "critical_labs",
    "abnormal_ratio",
    "last_glucose",
    "total_appts",
    "completion_rate",
    "no_show_count",
    "total_scripts",
    "unique_meds",
    "adherence_risk_proxy",
]


def train():
    print("Building readmission-proxy training rows...")
    df = build_readmission_proxy_training_df()

    if len(df) < 10:
        print("Not enough training rows. Need at least 10 (patient x index_visit) rows.")
        return

    X = df[FEATURES].fillna(0)
    y = df["label"].astype(int)
    groups = df["patient_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    df_train = pd.concat([X_train, y_train], axis=1)
    df_majority = df_train[df_train.label == 0]
    df_minority = df_train[df_train.label == 1]
    if len(df_minority) > 0 and len(df_minority) < len(df_majority):
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42,
        )
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        X_train = df_balanced[FEATURES].fillna(0)
        y_train = df_balanced["label"]

    stratify = y_train if y_train.nunique() > 1 else None
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    print("\n-- Readmission proxy model evaluation --")
    print(classification_report(y_test, pipeline.predict(X_test), zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "readmission_model.joblib"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "readmission_features.joblib"))
    joblib.dump(FEATURE_VERSION, os.path.join(MODEL_DIR, "readmission_feature_version.joblib"))
    print(f"Model saved -> {MODEL_DIR}/readmission_model.joblib (feature_version={FEATURE_VERSION})")


if __name__ == "__main__":
    train()
