"""
Chronic / composite disease risk tiers aligned with Node `computeChronicDiseaseScore`
(0–100 burden) then bucketed into Low / Moderate / High / Critical.
"""

import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from extract_features import FEATURE_VERSION, extract_all_patients

load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR", "./models")

FEATURES = [
    "age",
    "gender_male",
    "total_visits",
    "visits_90d",
    "avg_systolic",
    "avg_diastolic",
    "avg_heart_rate",
    "avg_weight",
    "high_bp_count",
    "chronic_keyword_hits",
    "total_labs",
    "abnormal_labs",
    "critical_labs",
    "abnormal_ratio",
    "last_glucose",
    "total_scripts",
    "unique_meds",
    "adherence_risk_proxy",
]


def build_labels(df: pd.DataFrame) -> pd.Series:
    def tier_from_burden(score: float) -> str:
        if score >= 70:
            return "Critical"
        if score >= 45:
            return "High"
        if score >= 20:
            return "Moderate"
        return "Low"

    return df["chronic_burden_score"].apply(tier_from_burden)


def train():
    print("Extracting features...")
    df = extract_all_patients()

    if len(df) < 10:
        print("Not enough data to train.")
        return

    df["label"] = build_labels(df)
    X = df[FEATURES].fillna(0)
    y = df["label"]

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("\n-- Chronic composite risk model evaluation --")
    print(
        classification_report(
            y_test,
            model.predict(X_test),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    importances = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: -x[1],
    )
    print("\nTop 5 features:")
    for feature, importance in importances[:5]:
        print(f"  {feature}: {importance:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "chronic_risk_model.joblib"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "chronic_risk_encoder.joblib"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "chronic_risk_features.joblib"))
    joblib.dump(FEATURE_VERSION, os.path.join(MODEL_DIR, "chronic_risk_feature_version.joblib"))
    print(f"Model saved -> {MODEL_DIR}/chronic_risk_model.joblib (feature_version={FEATURE_VERSION})")


if __name__ == "__main__":
    train()
