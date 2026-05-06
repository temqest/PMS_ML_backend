import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "pms")
FEATURE_VERSION = "v2"

HIGH_RISK_KEYWORDS = [
    "hypertension",
    "type 2 diabetes",
    "chronic",
    "heart failure",
    "copd",
    "obesity",
    "renal",
    "kidney",
    "coronary",
    "stroke",
]


def get_db():
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI is required for ML feature extraction and training.")
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


def compute_chronic_burden_node_style(visits: list[dict[str, Any]], labs: list[dict[str, Any]]) -> float:
    """Match Node `riskScore.service` computeChronicDiseaseScore logic (0–100 cap)."""
    score = 0.0
    for v in visits:
        d = str((v.get("details") or {}).get("visitAssessment", "")).lower()
        for kw in HIGH_RISK_KEYWORDS:
            if kw in d:
                score += 10
                break
    abnormal_labs = [
        l
        for l in labs
        if (l.get("details") or {}).get("labStatus") and (l.get("details") or {}).get("labStatus") != "Normal"
    ]
    score += len(abnormal_labs) * 8
    for v in visits:
        try:
            systolic = int(float(str((v.get("details") or {}).get("visitBpSystolic", "0") or "0")))
        except (TypeError, ValueError):
            systolic = 0
        if systolic >= 140:
            score += 6
    return float(min(score, 100))


def extract_features_for_patient_at(
    patient_id: str, as_of: datetime, db=None
) -> dict[str, Any] | None:
    """
    Feature vector using only data strictly before `as_of` (for leakage-free training rows).
    Appointments: scheduled_at < as_of.
    """
    if db is None:
        db = get_db()

    patient = db.patients.find_one({"patient_id": patient_id})
    if not patient:
        return None

    records = list(
        db.healthrecords.find({"patient_id": patient_id, "save_state": "final"})
    )
    appointments = list(db.appointments.find({"patient_id": patient_id}))

    def before_as_of(rec: dict[str, Any]) -> bool:
        return _record_date(rec) < as_of

    visits = [
        r
        for r in records
        if r.get("record_type") == "Visit" and before_as_of(r)
    ]
    labs = [
        r
        for r in records
        if r.get("record_type") == "Lab Result" and before_as_of(r)
    ]
    scripts = [
        r
        for r in records
        if r.get("record_type") == "Prescription" and before_as_of(r)
    ]
    appointments = [a for a in appointments if _appointment_at(a) < as_of]

    cutoff_90 = as_of - timedelta(days=90)
    cutoff_180 = as_of - timedelta(days=180)

    dob = _to_datetime(patient.get("date_of_birth"))
    age = (as_of - dob).days / 365.25 if dob else 40.0
    gender_male = 1 if str(patient.get("gender", "")).lower() == "male" else 0

    total_visits = len(visits)
    visits_90d = sum(1 for v in visits if cutoff_90 <= _record_date(v) < as_of)
    visits_180d = sum(1 for v in visits if cutoff_180 <= _record_date(v) < as_of)
    urgent_visits_prior = sum(
        1
        for v in visits
        if _visit_type(v) in ("Urgent", "Follow-up")
    )

    systolics = _extract_numeric_details(visits, "visitBpSystolic")
    diastolics = _extract_numeric_details(visits, "visitBpDiastolic")
    heart_rates = _extract_numeric_details(visits, "visitHeartRate")
    weights = _extract_numeric_details(visits, "visitWeight")
    temperatures = _extract_numeric_details(visits, "visitTemperature")

    avg_systolic = float(np.mean(systolics)) if systolics else 120.0
    avg_diastolic = float(np.mean(diastolics)) if diastolics else 80.0
    avg_heart_rate = float(np.mean(heart_rates)) if heart_rates else 72.0
    avg_weight = float(np.mean(weights)) if weights else 160.0
    avg_temperature = float(np.mean(temperatures)) if temperatures else 98.6
    high_bp_count = sum(1 for s in systolics if s >= 140)

    all_diagnoses = " ".join(
        [str((v.get("details") or {}).get("visitAssessment", "")).lower() for v in visits]
    )
    chronic_keyword_hits = sum(1 for kw in HIGH_RISK_KEYWORDS if kw in all_diagnoses)

    total_labs = len(labs)
    abnormal_labs = sum(
        1
        for l in labs
        if (l.get("details") or {}).get("labStatus") in ("Abnormal", "Critical")
    )
    critical_labs = sum(
        1 for l in labs if (l.get("details") or {}).get("labStatus") == "Critical"
    )
    abnormal_ratio = abnormal_labs / total_labs if total_labs else 0.0

    glucose_records = [
        l
        for l in labs
        if "glucose" in str((l.get("details") or {}).get("labTestName", "")).lower()
    ]
    glucose_records.sort(key=lambda x: _record_date(x))
    last_glucose = (
        _extract_lab_numeric(glucose_records[-1].get("details") or {})
        if glucose_records
        else 100.0
    )

    total_appts = len(appointments)
    completed = sum(1 for a in appointments if a.get("status") == "Completed")
    completion_rate = completed / total_appts if total_appts else 1.0
    no_show_count = total_appts - completed

    total_scripts = len(scripts)
    unique_meds = len({name for s in scripts for name in _prescription_medicine_names(s)})

    chronic_burden_score = compute_chronic_burden_node_style(visits, labs)

    adherence_rows = list(
        db.adherencerecords.find(
            {"patient_id": patient_id, "last_assessed_at": {"$lt": as_of}}
        )
    )
    if adherence_rows:
        avg_adh = float(
            np.mean(
                [
                    float(r.get("adherence_score") or 100)
                    for r in adherence_rows
                ]
            )
        )
        adherence_risk_proxy = float(min(100, max(0, 100 - avg_adh)))
    else:
        adherence_risk_proxy = 0.0

    return {
        "feature_version": FEATURE_VERSION,
        "age": age,
        "gender_male": gender_male,
        "total_visits": float(total_visits),
        "visits_90d": float(visits_90d),
        "visits_180d": float(visits_180d),
        "urgent_visits_prior": float(urgent_visits_prior),
        "avg_systolic": avg_systolic,
        "avg_diastolic": avg_diastolic,
        "avg_heart_rate": avg_heart_rate,
        "avg_weight": avg_weight,
        "avg_temperature": avg_temperature,
        "high_bp_count": float(high_bp_count),
        "chronic_keyword_hits": float(chronic_keyword_hits),
        "chronic_burden_score": chronic_burden_score,
        "total_labs": float(total_labs),
        "abnormal_labs": float(abnormal_labs),
        "critical_labs": float(critical_labs),
        "abnormal_ratio": abnormal_ratio,
        "last_glucose": last_glucose,
        "total_appts": float(total_appts),
        "completion_rate": completion_rate,
        "no_show_count": float(no_show_count),
        "total_scripts": float(total_scripts),
        "unique_meds": float(unique_meds),
        "adherence_risk_proxy": adherence_risk_proxy,
    }


def extract_features_for_patient(patient_id: str, db=None) -> dict[str, Any] | None:
    """Inference-time features: all final records up to now (record_date <= now)."""
    if db is None:
        db = get_db()
    now = datetime.utcnow()
    patient = db.patients.find_one({"patient_id": patient_id})
    if not patient:
        return None

    records = list(
        db.healthrecords.find({"patient_id": patient_id, "save_state": "final"})
    )
    appointments = list(db.appointments.find({"patient_id": patient_id}))

    def upto_now(rec: dict[str, Any]) -> bool:
        return _record_date(rec) <= now

    visits = [r for r in records if r.get("record_type") == "Visit" and upto_now(r)]
    labs = [r for r in records if r.get("record_type") == "Lab Result" and upto_now(r)]
    scripts = [r for r in records if r.get("record_type") == "Prescription" and upto_now(r)]
    appointments = [a for a in appointments if _appointment_at(a) <= now]

    as_of_effective = now
    cutoff_90 = as_of_effective - timedelta(days=90)
    cutoff_180 = as_of_effective - timedelta(days=180)

    dob = _to_datetime(patient.get("date_of_birth"))
    age = (as_of_effective - dob).days / 365.25 if dob else 40.0
    gender_male = 1 if str(patient.get("gender", "")).lower() == "male" else 0

    total_visits = len(visits)
    visits_90d = sum(1 for v in visits if cutoff_90 <= _record_date(v) <= now)
    visits_180d = sum(1 for v in visits if cutoff_180 <= _record_date(v) <= now)
    urgent_visits_prior = sum(
        1
        for v in visits
        if _visit_type(v) in ("Urgent", "Follow-up")
    )

    systolics = _extract_numeric_details(visits, "visitBpSystolic")
    diastolics = _extract_numeric_details(visits, "visitBpDiastolic")
    heart_rates = _extract_numeric_details(visits, "visitHeartRate")
    weights = _extract_numeric_details(visits, "visitWeight")
    temperatures = _extract_numeric_details(visits, "visitTemperature")

    avg_systolic = float(np.mean(systolics)) if systolics else 120.0
    avg_diastolic = float(np.mean(diastolics)) if diastolics else 80.0
    avg_heart_rate = float(np.mean(heart_rates)) if heart_rates else 72.0
    avg_weight = float(np.mean(weights)) if weights else 160.0
    avg_temperature = float(np.mean(temperatures)) if temperatures else 98.6
    high_bp_count = sum(1 for s in systolics if s >= 140)

    all_diagnoses = " ".join(
        [str((v.get("details") or {}).get("visitAssessment", "")).lower() for v in visits]
    )
    chronic_keyword_hits = sum(1 for kw in HIGH_RISK_KEYWORDS if kw in all_diagnoses)

    total_labs = len(labs)
    abnormal_labs = sum(
        1
        for l in labs
        if (l.get("details") or {}).get("labStatus") in ("Abnormal", "Critical")
    )
    critical_labs = sum(
        1 for l in labs if (l.get("details") or {}).get("labStatus") == "Critical"
    )
    abnormal_ratio = abnormal_labs / total_labs if total_labs else 0.0

    glucose_records = [
        l
        for l in labs
        if "glucose" in str((l.get("details") or {}).get("labTestName", "")).lower()
    ]
    glucose_records.sort(key=lambda x: _record_date(x))
    last_glucose = (
        _extract_lab_numeric(glucose_records[-1].get("details") or {})
        if glucose_records
        else 100.0
    )

    total_appts = len(appointments)
    completed = sum(1 for a in appointments if a.get("status") == "Completed")
    completion_rate = completed / total_appts if total_appts else 1.0
    no_show_count = total_appts - completed

    total_scripts = len(scripts)
    unique_meds = len({name for s in scripts for name in _prescription_medicine_names(s)})

    chronic_burden_score = compute_chronic_burden_node_style(visits, labs)

    adherence_rows = list(db.adherencerecords.find({"patient_id": patient_id}))
    if adherence_rows:
        avg_adh = float(
            np.mean([float(r.get("adherence_score") or 100) for r in adherence_rows])
        )
        adherence_risk_proxy = float(min(100, max(0, 100 - avg_adh)))
    else:
        adherence_risk_proxy = 0.0

    return {
        "feature_version": FEATURE_VERSION,
        "age": age,
        "gender_male": gender_male,
        "total_visits": float(total_visits),
        "visits_90d": float(visits_90d),
        "visits_180d": float(visits_180d),
        "urgent_visits_prior": float(urgent_visits_prior),
        "avg_systolic": avg_systolic,
        "avg_diastolic": avg_diastolic,
        "avg_heart_rate": avg_heart_rate,
        "avg_weight": avg_weight,
        "avg_temperature": avg_temperature,
        "high_bp_count": float(high_bp_count),
        "chronic_keyword_hits": float(chronic_keyword_hits),
        "chronic_burden_score": chronic_burden_score,
        "total_labs": float(total_labs),
        "abnormal_labs": float(abnormal_labs),
        "critical_labs": float(critical_labs),
        "abnormal_ratio": abnormal_ratio,
        "last_glucose": last_glucose,
        "total_appts": float(total_appts),
        "completion_rate": completion_rate,
        "no_show_count": float(no_show_count),
        "total_scripts": float(total_scripts),
        "unique_meds": float(unique_meds),
        "adherence_risk_proxy": adherence_risk_proxy,
    }


def extract_all_patients() -> pd.DataFrame:
    db = get_db()
    patients = list(db.patients.find({}, {"patient_id": 1, "_id": 0}))
    rows = []
    for patient in patients:
        pid = patient.get("patient_id")
        if not pid:
            continue
        features = extract_features_for_patient(pid, db)
        if features:
            features["patient_id"] = pid
            rows.append(features)
    return pd.DataFrame(rows)


def build_readmission_proxy_training_df(
    db=None,
    window_days: int = 90,
    target_visit_types: Tuple[str, ...] = ("Urgent", "Follow-up"),
) -> pd.DataFrame:
    """
    One row per (patient, index_visit): features from data strictly before index time;
    label = 1 if any visit of target_visit_types occurs in (index_time, index_time + window_days].
    """
    if db is None:
        db = get_db()
    patients = list(db.patients.find({}, {"patient_id": 1, "_id": 0}))
    rows: list[dict[str, Any]] = []
    window = timedelta(days=window_days)

    for patient in patients:
        pid = patient.get("patient_id")
        if not pid:
            continue
        all_visits = list(
            db.healthrecords.find(
                {
                    "patient_id": pid,
                    "record_type": "Visit",
                    "save_state": "final",
                },
                {"record_date": 1, "details": 1},
            )
        )
        all_visits.sort(key=_record_date)
        if len(all_visits) < 1:
            continue

        for idx_visit in all_visits:
            d0 = _record_date(idx_visit)
            feats = extract_features_for_patient_at(pid, d0, db)
            if not feats:
                continue
            end = d0 + window
            label = 0
            for v in all_visits:
                vd = _record_date(v)
                if d0 < vd <= end:
                    vt = _visit_type(v)
                    if vt in target_visit_types:
                        label = 1
                        break
            row = {k: v for k, v in feats.items() if k != "feature_version"}
            row["label"] = label
            row["patient_id"] = pid
            rows.append(row)

    return pd.DataFrame(rows)


def _appointment_at(appt: dict[str, Any]) -> datetime:
    raw = appt.get("scheduled_at")
    parsed = _to_datetime(raw)
    return parsed if parsed else datetime.utcnow()


def _record_date(record: dict[str, Any]) -> datetime:
    date_value = record.get("record_date")
    parsed = _to_datetime(date_value)
    return parsed if parsed else datetime.utcnow()


def _to_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _extract_numeric_details(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        raw = (record.get("details") or {}).get(key)
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


def _visit_type(record: dict[str, Any]) -> str:
    details = record.get("details") or {}
    return str(details.get("visitType") or details.get("visitDisposition") or "")


def _prescription_medicine_names(record: dict[str, Any]) -> list[str]:
    details = record.get("details") or {}
    names: list[str] = []

    for key in ("medicationName", "prescriptionMedicationName"):
        value = str(details.get(key) or "").strip().lower()
        if value:
            names.append(value)

    medicines = details.get("medicines")
    if isinstance(medicines, list):
        for medicine in medicines:
            if not isinstance(medicine, dict):
                continue
            value = str(
                medicine.get("medicineName")
                or medicine.get("medicationName")
                or medicine.get("name")
                or ""
            ).strip().lower()
            if value:
                names.append(value)

    return names or ["unknown"]


def _extract_lab_numeric(details: dict[str, Any]) -> float:
    numeric = details.get("labResultNumeric")
    if numeric is not None:
        try:
            return float(numeric)
        except (TypeError, ValueError):
            pass
    return _parse_lab_value(details.get("labResultValue", ""))


def _parse_lab_value(raw: Any) -> float:
    try:
        return float(str(raw).split()[0])
    except (ValueError, IndexError):
        return 0.0
