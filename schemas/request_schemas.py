from typing import List, Optional

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """Feature vector from GET /features/{id} or training pipeline (v2 adds proxy fields)."""

    feature_version: Optional[str] = None
    age: float = 0.0
    gender_male: int = 0
    total_visits: float = 0.0
    visits_90d: float = 0.0
    visits_180d: float = 0.0
    urgent_visits_prior: float = Field(default=0.0, description="Urgent/Follow-up visits strictly before as-of")
    avg_systolic: float = 0.0
    avg_diastolic: float = 0.0
    avg_heart_rate: float = 0.0
    avg_weight: float = 0.0
    avg_temperature: float = 0.0
    high_bp_count: float = 0.0
    chronic_keyword_hits: float = 0.0
    chronic_burden_score: Optional[float] = None
    total_labs: float = 0.0
    abnormal_labs: float = 0.0
    critical_labs: float = 0.0
    abnormal_ratio: float = 0.0
    last_glucose: float = 0.0
    total_appts: float = 0.0
    completion_rate: float = 0.0
    no_show_count: float = 0.0
    total_scripts: float = 0.0
    unique_meds: float = 0.0
    adherence_risk_proxy: float = 0.0


class ReadmissionRequest(BaseModel):
    patient_id: str
    features: PatientFeatures


class ChronicRiskRequest(BaseModel):
    patient_id: str
    features: PatientFeatures


class LabForecastRequest(BaseModel):
    patient_id: str
    test_name: str
    last_values: List[float]


class AnomalyRequest(BaseModel):
    patient_id: str
    features: PatientFeatures


class TrainRequest(BaseModel):
    models: Optional[List[str]] = ["all"]
