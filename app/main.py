from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd

app = FastAPI()


class InsuranceData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "insurance_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "insurance_features.pkl"

model = joblib.load(MODEL_PATH)
feature_order = joblib.load(FEATURES_PATH)
region_columns = [col for col in feature_order if col.startswith("region_")]


def build_feature_vector(payload: InsuranceData):
    """Mirror the training pipeline encoding so inference stays in sync."""
    normalized_sex = payload.sex.strip().lower()
    normalized_smoker = payload.smoker.strip().lower()
    normalized_region = payload.region.strip().lower()

    base_features = {
        "age": payload.age,
        "bmi": payload.bmi,
        "children": payload.children,
        "sex_male": 1 if normalized_sex == "male" else 0,
        "smoker_yes": 1 if normalized_smoker == "yes" else 0,
    }

    for col in region_columns:
        slug = col.replace("region_", "")
        base_features[col] = 1 if normalized_region == slug else 0

    ordered_features = {column: base_features.get(column, 0) for column in feature_order}
    return pd.DataFrame([ordered_features], columns=feature_order)


@app.post("/predict")
def predict(data: InsuranceData):
    vector = build_feature_vector(data)
    pred = model.predict(vector)[0]
    return {"prediction": float(pred)}
