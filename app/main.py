from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

# --- Input schema ---
class InsuranceData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# --- Load model ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "insurance_model.pkl"

model = joblib.load(MODEL_PATH)

# --- Prediction route ---
@app.post("/predict")
def predict(data: InsuranceData):
    # convert input into model format (order must match your model)
    X = [[
        data.age,
        1 if data.sex == "male" else 0,
        data.bmi,
        data.children,
        1 if data.smoker == "yes" else 0,
        {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}[data.region]
    ]]

    pred = model.predict(X)[0]
    return {"prediction": float(pred)}
