from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")


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


@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "form_values": {}, "prediction": None, "error": None},
    )


@app.post("/", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    age: int = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: str = Form(...),
    region: str = Form(...),
):
    try:
        payload = InsuranceData(
            age=age,
            sex=sex,
            bmi=bmi,
            children=children,
            smoker=smoker,
            region=region,
        )
        vector = build_feature_vector(payload)
        prediction = model.predict(vector)[0]
        form_values = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "form_values": form_values,
                "prediction": round(float(prediction), 2),
                "error": None,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "form_values": dict(request.form()),
                "prediction": None,
                "error": f"Prediction failed: {exc}",
            },
        )
