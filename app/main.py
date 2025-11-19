from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

model = joblib.load("app/models/insurance_model.pkl")
FEATURES = joblib.load("app/models/insurance_features.pkl")


def render_page(request: Request, prediction: float | None = None,
                form_values: dict | None = None, error: str | None = None) -> HTMLResponse:
    context = {
        "request": request,
        "prediction": prediction,
        "form_values": form_values or {},
        "error": error,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return render_page(request)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  age: float = Form(...),
                  sex: str = Form(...),
                  bmi: float = Form(...),
                  children: int = Form(...),
                  smoker: str = Form(...),
                  region: str = Form(...)):

    form_values = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }

    try:
        features_dict = {
            "age": float(age),
            "bmi": float(bmi),
            "children": int(children),
            "sex_male": 1 if sex.lower() == "male" else 0,
            "smoker_yes": 1 if smoker.lower() == "yes" else 0,
            "region_northwest": 1 if region == "northwest" else 0,
            "region_southeast": 1 if region == "southeast" else 0,
            "region_southwest": 1 if region == "southwest" else 0,
        }

        feature_row = [features_dict.get(name, 0) for name in FEATURES]
        features_array = np.array([feature_row])

        prediction = float(model.predict(features_array)[0])
        prediction = round(prediction, 2)
        return render_page(request, prediction=prediction, form_values=form_values)
    except Exception:
        error_message = "We couldn't generate a prediction right now. Please verify the inputs and try again."
        return render_page(request, form_values=form_values, error=error_message)


if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)
