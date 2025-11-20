# Smart Insurance

**Live Demo:** https://smart-insurance-app.onrender.com

Smart Insurance is an end-to-end project that prepares the classic medical-cost dataset, trains a scikit-learn regression pipeline, and serves a polished FastAPI experience where visitors can explore premium scenarios, view projections, and reach the advisory team directly.

## Repository Layout

```
.
├── app/                    # FastAPI application (server, templates, CSS, and model artifacts)
│   ├── main.py             # API + inference endpoint
│   ├── models/             # Trained pipeline, feature list, and metrics JSON
│   ├── static/             # Global stylesheet for the Smart Insurance UI
│   └── templates/          # index.html rendered for GET/POST
├── data/                   # Raw and cleaned insurance datasets (Git tracked for convenience)
├── notebooks/              # Jupyter notebooks for EDA, cleaning, and modeling experiments
├── requirements.txt        # Shared dependencies for the notebooks and API
├── src/                    # (Reserved) python modules for future automation
└── README.md
```

## Quick Start

> Commands assume macOS/Linux. Use `python` vs `python3` depending on your shell.

1. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Launch the Smart Insurance web app**
   ```bash
   uvicorn app.main:app --reload
   ```
   Visit `http://127.0.0.1:8000/` to try the estimator. The "Projection summary" card updates in-place and the contact section mirrors the marketing site flow your teacher demonstrated.

4. **(Optional) Regenerate the model artifacts**
   - Use the notebooks in `notebooks/` (`01_exploration` → `04_modeling`) to refresh the dataset and model.
   - Export the trained pipeline and metrics to `app/models/` so the FastAPI service picks them up automatically. The app reads `insurance_model.pkl` and the `insurance_features.pkl` feature order at startup.

## Data Workflow

| Stage | Notebook / Script | Output |
| ----- | ----------------- | ------ |
| Exploration | `notebooks/01_exploration.ipynb` | sanity checks, visualizations, and target distribution |
| Cleaning | `notebooks/02_preprocessing.ipynb`, `03_cleaning.ipynb` | `data/insurance_cleaned.csv` with normalized column names and filtered outliers |
| Modeling | `notebooks/04_modeling.ipynb` | serialized linear regression pipeline + feature list + `insurance_metrics.json` |

### Engineered Features
- `sex_male`, `smoker_yes`, `region_northwest`, `region_southeast`, `region_southwest` one-hot encoded from the categorical fields.
- Numeric fields (`age`, `bmi`, `children`) scaled/validated for impossible entries.
- Charges column retained only for training; the web app predicts it from the remaining fields.

## Model & Metrics

| Model | R² | MAE (USD) | RMSE (USD) | Artifact |
| ----- | -- | --------- | ---------- | -------- |
| Linear Regression (scikit-learn) | 0.807 | 4,177 | 5,956 | `app/models/insurance_model.pkl` |

`app/models/insurance_metrics.json` mirrors the table above and is generated at training time.

## Web Application

- **Tech stack** – FastAPI + Jinja templates + vanilla CSS.
- **Form experience** – Responsive layout, segmented controls for smoker status, contextual helper text, and contact CTA modeled after the teacher's "House Maxify" site.
- **Inference flow** – POST `/predict` builds a feature vector from cleaned form inputs, feeds it to the trained pipeline, and redisplays the page with persisted values and formatted prediction.
- **Error handling** – Graceful message + preserved inputs whenever inference fails.
- **Static assets** – `app/static/styles.css` centralizes the Smart Insurance palette, navigation, hero benefits, and contact card animations.

## Datasets & Artifacts

- `data/insurance.csv` – Original Kaggle insurance dataset (age, sex, bmi, children, smoker, region, charges).
- `data/insurance_cleaned.csv` – Notebook-generated version with trimmed BMI/charge outliers and consistent casing.
- `app/models/insurance_model.pkl` – Latest regression model loaded by FastAPI.
- `app/models/insurance_features.pkl` – Ordered list of training features used during inference.
- `app/models/insurance_metrics.json` – Stored R²/MAE/RMSE for quick reporting.

## Development Tips & Troubleshooting

- If you retrain the model, ensure the new feature list matches the order expected by the API (update `insurance_features.pkl` alongside `insurance_model.pkl`).
- Use `uvicorn app.main:app --reload` during development so template and CSS tweaks hot-reload.
- When editing the UI, keep the `app/static/styles.css` palette consistent; navigation anchors (`#overview`, `#estimate`, `#insights`, `#contact`) must remain in the template for the header links.
- The repository includes `requirements.txt` so GitHub or Codespaces can install dependencies automatically—use `pip install -r requirements.txt` before running notebooks.

## Next Steps & Ideas

1. Add automated tests around the `/predict` endpoint using `fastapi.testclient`.
2. Containerize the project (Docker + `docker-compose`) for reproducible deployments.
3. Publish the FastAPI app to Render, Railway, or Fly.io and wire the contact form to a CRM webhook.
4. Extend the notebooks with SHAP or permutation importance visuals to explain contribution per feature.

## Deployment (Render)

Deploying the app so others—like your instructor—can access it only requires a hosted `uvicorn` process. Render keeps the repo in sync automatically:

1. Push this repository to GitHub (already done).
2. Create an account at [Render](https://render.com) and click **New → Web Service**.
3. Connect your GitHub account, choose the `smart-insurance-app` repo, and fill in:
   - **Environment**: Python 3.12 (or latest available)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Click **Create Web Service**. Render installs dependencies, launches the server, and returns a public URL such as `https://smart-insurance-app.onrender.com`.
5. Add that URL to the top of this README (e.g., “Live demo: …”) once the deployment finishes.

The same steps work for Railway, Fly.io, or Azure App Service—just ensure the start command binds to `0.0.0.0` and reads the platform-provided port.

---

Questions or improvements? Open an issue or submit a pull request—contributions are welcome!
