"""
End-to-end insurance workflow implemented as a pure Python script.

This converts the exploratory/cleaning/modeling notebooks into repeatable
functions that can be executed with:

    python -m src.pipeline --stage all

Each stage can also be called independently (explore, clean, preprocess, train).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/insurance.csv")
CLEAN_DATA_PATH = Path("data/insurance_cleaned.csv")
PREPARED_DATA_PATH = Path("data/insurance_model_ready.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path("app/models")


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")
    return pd.read_csv(path)


def explore_data(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    describe_path = REPORTS_DIR / "describe.csv"
    df.describe().to_csv(describe_path)
    print(f"[explore] Summary statistics saved to {describe_path}")

    dtypes_path = REPORTS_DIR / "dtypes.csv"
    df.dtypes.to_frame("dtype").to_csv(dtypes_path)
    print(f"[explore] Column data types saved to {dtypes_path}")

    corr = df.corr(numeric_only=True)
    corr_path = REPORTS_DIR / "correlation.csv"
    corr.to_csv(corr_path)

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
    except ImportError:
        print(
            "[explore] matplotlib/seaborn not installed; skipped heatmap + categorical plots "
            "(install them if you need the visuals)."
        )
    else:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        corr_fig = FIGURES_DIR / "correlation_matrix.png"
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(corr_fig)
        plt.close()
        print(f"[explore] Correlation heatmap saved to {corr_fig}")

        categorical_fig = FIGURES_DIR / "categorical_distribution.png"
        plt.figure(figsize=(12, 4))
        for i, col in enumerate(["sex", "smoker", "region"], start=1):
            plt.subplot(1, 3, i)
            sns.countplot(data=df, x=col)
            plt.title(col.title())
        plt.tight_layout()
        plt.savefig(categorical_fig)
        plt.close()
        print(f"[explore] Categorical distribution plot saved to {categorical_fig}")

    for col in ["sex", "smoker", "region"]:
        counts_path = REPORTS_DIR / f"{col}_value_counts.csv"
        df[col].value_counts().to_csv(counts_path, header=["count"])
        print(f"[explore] Value counts for {col} saved to {counts_path}")


def clean_data(df: pd.DataFrame, *, output_path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    cleaned = df.drop_duplicates().copy()
    cleaned.to_csv(output_path, index=False)
    print(f"[clean] Cleaned dataset with {len(cleaned)} rows saved to {output_path}")
    return cleaned


def prepare_features(df: pd.DataFrame, *, output_path: Path = PREPARED_DATA_PATH) -> pd.DataFrame:
    processed = df.copy()
    processed["sex"] = processed["sex"].map({"male": 1, "female": 0})
    processed["smoker"] = processed["smoker"].map({"yes": 1, "no": 0})
    processed = pd.get_dummies(processed, columns=["region"], drop_first=True)
    processed.to_csv(output_path, index=False)
    print(f"[preprocess] Model-ready dataset saved to {output_path}")
    return processed


def _train_candidate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0),
        "Lasso(alpha=0.001)": Lasso(alpha=0.001),
        "RandomForest(n=300)": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting(n=500)": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    metrics: Dict[str, Dict[str, float]] = {}
    trained_models = {}

    for name, estimator in candidates.items():
        model = clone(estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)
        metrics[name] = {"mae": float(mae), "rmse": rmse, "r2": float(r2)}
        trained_models[name] = model

    return metrics, trained_models


def train_models(df: pd.DataFrame, *, models_dir: Path = MODELS_DIR) -> Dict[str, Dict[str, float]]:
    X = df.drop("charges", axis=1)
    y = df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics, trained_models = _train_candidate_models(X_train, X_test, y_train, y_test)
    best_model_name = max(metrics, key=lambda item: metrics[item]["r2"])
    best_model = trained_models[best_model_name]

    models_dir.mkdir(parents=True, exist_ok=True)
    feature_order = X_train.columns.tolist()
    joblib.dump(best_model, models_dir / "insurance_model.pkl")
    joblib.dump(feature_order, models_dir / "insurance_features.pkl")

    metrics_payload = {"model_name": best_model_name, **metrics[best_model_name]}
    with (models_dir / "insurance_metrics.json").open("w") as fp:
        json.dump(metrics_payload, fp, indent=2)

    print(f"[train] Saved best model ({best_model_name}) and metrics to {models_dir}")
    return metrics


def ensure_clean_dataset() -> pd.DataFrame:
    if CLEAN_DATA_PATH.exists():
        return pd.read_csv(CLEAN_DATA_PATH)
    return clean_data(load_raw_dataset())


def ensure_prepared_dataset() -> pd.DataFrame:
    if PREPARED_DATA_PATH.exists():
        return pd.read_csv(PREPARED_DATA_PATH)
    clean_df = ensure_clean_dataset()
    return prepare_features(clean_df)


def run_pipeline(stage: str) -> None:
    stage = stage.lower()
    valid_stages = {"explore", "clean", "preprocess", "train", "all"}
    if stage not in valid_stages:
        raise ValueError(f"Unknown stage '{stage}'. Options: {sorted(valid_stages)}")

    df_raw = load_raw_dataset()
    if stage in {"explore", "all"}:
        explore_data(df_raw)
        if stage == "explore":
            return

    clean_df = df_raw
    if stage == "all":
        clean_df = clean_data(df_raw)
    elif stage in {"clean"}:
        clean_data(df_raw)
        return
    else:
        clean_df = ensure_clean_dataset()

    if stage in {"preprocess", "all"}:
        processed_df = prepare_features(clean_df)
        if stage == "preprocess":
            return
    else:
        processed_df = ensure_prepared_dataset()

    if stage in {"train", "all"}:
        metrics = train_models(processed_df)
        print("[train] Metrics by model:")
        for name, stats in metrics.items():
            print(
                f"  - {name}: MAE={stats['mae']:.2f} | RMSE={stats['rmse']:.2f} | R2={stats['r2']:.3f}"
            )


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Insurance data workflow (pure Python).")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["explore", "clean", "preprocess", "train", "all"],
        help="Run only a specific stage of the workflow.",
    )
    return parser.parse_args(args=args)


def main() -> None:
    args = parse_args()
    run_pipeline(args.stage)


if __name__ == "__main__":
    main()
