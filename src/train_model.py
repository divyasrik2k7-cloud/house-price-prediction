"""Train regression models and save fitted pipelines."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import load_dataset, split_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
VIS_DIR = ROOT / "visualizations"


def build_models(random_state=42):
    """Return named, reproducible model pipelines."""
    return {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(random_state=random_state, max_depth=12)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1,
                min_samples_leaf=2
            )),
        ]),
    }


def regression_metrics(y_true, y_pred):
    """Compute standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_save():
    """Train models, save pipelines and return metrics and test data."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    results = {}
    predictions = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = regression_metrics(y_test, y_pred)
        predictions[name] = y_pred
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return {
        "results": results, "X_test": X_test, "y_test": y_test,
        "predictions": predictions, "feature_names": list(X.columns)
    }


def main():
    data = train_and_save()
    print("\nModel evaluation (held-out test set)")
    print("-" * 76)
    print(f'{"Model":<22} {"MAE":>10} {"MSE":>12} {"RMSE":>10} {"R²":>10}')
    for name, m in data["results"].items():
        print(f'{name:<22} {m["MAE"]:>10.4f} {m["MSE"]:>12.4f} {m["RMSE"]:>10.4f} {m["R2"]:>10.4f}')
    print("\nSaved models to models/ and metrics to outputs/metrics.json")
    print("Metrics are computed by running this script; values are not hard-coded.")


if __name__ == "__main__":
    main()
