"""Data loading, training, evaluation, prediction, and tuning helpers."""
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

FEATURE_LABELS = {
    "MedInc": "Median income (in $10,000s)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "Block-group population",
    "AveOccup": "Average household occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}

MODEL_NAMES = ["Linear Regression", "Decision Tree", "Random Forest"]


@lru_cache(maxsize=1)
def dataset():
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.data[FEATURES].copy()
    y = bunch.target.copy()

    df = X.copy()
    df["MedHouseVal"] = y
    return X, y, df


def make_models():
    return {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
    }


def safe_predict(model, X):
    """Use einsum for Linear Regression predictions."""
    if (
        isinstance(model, Pipeline)
        and isinstance(model.named_steps.get("model"), LinearRegression)
    ):
        transformed = model[:-1].transform(X)
        regressor = model.named_steps["model"]

        predictions = (
            np.einsum(
                "ij,j->i",
                np.asarray(transformed),
                np.asarray(regressor.coef_),
            )
            + regressor.intercept_
        )
        return predictions

    return model.predict(X)


@lru_cache(maxsize=1)
def trained():
    X, y, _ = dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = make_models()
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = safe_predict(model, X_test)

        mse = mean_squared_error(y_test, pred)
        metrics[name] = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, pred)),
        }

    return models, X_train, X_test, y_train, y_test, metrics


def metrics_payload():
    _, _, _, _, y_test, metrics = trained()
    _, _, df = dataset()

    return {
        "metrics": metrics,
        "test_rows": int(len(y_test)),
        "train_rows": int(len(df) - len(y_test)),
        "dataset_rows": int(len(df)),
        "target_units": "$100,000 units",
        "model_names": MODEL_NAMES,
        "note": (
            "California Housing targets are area-level median values, "
            "not individual home valuations."
        ),
    }


def predict_row(payload):
    models, *_ = trained()
    row = pd.DataFrame(
        [{feature: payload[feature] for feature in FEATURES}],
        columns=FEATURES,
    )

    results = {}
    for name, model in models.items():
        value = float(safe_predict(model, row)[0])
        results[name] = {
            "value": value,
            "usd_estimate": value * 100000,
        }

    return results


def feature_importance():
    models, *_ = trained()
    values = models["Random Forest"].named_steps["model"].feature_importances_

    return sorted(
        [
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "importance": float(value),
            }
            for feature, value in zip(FEATURES, values)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )


def cross_validation():
    X, y, _ = dataset()
    results = {}

    # Evaluate each fold explicitly so Linear Regression uses safe_predict.
    for name, model in make_models().items():
        fold_mae = []
        fold_r2 = []

        from sklearn.model_selection import KFold

        cv = KFold(n_splits=5, shuffle=False)

        for train_indices, test_indices in cv.split(X):
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]

            model.fit(X_train, y_train)
            predictions = safe_predict(model, X_test)

            fold_mae.append(mean_absolute_error(y_test, predictions))
            fold_r2.append(r2_score(y_test, predictions))

        results[name] = {
            "folds": 5,
            "mae_mean": float(np.mean(fold_mae)),
            "mae_std": float(np.std(fold_mae)),
            "r2_mean": float(np.mean(fold_r2)),
            "r2_std": float(np.std(fold_r2)),
        }

    return results


def tune_model(model_name):
    if model_name not in ("Decision Tree", "Random Forest"):
        raise ValueError(
            "Tuning is supported for Decision Tree and Random Forest."
        )

    X, y, _ = dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_name == "Decision Tree":
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(random_state=42)),
        ])
        params = {
            "model__max_depth": [6, 10, 14],
            "model__min_samples_leaf": [1, 2, 4],
        }
    else:
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=80,
                n_jobs=-1,
                random_state=42,
            )),
        ])
        params = {
            "model__max_depth": [12, None],
            "model__min_samples_leaf": [1, 2],
        }

    search = GridSearchCV(
        estimator,
        params,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = safe_predict(best_model, X_test)
    mse = mean_squared_error(y_test, predictions)

    return {
        "model": model_name,
        "best_params": search.best_params_,
        "cv_mae": float(-search.best_score_),
        "test_metrics": {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, predictions)),
        },
        "note": (
            "Tuning result is reported separately; baseline model "
            "comparison remains unchanged."
        ),
    }


def dataset_summary():
    _, _, df = dataset()

    sample = df.head(100).replace({np.nan: None}).to_dict(
        orient="records"
    )
    stats = (
        df.describe()
        .round(3)
        .reset_index()
        .rename(columns={"index": "statistic"})
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    return {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "features": [
            {"name": feature, "label": FEATURE_LABELS[feature]}
            for feature in FEATURES
        ],
        "summary": stats,
        "sample": sample,
    }
