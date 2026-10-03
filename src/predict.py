"""Load a trained model and predict median house value."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_FILES = {
    "Linear Regression": "linear_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
}


def predict_one(features, model_name="Random Forest"):
    """Predict for one row; features may be a dict keyed by dataset feature names."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_FILES)}")
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python -m src.train_model` first."
        )
    row = pd.DataFrame([features])
    model = joblib.load(model_path)
    prediction = float(model.predict(row)[0])
    return prediction
