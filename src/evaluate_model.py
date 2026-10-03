"""Generate evaluation plots and a readable model comparison table."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.train_model import train_and_save, VIS_DIR, OUTPUT_DIR


def create_plots(X_test, y_test, predictions, feature_names):
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Target distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(y_test, kde=True)
    plt.title("Distribution of Test-Set House Values")
    plt.xlabel("Median house value (units of $100,000)")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "target_distribution.png", dpi=160)
    plt.close()

    # Feature correlation
    frame = X_test.copy()
    frame["MedHouseVal"] = y_test
    plt.figure(figsize=(10, 7))
    sns.heatmap(frame.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Feature and Target Correlation (Test Subset)")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    # Actual vs predicted and residual plots
    for name, y_pred in predictions.items():
        safe = name.lower().replace(" ", "_")
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.35, s=12)
        low = min(float(y_test.min()), float(y_pred.min()))
        high = max(float(y_test.max()), float(y_pred.max()))
        plt.plot([low, high], [low, high], "k--", linewidth=1)
        plt.xlabel("Actual value")
        plt.ylabel("Predicted value")
        plt.title(f"Actual vs Predicted — {name}")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"{safe}_actual_vs_predicted.png", dpi=160)
        plt.close()

        residuals = y_test.to_numpy() - y_pred
        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.35, s=12)
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Predicted value")
        plt.ylabel("Residual (actual - predicted)")
        plt.title(f"Residual Plot — {name}")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"{safe}_residuals.png", dpi=160)
        plt.close()


def main():
    data = train_and_save()
    create_plots(data["X_test"], data["y_test"], data["predictions"], data["feature_names"])
    table = pd.DataFrame(data["results"]).T
    print("\nModel comparison:")
    print(table.round(4).to_string())
    print(f"\nPlots saved in: {VIS_DIR}")
    print(f"Metrics saved in: {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
