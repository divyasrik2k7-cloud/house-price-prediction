# House Price Prediction Using Regression Techniques

A reproducible Data Mining project that trains and compares three regression models on the California Housing dataset:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

## Features
- Dataset loading with scikit-learn
- Reusable train/test split
- Model training and persistence with Joblib
- Evaluation using MAE, MSE, RMSE, and R²
- EDA, actual-vs-predicted, and residual plots
- Automated unit tests
- Optional Streamlit interface

> The California Housing dataset is downloaded by scikit-learn on first use, so internet access may be required the first time. The dataset is not committed to this repository.

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python -m src.train_model
python -m src.evaluate_model
```

Training writes model files to `models/`, metrics to `outputs/metrics.json`, and evaluation plots to `visualizations/`.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Optional web app

```bash
streamlit run app.py
```

Run training first so the saved model files are available.

## Project structure

```text
data/                 Dataset notes (dataset is fetched by scikit-learn)
notebooks/            Exploratory data analysis notebook
src/                  Preprocessing, training, prediction, evaluation
models/               Locally generated serialized models
outputs/              Locally generated evaluation metrics
visualizations/       Locally generated plots
tests/                Automated tests
documentation/        Architecture and methodology
app.py                Optional Streamlit prediction interface
requirements.txt      Python dependencies
```

## Methodology
1. Load the California Housing dataset.
2. Split it into training and test subsets with a fixed random seed.
3. Fit preprocessing and each regression model on training data only.
4. Evaluate each model on the same held-out test set.
5. Compare the models using MAE, MSE, RMSE, and R².

The target is the dataset's median house value, represented in units of $100,000. The dataset is a teaching dataset and should not be treated as a current property valuation service.
