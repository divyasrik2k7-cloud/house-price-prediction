# Claude prompt — enhance my existing project

You are a senior full-stack engineer and machine-learning engineer. Work directly with my existing repository named **House Price Prediction Using Regression Techniques**. I want a complete, runnable academic Data Mining project with a modern professional dashboard, using **React + Vite frontend** and **FastAPI + scikit-learn backend**.

## Preserve first
1. Inspect the repository before changing anything. Reuse existing files and code wherever sound; do not blindly replace or duplicate the project.
2. Preserve the California Housing dataset (`sklearn.datasets.fetch_california_housing`) and these baseline regressors: Linear Regression, Decision Tree Regression, and Random Forest Regression.
3. Preserve real evaluation using MAE, MSE, RMSE, and R² on a consistent held-out test set. Never hard-code or fabricate metric values.
4. Keep the existing preprocessing, training, prediction, evaluation, documentation, and test code when possible. If architecture changes are required, explain them and retain compatible functionality.
5. Keep secrets out of source control. Add sensible `.gitignore` rules.

## Build the following
- Modern professional responsive dashboard with sidebar navigation, summary cards, charts, clear loading/error states, API health indicator, and a light/dark theme.
- Model comparison page for all three regressors with real MAE, MSE, RMSE, and R².
- Single prediction form with validated inputs and side-by-side predictions from all three models.
- Prediction history stored locally in the browser, including timestamp, inputs, and results; allow loading an old prediction and clearing history.
- CSV batch prediction: validate required feature columns, numeric values, empty input and a configurable safe row limit; return a downloadable CSV with outputs for all three models.
- Random Forest feature importance visualization with a clear caveat that impurity-based importance is not causal.
- Hyperparameter tuning using a bounded `GridSearchCV` for Decision Tree and Random Forest. Report best parameters and separate test metrics; do not silently replace baseline model metrics.
- Five-fold cross-validation for all three models, showing mean and standard deviation for MAE and R².
- Dataset explorer with searchable preview, descriptive statistics, feature descriptions, and download options where useful.
- Downloadable evaluation report in JSON and model metrics in CSV.
- Input validation, API error handling, accessible labels, responsive layout, and clear setup instructions.
- Backend tests for health, validation, predictions (with network-aware handling), tuning errors, and batch CSV validation. Avoid tests that require expensive model training unless marked as integration tests.

## Dataset correctness
The California Housing dataset describes census block groups, not individual houses. The target is median area house value in units of $100,000. Do not imply the app can estimate a particular home's precise current sale price. Display this caveat in the UI and report. Explain that the dataset is historical and not current market data.

## Engineering requirements
- Frontend: React 18, Vite, Axios, Recharts, Lucide React.
- Backend: FastAPI, Pydantic, pandas, NumPy, scikit-learn.
- Use sklearn Pipelines for preprocessing and estimators to prevent leakage.
- Use deterministic train/test split (`random_state=42`) for baseline evaluation.
- Keep API URLs configurable through `VITE_API_URL`, defaulting to local FastAPI.
- Do not send localStorage history to the backend.
- Avoid unnecessary new dependencies.
- Make all commands work on macOS and include Windows alternatives where relevant.
- No mock predictions or fake metrics in the production UI. Show a friendly error when the API or dataset is unavailable.
- Add `docs/FEATURE_SPECIFICATION.md`, update README, and document architecture, methodology, endpoints, limitations, and how to run tests.

## Required deliverables
1. Complete runnable source code for frontend and backend.
2. `requirements.txt` and `package.json` with scripts.
3. Automated tests.
4. Setup/run instructions and API endpoint documentation.
5. Feature specification.
6. Keep generated model binaries, virtual environments, node_modules, and dataset cache out of the ZIP/repository.
7. After editing, install dependencies if the environment permits, run backend tests, run a production frontend build, and fix errors. Clearly disclose anything you could not run; never claim tests passed unless you actually ran them.
8. Provide the final result as a ZIP of the complete repository, plus a concise change summary.

Start by listing the current file tree and identifying existing functionality. Then implement incrementally, run checks, and summarize changed files and exact commands to run the application.
