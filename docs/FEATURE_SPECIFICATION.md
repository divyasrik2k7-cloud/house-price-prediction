# Enhanced Feature Specification
## House Price Prediction Using Regression Techniques

### 1. Objective
Upgrade the existing academic regression project into a polished full-stack application with a React frontend and FastAPI backend. Preserve the California Housing dataset and the baseline algorithms: Linear Regression, Decision Tree Regression, and Random Forest Regression.

### 2. Architecture
- **Frontend:** React 18, Vite, Recharts, Lucide React, Axios.
- **Backend:** FastAPI, Pydantic, scikit-learn, pandas, NumPy.
- **Model layer:** sklearn pipelines, shared 80/20 split, random state 42, held-out evaluation.
- **State:** Browser localStorage for prediction history.
- **API:** JSON endpoints plus CSV upload/download and report downloads.

### 3. Feature requirements
1. **Modern professional dashboard:** responsive sidebar, metric cards, charts, loading/error states, API status, theme toggle.
2. **Model comparison:** show MAE, MSE, RMSE and R² for all three baseline regressors, computed from actual held-out predictions.
3. **Prediction workspace:** accept the eight numeric dataset features and show predictions from all three models. The user can select a model to highlight its output.
4. **Prediction history:** store up to 100 prediction events in browser localStorage; allow viewing a past input/result and clearing local history.
5. **Batch prediction:** validate CSV extension, required columns, row count (max 5,000), numeric data, and return a downloadable CSV with predictions from all models.
6. **Feature importance:** show Random Forest impurity-based feature importance with caveat that it is not causal and may have biases.
7. **Hyperparameter tuning:** bounded GridSearchCV for Decision Tree and Random Forest; report best parameters and separate validation/test metrics. Baseline model comparison is not silently overwritten.
8. **Cross-validation:** five-fold CV for all baseline regressors with mean and standard deviation of MAE and R².
9. **Reports:** download JSON containing model metrics, feature importance, CV results; download CSV containing baseline model metrics.
10. **Dataset explorer:** show record count, column names, descriptive statistics, and searchable first 100 records.
11. **Input validation and transparency:** range validation, API errors, dataset caveat, target units, no fabricated metric values.
12. **Tests:** health endpoint, prediction request behavior, and invalid tuning model validation.

### 4. API endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | API health |
| GET | `/api/metrics` | Baseline metrics and dataset split information |
| POST | `/api/predict` | Three model predictions for one feature vector |
| POST | `/api/batch-predict` | CSV upload; CSV download response |
| GET | `/api/importance` | Random Forest feature importance |
| POST | `/api/tune` | Grid search for a tree-based model |
| GET | `/api/cross-validation` | Five-fold CV summary |
| GET | `/api/dataset` | Dataset preview and descriptive statistics |
| GET | `/api/report.json` | Full evaluation report |
| GET | `/api/report.csv` | Baseline metric table |

### 5. Dataset interpretation and limitations
The California Housing dataset consists of census block-group measurements, not individual property listings. Its target is median house value expressed in units of $100,000, based on historical data. The USD display is the target value multiplied by 100,000 for easier interpretation; it is not an inflation-adjusted current price or a professional appraisal. Feature importance is model-specific and not causal.

### 6. Acceptance criteria
- Backend launches using the README instructions.
- Frontend launches using Vite and displays responsive pages.
- Model metrics are calculated by scikit-learn, not hard-coded.
- Prediction uses all three saved-in-memory baseline models.
- Batch endpoint checks columns and row limit.
- CV and tuning invoke real scikit-learn routines.
- Report endpoints return downloadable files.
- Tests pass in an environment with dependencies installed and dataset access.
- Limitations are visible to users in the UI and documentation.
