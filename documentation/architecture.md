# System Architecture

```mermaid
flowchart TD
    A[California Housing dataset via scikit-learn] --> B[Data loading and train/test split]
    B --> C[Model pipelines: imputation + regressor]
    C --> D[Linear Regression]
    C --> E[Decision Tree Regression]
    C --> F[Random Forest Regression]
    D --> G[Evaluation metrics]
    E --> G
    F --> G
    G --> H[Metrics JSON and comparison]
    G --> I[Plots]
    C --> J[Saved model pipelines]
    J --> K[Prediction module / optional Streamlit app]
```

## Components
- `src/data_preprocessing.py`: loads data and creates reproducible train/test splits.
- `src/train_model.py`: defines, trains, evaluates, and saves model pipelines.
- `src/evaluate_model.py`: generates plots and displays a comparison.
- `src/predict.py`: loads a saved pipeline and predicts from feature values.
- `app.py`: optional Streamlit user interface.
