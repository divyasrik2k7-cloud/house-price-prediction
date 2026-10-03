from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import io, json
from .ml import FEATURES, metrics_payload, predict_row, feature_importance, cross_validation, tune_model, dataset_summary

app = FastAPI(title="House Price Prediction API", version="2.0.0",
              description="Regression models using the California Housing dataset.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PredictionInput(BaseModel):
    MedInc: float = Field(ge=0, le=20)
    HouseAge: float = Field(ge=0, le=100)
    AveRooms: float = Field(gt=0, le=100)
    AveBedrms: float = Field(gt=0, le=30)
    Population: float = Field(ge=0, le=100000)
    AveOccup: float = Field(gt=0, le=1000)
    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)

@app.get("/api/health")
def health():
    return {"status":"ok","service":"House Price Prediction API"}

@app.get("/api/metrics")
def get_metrics():
    try: return metrics_payload()
    except Exception as exc: raise HTTPException(503, f"Could not load/train dataset and models: {exc}")

@app.post("/api/predict")
def predict(data: PredictionInput):
    try: return {"predictions":predict_row(data.model_dump()),"input":data.model_dump(),
                 "note":"Area-level estimate, not an individual property valuation."}
    except Exception as exc: raise HTTPException(503,str(exc))

@app.get("/api/importance")
def importance():
    try: return {"features":feature_importance()}
    except Exception as exc: raise HTTPException(503,str(exc))

@app.get("/api/cross-validation")
def cv():
    try: return {"results":cross_validation(),"folds":5}
    except Exception as exc: raise HTTPException(503,str(exc))

@app.post("/api/tune")
def tune(body: dict):
    try: return tune_model(body.get("model","Decision Tree"))
    except ValueError as exc: raise HTTPException(400,str(exc))
    except Exception as exc: raise HTTPException(503,str(exc))

@app.get("/api/dataset")
def explore_dataset():
    try: return dataset_summary()
    except Exception as exc: raise HTTPException(503,str(exc))

@app.get("/api/report.json")
def report_json():
    try:
        report={"project":"House Price Prediction Using Regression Techniques","dataset":"California Housing",
                "metrics":metrics_payload(),"feature_importance":feature_importance(),"cross_validation":cross_validation()}
        return StreamingResponse(io.BytesIO(json.dumps(report,indent=2).encode()),media_type="application/json",
            headers={"Content-Disposition":'attachment; filename="house_price_evaluation_report.json"'})
    except Exception as exc: raise HTTPException(503,str(exc))

@app.get("/api/report.csv")
def report_csv():
    try:
        frame=pd.DataFrame(metrics_payload()["metrics"]).T.reset_index(names="model")
        stream=io.StringIO(); frame.to_csv(stream,index=False)
        return StreamingResponse(io.BytesIO(stream.getvalue().encode()),media_type="text/csv",
            headers={"Content-Disposition":'attachment; filename="model_metrics.csv"'})
    except Exception as exc: raise HTTPException(503,str(exc))

@app.post("/api/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Upload a CSV file.")
    try:
        frame=pd.read_csv(io.BytesIO(await file.read()))
        missing=[f for f in FEATURES if f not in frame.columns]
        if missing: raise HTTPException(400,f"CSV is missing required columns: {', '.join(missing)}")
        if frame.empty: raise HTTPException(400,"CSV contains no rows.")
        if len(frame)>5000: raise HTTPException(400,"Maximum 5,000 rows per upload.")
        X=frame[FEATURES].apply(pd.to_numeric,errors="coerce")
        if X.isna().all(axis=1).any(): raise HTTPException(400,"Some rows contain no valid numeric feature values.")
        from .ml import trained
        models,*_=trained(); result=frame.copy()
        for name,model in models.items():
            pred=model.predict(X); safe=name.lower().replace(" ","_")
            result[f"{safe}_predicted_value_100k"]=pred
            result[f"{safe}_predicted_usd"]=pred*100000
        stream=io.StringIO(); result.to_csv(stream,index=False)
        return StreamingResponse(io.BytesIO(stream.getvalue().encode()),media_type="text/csv",
            headers={"Content-Disposition":'attachment; filename="batch_predictions.csv"'})
    except HTTPException: raise
    except Exception as exc: raise HTTPException(400,f"Could not process CSV: {exc}")
