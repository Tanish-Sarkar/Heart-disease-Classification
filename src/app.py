# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from src.predict import predict_from_json

app = FastAPI(title="Heart Disease Prediction API")

class PredictRequest(BaseModel):
    data: Dict[str, Any]  # e.g., {"age": 54, "cholesterol": 233, ...}

class PredictResponse(BaseModel):
    prediction: Any
    probabilities: Any = None

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    res = predict_from_json(req.data)
    return res

# Run with: uvicorn src.app:app --reload --port 8000
