import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Property Price Prediction API")

try:
    with open("rf_model_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    model = artifacts["model"]
    model_columns = artifacts["model_columns"]
    categorical_cols = artifacts.get("categorical_cols", [])
    mae_metrics = artifacts["mae_metrics"]
except FileNotFoundError:
    model, model_columns, categorical_cols, mae_metrics = None, [], [], {}

class PropertyFeatures(BaseModel):
    Area_SqFt: float
    Rooms: int
    Build_Year: int
    Location: str
    Street_Type: str
    Furnishing: str
    Property_Type: str
    Has_Pool: int

@app.get("/metrics/mae")
def get_mae_metrics():
    if not mae_metrics:
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    return mae_metrics

@app.post("/predict")
def predict_price(features: PropertyFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        # Convert incoming JSON features to a DataFrame
        input_data = pd.DataFrame([features.model_dump()])
        
        # Explicitly apply get_dummies using the tracked categorical source columns
        input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
        
        # Enforce exact column schema alignment with the trained model structure
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        prediction = model.predict(input_aligned)
        return {"predicted_price": round(float(prediction), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
