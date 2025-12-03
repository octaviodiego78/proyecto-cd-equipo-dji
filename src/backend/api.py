"""
FastAPI service for Gold Price Prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import numpy as np
from datetime import datetime, timedelta

from model_utils import load_champion_model
from preprocessing import (
    load_scaler,
    load_feature_columns,
    load_model_metadata,
    prepare_features_for_prediction
)
from data_fetcher import fetch_live_data_for_prediction


# Initialize FastAPI app
app = FastAPI(
    title="Gold Price Prediction API",
    description="API for predicting tomorrow's gold price using live data from Yahoo Finance",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model, scaler, and metadata
model = None
scaler = None
feature_cols = None
model_metadata = None


class PredictionRequest(BaseModel):
    """Request model for prediction"""
    predict_tomorrow: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "predict_tomorrow": True
            }
        }


class PredictionResponse(BaseModel):
    """Response model with prediction results"""
    prediction: float
    predicted_date: str
    today_date: str
    model_name: str
    model_type: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1950.75,
                "predicted_date": "2024-12-03",
                "today_date": "2024-12-02",
                "model_name": "workspace.default.equipo_dji_gold_prediction_model",
                "model_type": "MLP"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    feature_cols_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Load model and preprocessing artifacts on startup"""
    global model, scaler, feature_cols, model_metadata
    
    try:
        print("Loading model from MLflow...")
        model = load_champion_model()
        
        print("Loading scaler...")
        scaler_path = os.getenv("SCALER_PATH", "data/processed/scaler.pkl")
        scaler = load_scaler(scaler_path)
        
        print("Loading feature columns...")
        feature_cols_path = os.getenv("FEATURE_COLS_PATH", "data/processed/feature_columns.json")
        feature_cols = load_feature_columns(feature_cols_path)
        
        print("Loading model metadata...")
        metadata_path = os.getenv("MODEL_METADATA_PATH", "data/processed/model_metadata.json")
        model_metadata = load_model_metadata(metadata_path)
        
        print("All components loaded successfully!")
        
    except Exception as e:
        print(f"Error loading components: {e}")
        print("API will not be able to make predictions until components are loaded.")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if (model is not None and scaler is not None and feature_cols is not None) else "unhealthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        feature_cols_loaded=feature_cols is not None
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest = None):
    """
    Predict tomorrow's gold price
    
    Flow:
    1. Fetch recent gold and S&P 500 prices from Yahoo Finance
    2. Preprocess and create features
    3. Scale features
    4. Reshape for model type (MLP, CNN, LSTM)
    5. Make prediction
    6. Return tomorrow's predicted price
    """
    # Check if all components are loaded
    if model is None or scaler is None or feature_cols is None:
        raise HTTPException(
            status_code=500,
            detail="Model, scaler, or feature columns not loaded. Check server logs."
        )
    
    try:
        # Fetch live data from Yahoo Finance (last 30 days to ensure enough for lags/MA)
        print("Fetching live data from Yahoo Finance...")
        gold_df, sp500_df = fetch_live_data_for_prediction(days=30)
        
        # Prepare features for prediction
        print("Preparing features...")
        X_scaled, latest_date = prepare_features_for_prediction(
            gold_df, sp500_df, scaler, feature_cols
        )
        
        # Reshape based on model type
        model_type = model_metadata.get('model_type', 'MLP') if model_metadata else 'MLP'
        print(f"Using model type: {model_type}")
        
        if model_type in ['CNN', 'LSTM']:
            # Reshape for CNN/LSTM: (batch_size, features, 1)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        else:
            # MLP uses flat features
            X_reshaped = X_scaled
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(X_reshaped, verbose=0).flatten()[0]
        
        # Calculate tomorrow's date
        today_date = latest_date
        tomorrow_date = today_date + timedelta(days=1)
        
        # Format dates as strings
        today_str = today_date.strftime('%Y-%m-%d')
        tomorrow_str = tomorrow_date.strftime('%Y-%m-%d')
        
        print(f"Prediction: ${prediction:.2f} for {tomorrow_str}")
        
        return PredictionResponse(
            prediction=float(prediction),
            predicted_date=tomorrow_str,
            today_date=today_str,
            model_name=model_metadata.get('model_name', 'workspace.default.equipo_dji_gold_prediction_model'),
            model_type=model_type
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

