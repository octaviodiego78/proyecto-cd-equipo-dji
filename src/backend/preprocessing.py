"""
Preprocessing utilities for gold price prediction
Reusable functions for both training and inference
"""

import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(gold_df, sp500_df):
    """
    Merge and prepare gold and S&P 500 data
    
    Args:
        gold_df: DataFrame with gold prices (Date, Close, High, Low, Open, Volume)
        sp500_df: DataFrame with S&P 500 prices (Date, Close, High, Low, Open, Volume)
        
    Returns:
        Merged and sorted DataFrame
    """
    # Ensure Date columns are datetime
    gold_df['Date'] = pd.to_datetime(gold_df['Date'])
    sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
    
    # Merge on Date
    df = pd.merge(gold_df, sp500_df, on='Date', suffixes=('_gold', '_sp500'))
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def create_features(df):
    """
    Create engineered features: lags, moving averages, volatility
    This must match EXACTLY what was done during training
    
    Args:
        df: DataFrame with merged gold and S&P 500 data
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Feature engineering: lags and moving averages (matching train_pipeline.py lines 61-69)
    for lag in range(1, 3):
        df[f'sp500_close_lag{lag}'] = df['Close_sp500'].shift(lag)
        df[f'gold_close_lag{lag}'] = df['Close_gold'].shift(lag)
    
    df['gold_ma_5'] = df['Close_gold'].rolling(window=5).mean()
    df['sp500_ma_5'] = df['Close_sp500'].rolling(window=5).mean()
    
    df['gold_volatility_5'] = df['Close_gold'].pct_change().rolling(window=5).std()
    df['sp500_returns_lag1'] = df['Close_sp500'].pct_change().shift(1)
    
    # Drop NaN values
    df = df.dropna().reset_index(drop=True)
    
    return df


def get_feature_columns(df):
    """
    Get feature column names (excluding target and non-feature columns)
    This matches train_pipeline.py line 77
    
    Args:
        df: DataFrame with all columns
        
    Returns:
        List of feature column names
    """
    exclude_cols = ['Date', 'Close_gold', 'Close_sp500', 'High_gold', 'Low_gold', 
                   'Open_gold', 'Volume_gold', 'High_sp500', 'Low_sp500', 
                   'Open_sp500', 'Volume_sp500']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    return feature_cols


def save_scaler(scaler, filepath='data/processed/scaler.pkl'):
    """
    Save StandardScaler to pickle file
    
    Args:
        scaler: Fitted StandardScaler object
        filepath: Path to save the scaler
    """
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {filepath}")


def load_scaler(filepath='data/processed/scaler.pkl'):
    """
    Load StandardScaler from pickle file
    
    Args:
        filepath: Path to the saved scaler
        
    Returns:
        StandardScaler object
    """
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {filepath}")
    return scaler


def save_feature_columns(feature_cols, filepath='data/processed/feature_columns.json'):
    """
    Save feature column names to JSON file
    
    Args:
        feature_cols: List of feature column names
        filepath: Path to save the feature columns
    """
    with open(filepath, 'w') as f:
        json.dump({'feature_columns': feature_cols}, f, indent=2)
    print(f"Feature columns saved to {filepath}")


def load_feature_columns(filepath='data/processed/feature_columns.json'):
    """
    Load feature column names from JSON file
    
    Args:
        filepath: Path to the saved feature columns
        
    Returns:
        List of feature column names
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    feature_cols = data['feature_columns']
    print(f"Feature columns loaded from {filepath}")
    return feature_cols


def save_model_metadata(model_type, filepath='data/processed/model_metadata.json'):
    """
    Save model metadata (type, etc.) to JSON file
    
    Args:
        model_type: Type of model (MLP, CNN, LSTM)
        filepath: Path to save the metadata
    """
    metadata = {
        'model_type': model_type,
        'model_name': 'workspace.default.equipo_dji_gold_prediction_model'
    }
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to {filepath}")


def load_model_metadata(filepath='data/processed/model_metadata.json'):
    """
    Load model metadata from JSON file
    
    Args:
        filepath: Path to the saved metadata
        
    Returns:
        Dictionary with model metadata
    """
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    print(f"Model metadata loaded from {filepath}")
    return metadata


def prepare_features_for_prediction(gold_df, sp500_df, scaler, feature_cols):
    """
    Complete preprocessing pipeline for prediction
    
    Args:
        gold_df: DataFrame with gold prices
        sp500_df: DataFrame with S&P 500 prices
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        
    Returns:
        Scaled feature array for the most recent date (as numpy array)
        Latest date (as pandas Timestamp)
    """
    # Merge and prepare data
    df = load_and_prepare_data(gold_df, sp500_df)
    
    # Create features
    df = create_features(df)
    
    if len(df) == 0:
        raise ValueError("No data available after feature engineering")
    
    # Get the most recent row
    latest_row = df.iloc[-1:].copy()
    
    # Extract features in correct order
    X = latest_row[feature_cols].values
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Return date as pandas Timestamp (not numpy datetime64)
    latest_date = pd.Timestamp(latest_row['Date'].iloc[0])
    
    return X_scaled, latest_date

