import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from mlflow.models import infer_signature
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import time
from datetime import datetime
from prefect import flow, task

# Add project root to Python path
# train_pipeline.py is in src/pipelines/, so we need to go up 3 levels to reach project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import preprocessing utilities
from src.backend.preprocessing import (
    save_scaler, 
    save_feature_columns, 
    save_model_metadata
)

load_dotenv()

np.random.seed(42)
tf.random.set_seed(42)

# MLflow configuration
# Update this with your Databricks user email
EXPERIMENT_NAME = "/Users/diegooctavioperez21@gmail.com/gold_predictions"

# Set tracking URI to Databricks
mlflow.set_tracking_uri("databricks")
# Unity Catalog is the default for workspaces with legacy registry disabled
mlflow.set_registry_uri("databricks-uc")
print("Using Databricks MLflow tracking with Unity Catalog")

# Create or get experiment (auto-creates if doesn't exist)
experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
print(f"Experiment: {EXPERIMENT_NAME}")

# Enable TensorFlow autologging
mlflow.tensorflow.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True,
    checkpoint=False,
    checkpoint_save_best_only=False,
    checkpoint_save_weights_only=False,
    checkpoint_monitor="val_loss"
)


@task(name="Load and Prepare Data", log_prints=True)
def load_and_prepare_data():
    gold = pd.read_csv('data/raw/gold_data.csv')
    sp500 = pd.read_csv('data/raw/sp500.csv')
    
    gold['Date'] = pd.to_datetime(gold['Date'], format='%d/%m/%Y')
    sp500['Date'] = pd.to_datetime(sp500['Date'], format='%d/%m/%Y')
    
    df = pd.merge(gold, sp500, on='Date', suffixes=('_gold', '_sp500'))
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Use all available data
    print(f"Training with data from {df['Date'].min().date()} to {df['Date'].max().date()} ({len(df)} days)")
    
    # Feature engineering: lags and moving averages
    for lag in range(1, 3):
        df[f'sp500_close_lag{lag}'] = df['Close_sp500'].shift(lag)
        df[f'gold_close_lag{lag}'] = df['Close_gold'].shift(lag)
    
    df['gold_ma_5'] = df['Close_gold'].rolling(window=5).mean()
    df['sp500_ma_5'] = df['Close_sp500'].rolling(window=5).mean()
    
    df['gold_volatility_5'] = df['Close_gold'].pct_change().rolling(window=5).std()
    df['sp500_returns_lag1'] = df['Close_sp500'].pct_change().shift(1)
    
    df = df.dropna().reset_index(drop=True)
    
    return df


@task(name="Prepare Train/Test Split", log_prints=True)
def prepare_train_test_split(df):
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close_gold', 'Close_sp500', 'High_gold', 'Low_gold', 'Open_gold', 'Volume_gold', 'High_sp500', 'Low_sp500', 'Open_sp500', 'Volume_sp500']]
    
    X = df[feature_cols].values
    y = df['Close_gold'].values
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {len(feature_cols)}")
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def build_mlp(input_dim, layer1_size=64, layer2_size=32, dropout_rate=0.2, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(layer1_size, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(layer2_size, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate), loss='mse')
    return model


def build_cnn(input_dim, filters=32, kernel_size=3, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(filters, kernel_size, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(filters//2, kernel_size, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate), loss='mse')
    return model


def build_lstm(input_dim, units1=32, units2=16, dropout_rate=0.2, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.LSTM(units1, return_sequences=True, dropout=dropout_rate),
        layers.LSTM(units2, dropout=dropout_rate),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate), loss='mse')
    return model


def train_and_log_model(model, X_train, y_train, X_test, y_test, model_name, params, epochs=50, batch_size=32):
    print(f"Training {model_name} ({epochs} epochs)...")
    with mlflow.start_run(run_name=model_name) as run:
        start_time = time.time()
        
        # Adjust batch size if needed
        effective_batch_size = min(batch_size, len(X_train) // 4)  # At least 4 batches
        if effective_batch_size < 1:
            effective_batch_size = len(X_train)
        
        # Train without validation - just fit on training data
        print(f"Starting fit() with batch_size={effective_batch_size}...")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=effective_batch_size, verbose=2)
        print(f"Fit complete!")
        
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # Log custom metrics not covered by autolog
        mape = calculate_mape(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metric("test_mape", mape)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("training_time", training_time)
        
        # Log custom tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("data_version", "v1")
        mlflow.set_tag("team", "Equipo DJI")
        
        # Log custom params (autolog doesn't capture hyperopt params)
        mlflow.log_params(params)
        
        # Log predictions as artifact
        predictions_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
        predictions_df.to_csv(f'predictions_{model_name}.csv', index=False)
        mlflow.log_artifact(f'predictions_{model_name}.csv')
        os.remove(f'predictions_{model_name}.csv')
        
        # Infer signature from training data and predictions (required for Unity Catalog)
        signature = infer_signature(X_train, y_pred)
        
        # Manually log the model with signature to ensure it's saved (required for Unity Catalog)
        mlflow.tensorflow.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train[:5] if len(X_train) >= 5 else X_train
        )
        
        print(f"{model_name} complete (MAPE: {mape:.4f}%, Time: {training_time:.1f}s)")
        return run.info.run_id, mape


@task(name="Train MLP Baseline", log_prints=True)
def train_mlp_baseline(X_train, y_train, X_test, y_test):
    model = build_mlp(X_train.shape[1])
    params = {
        "model": "MLP",
        "layer1_size": 64,
        "layer2_size": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32
    }
    return train_and_log_model(model, X_train, y_train, X_test, y_test, "MLP_baseline", params, batch_size=32)


@task(name="Train CNN Baseline", log_prints=True)
def train_cnn_baseline(X_train, y_train, X_test, y_test):
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = build_cnn(X_train.shape[1])
    params = {
        "model": "CNN",
        "filters": 32,
        "kernel_size": 3,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32
    }
    return train_and_log_model(model, X_train_cnn, y_train, X_test_cnn, y_test, "CNN_baseline", params, batch_size=32)


@task(name="Train LSTM Baseline", log_prints=True)
def train_lstm_baseline(X_train, y_train, X_test, y_test):
    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = build_lstm(X_train.shape[1])
    params = {
        "model": "LSTM",
        "units1": 32,
        "units2": 16,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32
    }
    return train_and_log_model(model, X_train_lstm, y_train, X_test_lstm, y_test, "LSTM_baseline", params, batch_size=32)


@task(name="Hyperopt MLP", log_prints=True)
def hyperopt_mlp(X_train, y_train, X_test, y_test):
    # Reduced search space for faster training
    space = {
        'layer1_size': hp.quniform('layer1_size', 48, 96, 16),  # Reduced from 32-128
        'layer2_size': hp.quniform('layer2_size', 24, 48, 8),  # Reduced from 16-64
        'dropout_rate': hp.uniform('dropout_rate', 0.15, 0.35),  # Reduced from 0.1-0.5
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0005), np.log(0.005))  # Narrowed range
    }
    
    def objective(params):
        print(f"Hyperopt iteration...")
        start_time = time.time()
        with mlflow.start_run(nested=True):
            # Disable autologging for hyperopt runs to avoid interference with nested runs
            mlflow.tensorflow.autolog(disable=True)
            
            model = build_mlp(
                X_train.shape[1],
                int(params['layer1_size']),
                int(params['layer2_size']),
                params['dropout_rate'],
                params['learning_rate']
            )
            
            # Adjust batch size safely
            batch_size = min(32, max(16, len(X_train) // 10))
            
            # Reduced epochs for faster hyperopt iterations
            model.fit(X_train, y_train, epochs=20, batch_size=batch_size, verbose=0)
            
            y_pred = model.predict(X_test, verbose=0).flatten()
            mape = calculate_mape(y_test, y_pred)
            
            # Log custom params and metrics
            mlflow.log_params(params)
            mlflow.log_metric("test_mape", mape)
            mlflow.set_tag("tuning_iteration", "hyperopt")
            
            training_time = time.time() - start_time
            print(f"MLP iteration complete (MAPE: {mape:.4f}%, Time: {training_time:.1f}s)")
            return {'loss': mape, 'status': STATUS_OK}
    
    with mlflow.start_run(run_name="MLP_hyperopt"):
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)  # Reduced from 20 to 5
        return best


@task(name="Hyperopt CNN", log_prints=True)
def hyperopt_cnn(X_train, y_train, X_test, y_test):
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Reduced search space for faster training
    space = {
        'filters': hp.quniform('filters', 24, 48, 8),  # Reduced from 16-64
        'kernel_size': hp.quniform('kernel_size', 2, 3, 1),  # Reduced from 2-4
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0005), np.log(0.005))  # Narrowed range
    }
    
    def objective(params):
        print(f"Hyperopt iteration (CNN)...")
        start_time = time.time()
        with mlflow.start_run(nested=True):
            # Disable autologging for hyperopt runs to avoid interference with nested runs
            mlflow.tensorflow.autolog(disable=True)
            
            model = build_cnn(
                X_train.shape[1],
                int(params['filters']),
                int(params['kernel_size']),
                params['learning_rate']
            )
            
            # Adjust batch size safely
            batch_size = min(32, max(16, len(X_train) // 10))
            
            # Reduced epochs for faster hyperopt iterations
            model.fit(X_train_cnn, y_train, epochs=20, batch_size=batch_size, verbose=0)
            
            y_pred = model.predict(X_test_cnn, verbose=0).flatten()
            mape = calculate_mape(y_test, y_pred)
            
            # Log custom params and metrics
            mlflow.log_params(params)
            mlflow.log_metric("test_mape", mape)
            mlflow.set_tag("tuning_iteration", "hyperopt")
            
            training_time = time.time() - start_time
            print(f"CNN iteration complete (MAPE: {mape:.4f}%, Time: {training_time:.1f}s)")
            return {'loss': mape, 'status': STATUS_OK}
    
    with mlflow.start_run(run_name="CNN_hyperopt"):
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)  # Reduced from 20 to 5
        return best


@task(name="Hyperopt LSTM", log_prints=True)
def hyperopt_lstm(X_train, y_train, X_test, y_test):
    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Reduced search space for faster training
    space = {
        'units1': hp.quniform('units1', 24, 48, 8),  # Reduced from 16-64
        'units2': hp.quniform('units2', 12, 24, 4),  # Reduced from 8-32
        'dropout_rate': hp.uniform('dropout_rate', 0.15, 0.35),  # Reduced from 0.1-0.5
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0005), np.log(0.005))  # Narrowed range
    }
    
    def objective(params):
        print(f"Hyperopt iteration (LSTM)...")
        start_time = time.time()
        with mlflow.start_run(nested=True):
            # Disable autologging for hyperopt runs to avoid interference with nested runs
            mlflow.tensorflow.autolog(disable=True)
            
            model = build_lstm(
                X_train.shape[1],
                int(params['units1']),
                int(params['units2']),
                params['dropout_rate'],
                params['learning_rate']
            )
            
            # Adjust batch size safely
            batch_size = min(32, max(16, len(X_train) // 10))
            
            # Reduced epochs for faster hyperopt iterations
            model.fit(X_train_lstm, y_train, epochs=20, batch_size=batch_size, verbose=0)
            
            y_pred = model.predict(X_test_lstm, verbose=0).flatten()
            mape = calculate_mape(y_test, y_pred)
            
            # Log custom params and metrics
            mlflow.log_params(params)
            mlflow.log_metric("test_mape", mape)
            mlflow.set_tag("tuning_iteration", "hyperopt")
            
            training_time = time.time() - start_time
            print(f"LSTM iteration complete (MAPE: {mape:.4f}%, Time: {training_time:.1f}s)")
            return {'loss': mape, 'status': STATUS_OK}
    
    with mlflow.start_run(run_name="LSTM_hyperopt"):
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)  # Reduced from 20 to 5
        return best


@task(name="Select Best Model", log_prints=True)
def select_best_model():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    # Filter to only parent runs (not nested) that have model_type tag and models logged
    # Nested hyperopt runs don't have model_type tag and don't log models
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED' AND tags.model_type != ''",
        order_by=["metrics.test_mape ASC"],
        max_results=1
    )
    
    if len(runs) > 0:
        best_run = runs.iloc[0]
        return best_run['run_id'], best_run['metrics.test_mape']
    return None, None


@task(name="Register Champion Model", log_prints=True)
def register_model_champion(run_id):
    """Register the best model to MLflow"""

    model_name = f"workspace.default.equipo_dji_gold_prediction_model"
    
    print(f"Attempting to register to: {model_name}")
    
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    client = mlflow.tracking.MlflowClient()
    model_version = registered_model.version
    
    description = f"""Best model from hyperparameter optimization"""
    
    client.update_model_version(name=model_name, version=model_version, description=description)
    
    client.set_registered_model_alias(name=model_name, alias="champion", version=model_version)
    
    return registered_model


@task(name="Save Artifacts", log_prints=True)
def save_artifacts_task(scaler, feature_cols, best_model_name):
    """
    Save scaler, feature columns, and model metadata to data/processed/
    
    Args:
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        best_model_name: Name of the best model (MLP, CNN, or LSTM)
    """
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save scaler
    save_scaler(scaler, 'data/processed/scaler.pkl')
    
    # Save feature columns
    save_feature_columns(feature_cols, 'data/processed/feature_columns.json')
    
    # Save model metadata (extract model type from name)
    model_type = best_model_name.split('_')[0] if '_' in best_model_name else best_model_name
    save_model_metadata(model_type, 'data/processed/model_metadata.json')
    
    print("All artifacts saved successfully!")
    return True


@flow(name="Gold Price Prediction Training Pipeline", log_prints=True)
def main():
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("Preparing train/test split...")
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_train_test_split(df)
    
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    print("\nTraining baseline models...")
    print("Training MLP...")
    mlp_run_id, mlp_mape = train_mlp_baseline(X_train, y_train, X_test, y_test)
    print(f"MLP MAPE: {mlp_mape:.4f}%")
    
    print("Training CNN...")
    cnn_run_id, cnn_mape = train_cnn_baseline(X_train, y_train, X_test, y_test)
    print(f"CNN MAPE: {cnn_mape:.4f}%")
    
    print("Training LSTM...")
    lstm_run_id, lstm_mape = train_lstm_baseline(X_train, y_train, X_test, y_test)
    print(f"LSTM MAPE: {lstm_mape:.4f}%")
    
    print("\nStarting hyperparameter tuning...")
    print("Tuning MLP...")
    best_mlp = hyperopt_mlp(X_train, y_train, X_test, y_test)
    print(f"Best MLP params: {best_mlp}")
    
    print("Tuning CNN...")
    best_cnn = hyperopt_cnn(X_train, y_train, X_test, y_test)
    print(f"Best CNN params: {best_cnn}")
    
    print("Tuning LSTM...")
    best_lstm = hyperopt_lstm(X_train, y_train, X_test, y_test)
    print(f"Best LSTM params: {best_lstm}")
    
    
    best_run_id, best_mape = select_best_model()
    
    if best_run_id:
        print(f"\nBest Model - Run ID: {best_run_id} | MAPE: {best_mape:.4f}%")
        registered = register_model_champion(best_run_id)
        print(f"Registered as: {registered.name} v{registered.version} (alias: champion)")
        
        # Get the best model name from MLflow
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"status = 'FINISHED' AND tags.run_id = '{best_run_id}'",
            max_results=1
        )
        best_model_name = runs.iloc[0].get('tags.model_type', 'MLP') if len(runs) > 0 else 'MLP'
        
        # Save artifacts
        save_artifacts_task(scaler, feature_cols, best_model_name)
        
        print(f"\nPipeline completed successfully!")
        print(f"Model: {registered.name} v{registered.version}")
        print(f"Model type: {best_model_name}")
        print(f"Artifacts saved to data/processed/")
    else:
        print("\nNo finished runs found")


if __name__ == "__main__":
    main()

