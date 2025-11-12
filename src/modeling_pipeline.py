import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import time
from datetime import datetime

load_dotenv()

np.random.seed(42)
tf.random.set_seed(42)

# MLflow configuration
# Update this with your Databricks user email
EXPERIMENT_NAME = "/Users/diegooctavioperez21@gmail.com/gold_predictions"

# Set tracking URI to Databricks
mlflow.set_tracking_uri("databricks")
print("Using Databricks MLflow tracking")

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


def prepare_train_test_split(df):
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close_gold', 'Close_sp500', 'High_gold', 'Low_gold', 'Open_gold', 'Volume_gold', 'High_sp500', 'Low_sp500', 'Open_sp500', 'Volume_sp500']]
    
    X = df[feature_cols].values
    y = df['Close_gold'].values
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
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
        layers.Conv1D(filters, kernel_size, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(filters//2, kernel_size, activation='relu'),
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


def train_and_log_model(model, X_train, y_train, X_test, y_test, model_name, params, epochs=50, batch_size=16):
    print(f"Training {model_name} ({epochs} epochs)...")
    with mlflow.start_run(run_name=model_name) as run:
        start_time = time.time()
        
        # Train without validation - just fit on training data
        print(f"Starting fit()...")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
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
        
        print(f"{model_name} complete (MAPE: {mape:.4f}%, Time: {training_time:.1f}s)")
        return run.info.run_id, mape


def train_mlp_baseline(X_train, y_train, X_test, y_test):
    model = build_mlp(X_train.shape[1])
    params = {
        "model": "MLP",
        "layer1_size": 64,
        "layer2_size": 32,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 16
    }
    return train_and_log_model(model, X_train, y_train, X_test, y_test, "MLP_baseline", params)


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
        "batch_size": 16
    }
    return train_and_log_model(model, X_train_cnn, y_train, X_test_cnn, y_test, "CNN_baseline", params)


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
        "batch_size": 16
    }
    return train_and_log_model(model, X_train_lstm, y_train, X_test_lstm, y_test, "LSTM_baseline", params)


def hyperopt_mlp(X_train, y_train, X_test, y_test):
    space = {
        'layer1_size': hp.quniform('layer1_size', 32, 128, 16),
        'layer2_size': hp.quniform('layer2_size', 16, 64, 8),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01))
    }
    
    def objective(params):
        print(f"Hyperopt iteration...")
        start_time = time.time()
        with mlflow.start_run(nested=True):
            model = build_mlp(
                X_train.shape[1],
                int(params['layer1_size']),
                int(params['layer2_size']),
                params['dropout_rate'],
                params['learning_rate']
            )
            
            # Train without validation
            model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
            
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
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
        return best


def hyperopt_cnn(X_train, y_train, X_test, y_test):
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    space = {
        'filters': hp.quniform('filters', 16, 64, 8),
        'kernel_size': hp.quniform('kernel_size', 2, 5, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01))
    }
    
    def objective(params):
        print(f"Hyperopt iteration (CNN)...")
        start_time = time.time()
        with mlflow.start_run(nested=True):
            model = build_cnn(
                X_train.shape[1],
                int(params['filters']),
                int(params['kernel_size']),
                params['learning_rate']
            )
            
            # Train without validation
            model.fit(X_train_cnn, y_train, epochs=50, batch_size=16, verbose=0)
            
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
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
        return best


def hyperopt_lstm(X_train, y_train, X_test, y_test):
    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    space = {
        'units1': hp.quniform('units1', 16, 64, 8),
        'units2': hp.quniform('units2', 8, 32, 4),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01))
    }
    
    def objective(params):
        print(f"Hyperopt iteration (LSTM)...")
        start_time = time.time()
        with mlflow.start_run(nested=True):
            model = build_lstm(
                X_train.shape[1],
                int(params['units1']),
                int(params['units2']),
                params['dropout_rate'],
                params['learning_rate']
            )
            
            # Train without validation
            model.fit(X_train_lstm, y_train, epochs=50, batch_size=16, verbose=0)
            
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
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
        return best


def select_best_model():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.test_mape ASC"],
        max_results=1
    )
    
    if len(runs) > 0:
        best_run = runs.iloc[0]
        return best_run['run_id'], best_run['metrics.test_mape']
    return None, None


def register_model(run_id, model_name="equipo_dji_gold_prediction_model"):
    model_uri = f"runs:/{run_id}/model"
    
    registered_model = mlflow.register_model(model_uri, model_name)
    
    client = mlflow.tracking.MlflowClient()
    
    model_version = registered_model.version
    
    description = f"""Gold Price Prediction Model
- Data: gold_data.csv + sp500.csv (2020-09-28 to present)
- Target: Next-day gold closing price
- Features: Previous day SP500 close + gold price history
- Date: {datetime.now().strftime('%Y-%m-%d')}
- Team: Equipo DJI
- Changelog: v{model_version} - Optimized model with hyperparameter tuning
"""
    
    client.update_model_version(
        name=model_name,
        version=model_version,
        description=description
    )
    
    client.set_registered_model_alias(model_name, "champion", model_version)
    
    return registered_model


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
    
    print("\nSelecting best model...")
    best_run_id, best_mape = select_best_model()
    
    if best_run_id:
        print(f"Best model run_id: {best_run_id}")
        print(f"Best model MAPE: {best_mape:.4f}%")
        
        print("\nRegistering best model...")
        registered_model = register_model(best_run_id)
        print(f"Model registered: {registered_model.name} v{registered_model.version}")
        print(f"Model URI: models:/{registered_model.name}/{registered_model.version}")
        print(f"Alias 'champion' set to version {registered_model.version}")
    else:
        print("No finished runs found")
    
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()

