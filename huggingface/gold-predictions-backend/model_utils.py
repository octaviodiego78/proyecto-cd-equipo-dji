"""
Utilities for loading models from Databricks Model Registry
"""

import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_champion_model(model_name="workspace.default.equipo_dji_gold_prediction_model"):
    """
    Load the champion model from MLflow Model Registry
    
    Args:
        model_name: Name of the model in Unity Catalog
        
    Returns:
        TensorFlow model
    """
    # Configure MLflow
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")  # Unity Catalog
    
    # Load model using Champion alias
    model_uri = f"models:/{model_name}@champion"
    
    print(f"Loading model from: {model_uri}")
    model = mlflow.tensorflow.load_model(model_uri)
    print(f"Model loaded successfully")
    
    return model


def load_model_by_version(model_name, version):
    """
    Load a specific version of the model
    
    Args:
        model_name: Name of the model in Unity Catalog
        version: Version number to load
        
    Returns:
        TensorFlow model
    """
    # Configure MLflow
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    
    # Load specific version
    model_uri = f"models:/{model_name}/{version}"
    
    print(f"Loading model from: {model_uri}")
    model = mlflow.tensorflow.load_model(model_uri)
    print(f"Model version {version} loaded successfully")
    
    return model


def load_model_by_alias(model_name, alias):
    """
    Load a model by alias (e.g., 'champion', 'challenger')
    
    Args:
        model_name: Name of the model in Unity Catalog
        alias: Alias of the model version
        
    Returns:
        TensorFlow model
    """
    # Configure MLflow
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    
    # Load model by alias
    model_uri = f"models:/{model_name}@{alias}"
    
    print(f"Loading model from: {model_uri}")
    model = mlflow.tensorflow.load_model(model_uri)
    print(f"Model with alias '{alias}' loaded successfully")
    
    return model

