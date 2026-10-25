"""RideWise — Inference Service
==============================
Reusable inference helper to load the serialized pipeline, validate feature
inputs, and output demand predictions.

Guarantees:
- Exact feature name alignment and encodings matching the training pipeline.
- Non-negative integer predictions (count cannot physically be < 0).
- Accepts single dictionary input, list of dictionaries, or pandas DataFrame.
"""

import json
import os
from typing import Dict, Any, List, Union

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "bike_model.joblib")
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_schema.json")

_PIPELINE = None
_SCHEMA = None


def load_artifacts():
    """Load serialized model pipeline and feature schema lazily with caching."""
    global _PIPELINE, _SCHEMA
    if _PIPELINE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model artifact not found at '{MODEL_PATH}'. "
                "Please run 'python model/train_model.py' first."
            )
        _PIPELINE = joblib.load(MODEL_PATH)

    if _SCHEMA is None:
        if not os.path.exists(SCHEMA_PATH):
            raise FileNotFoundError(
                f"Schema file not found at '{SCHEMA_PATH}'. "
                "Please run 'python model/train_model.py' first."
            )
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            _SCHEMA = json.load(f)

    return _PIPELINE, _SCHEMA


def predict_demand(input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> Dict[str, Any]:
    """Generate bike rental predictions for provided input features.
    
    Args:
        input_data: Single dictionary of features, list of dictionaries, or DataFrame.
        
    Returns:
        Dict with keys:
            - 'predictions': List of integer predicted counts (clamped >= 0)
            - 'raw_predictions': List of float predicted values
            - 'status': 'success'
    """
    pipeline, schema = load_artifacts()
    expected_features = schema["features"]

    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_input = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    # Check for missing feature columns
    missing_cols = [col for col in expected_features if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required features: {missing_cols}")

    # Reorder columns strictly to match feature schema
    df_aligned = df_input[expected_features]

    # Predict using the trained pipeline
    raw_preds = pipeline.predict(df_aligned)

    # Post-process: count must be non-negative integer
    clamped_rounded_preds = [max(0, int(round(p))) for p in raw_preds]

    return {
        "status": "success",
        "predictions": clamped_rounded_preds,
        "raw_predictions": [round(float(p), 2) for p in raw_preds]
    }


if __name__ == "__main__":
    # Quick sanity check sample
    sample_input = {
        "season": 3,        # Fall
        "mnth": 7,          # July
        "hr": 17,           # 5 PM rush hour
        "holiday": 0,       # Not a holiday
        "weekday": 3,       # Wednesday
        "workingday": 1,    # Working day
        "weathersit": 1,    # Clear
        "temp": 0.72,       # ~29.5°C
        "hum": 0.50,        # 50% humidity
        "windspeed": 0.20   # ~13.4 km/h
    }
    
    print("Testing inference with sample input...")
    try:
        result = predict_demand(sample_input)
        print(f"Predicted rental count: {result['predictions'][0]} bikes")
    except Exception as e:
        print(f"Prediction error: {e}")
