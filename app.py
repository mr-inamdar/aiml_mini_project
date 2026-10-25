"""RideWise — Flask Application & Machine Learning API
===================================================
Web server providing the user interface dashboard, real-time demand prediction,
dataset analytics, model evaluation metrics, and system health checks.

Routes:
- GET  /              : Main dashboard interface
- GET  /api/health    : System and model availability status
- GET  /api/metrics   : Evaluation metrics from held-out benchmarking
- GET  /api/analytics : Aggregated statistical summaries from authentic dataset
- POST /api/predict   : Real-time ML inference for hourly bike rental demand
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request

from model.predict import load_artifacts, predict_demand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RideWise.App")

# Application Initialization
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Paths to assets
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "hour.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "bike_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
SCHEMA_PATH = os.path.join(BASE_DIR, "model", "feature_schema.json")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "model", "feature_importance.json")

# In-memory cached analytics, schema, and actual vs predicted evaluation slice
_CACHED_ANALYTICS: Optional[Dict[str, Any]] = None
_FEATURE_SCHEMA: Optional[Dict[str, Any]] = None
_CACHED_ACTUAL_PRED: Optional[List[Dict[str, Any]]] = None


def get_feature_schema() -> Dict[str, Any]:
    """Retrieve and cache feature schema definitions."""
    global _FEATURE_SCHEMA
    if _FEATURE_SCHEMA is None:
        if not os.path.exists(SCHEMA_PATH):
            raise FileNotFoundError(f"Feature schema not found at {SCHEMA_PATH}")
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            _FEATURE_SCHEMA = json.load(f)
    return _FEATURE_SCHEMA


def compute_dataset_analytics() -> Dict[str, Any]:
    """Compute and cache descriptive statistical summaries from authentic dataset."""
    global _CACHED_ANALYTICS
    if _CACHED_ANALYTICS is not None:
        return _CACHED_ANALYTICS

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset hour.csv not found at {DATASET_PATH}")

    logger.info("Computing dataset analytics from %s...", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    # 1. Overall Summary
    total_records = int(len(df))
    avg_demand = round(float(df["cnt"].mean()), 2)
    max_demand = int(df["cnt"].max())
    min_demand = int(df["cnt"].min())
    std_demand = round(float(df["cnt"].std()), 2)

    # 2. Hourly Trends (Overall, Working Day, Weekend/Holiday)
    hourly_overall = df.groupby("hr")["cnt"].mean()
    hourly_working = df[df["workingday"] == 1].groupby("hr")["cnt"].mean()
    hourly_nonworking = df[df["workingday"] == 0].groupby("hr")["cnt"].mean()

    hourly_list = []
    for hr in range(24):
        hourly_list.append({
            "hr": hr,
            "overall_avg": round(float(hourly_overall.get(hr, 0)), 1),
            "workingday_avg": round(float(hourly_working.get(hr, 0)), 1),
            "nonworking_avg": round(float(hourly_nonworking.get(hr, 0)), 1)
        })

    # 3. Weekday Trends (0: Sun to 6: Sat)
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    weekday_mean = df.groupby("weekday")["cnt"].mean()
    weekday_list = [
        {"weekday": d, "day_name": day_names[d], "avg_demand": round(float(weekday_mean.get(d, 0)), 1)}
        for d in range(7)
    ]

    # 4. Seasonal Trends (1: Spring to 4: Winter)
    season_names = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    season_mean = df.groupby("season")["cnt"].mean()
    seasonal_list = [
        {"season": s, "season_name": season_names[s], "avg_demand": round(float(season_mean.get(s, 0)), 1)}
        for s in [1, 2, 3, 4]
    ]

    # 5. Weather Situation Trends (1: Clear to 4: Storm/Heavy Rain)
    weather_names = {
        1: "Clear / Few Clouds",
        2: "Mist / Cloudy",
        3: "Light Snow / Rain",
        4: "Heavy Rain / Storm / Snow"
    }
    weather_mean = df.groupby("weathersit")["cnt"].mean()
    weather_list = [
        {"weathersit": w, "weather_name": weather_names[w], "avg_demand": round(float(weather_mean.get(w, 0)), 1)}
        for w in [1, 2, 3, 4]
    ]

    _CACHED_ANALYTICS = {
        "status": "success",
        "summary": {
            "total_records": total_records,
            "avg_hourly_demand": avg_demand,
            "max_hourly_demand": max_demand,
            "min_hourly_demand": min_demand,
            "std_hourly_demand": std_demand
        },
        "hourly_trends": hourly_list,
        "weekday_trends": weekday_list,
        "seasonal_trends": seasonal_list,
        "weather_trends": weather_list
    }
    logger.info("Dataset analytics computed successfully.")
    return _CACHED_ANALYTICS


def validate_prediction_input(payload: Any) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Strictly validate input data types, ranges, categories, NaNs, and infinities.
    
    Returns:
        (is_valid, error_message, validated_clean_dict)
    """
    if not isinstance(payload, dict):
        return False, "Request body must be a valid JSON object.", None

    schema = get_feature_schema()
    expected_features = schema["features"]

    # 1. Check for missing required features
    missing_fields = [f for f in expected_features if f not in payload]
    if missing_fields:
        return False, f"Missing required feature(s): {', '.join(missing_fields)}", None

    cleaned: Dict[str, Any] = {}

    # 2. Integer Categorical / Binary Validations
    int_specs = {
        "season": (1, 4, [1, 2, 3, 4]),
        "mnth": (1, 12, list(range(1, 13))),
        "hr": (0, 23, list(range(0, 24))),
        "weekday": (0, 6, list(range(0, 7))),
        "holiday": (0, 1, [0, 1]),
        "workingday": (0, 1, [0, 1]),
        "weathersit": (1, 4, [1, 2, 3, 4])
    }

    for field, (min_val, max_val, allowed_set) in int_specs.items():
        val = payload[field]
        if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
            return False, f"Field '{field}' must be an integer, got: {type(val).__name__}", None

        # Disallow fractional values for integer fields
        if isinstance(val, float) and not val.is_integer():
            return False, f"Field '{field}' must be a whole integer, got decimal: {val}", None

        val_int = int(val)
        if val_int not in allowed_set:
            return (
                False,
                f"Field '{field}' must be one of {allowed_set} (received {val_int}).",
                None
            )
        cleaned[field] = val_int

    # 3. Numeric Float Validations (Normalized 0.0 to 1.0)
    float_specs = {
        "temp": "Normalized temperature (temp = celsius / 41)",
        "hum": "Normalized humidity (hum = percent / 100)",
        "windspeed": "Normalized wind speed (windspeed = kmh / 67)"
    }

    for field, desc in float_specs.items():
        val = payload[field]
        if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
            return False, f"Field '{field}' must be a numeric float, got: {type(val).__name__}", None

        # Check for NaN and Infinity
        if math.isnan(val) or math.isinf(val):
            return False, f"Field '{field}' cannot be NaN or Infinity.", None

        val_float = float(val)
        if val_float < 0.0 or val_float > 1.0:
            return (
                False,
                f"Field '{field}' must be between 0.0 and 1.0 ({desc}). Received: {val_float}",
                None
            )
        cleaned[field] = round(val_float, 4)

    return True, None, cleaned


# ==========================================
# Application Routes
# ==========================================

@app.route("/", methods=["GET"])
def index():
    """Serve the primary HTML5 dashboard."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Check system operational status and model readiness."""
    model_loaded = False
    model_error = None

    try:
        pipeline, _ = load_artifacts()
        model_loaded = pipeline is not None
    except Exception as e:
        model_error = str(e)
        logger.warning("Health check detected model issue: %s", e)

    dataset_exists = os.path.exists(DATASET_PATH)
    status_str = "healthy" if (model_loaded and dataset_exists) else "degraded"

    response = {
        "status": status_str,
        "model_loaded": model_loaded,
        "dataset_available": dataset_exists,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    if model_error:
        response["model_error"] = model_error

    return jsonify(response), 200


def get_actual_vs_predicted_sample() -> List[Dict[str, Any]]:
    """Generate and cache an authentic test-set evaluation sample of actual vs predicted demand."""
    global _CACHED_ACTUAL_PRED
    if _CACHED_ACTUAL_PRED is not None:
        return _CACHED_ACTUAL_PRED

    if not os.path.exists(DATASET_PATH) or not os.path.exists(MODEL_PATH):
        return []

    try:
        df = pd.read_csv(DATASET_PATH)
        # Test split corresponds chronologically to the final 20%
        test_start_idx = int(len(df) * 0.8)
        test_df = df.iloc[test_start_idx:].copy()

        # Extract a continuous 72-hour slice (3 calendar days) representing varied conditions
        sample_slice = test_df.iloc[100:172]
        result = predict_demand(sample_slice)
        preds = result["predictions"]

        day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        records = []
        for i, (_, row) in enumerate(sample_slice.iterrows()):
            hr = int(row["hr"])
            dteday = str(row["dteday"])
            wkday = day_names[int(row["weekday"])]
            records.append({
                "time_label": f"{dteday} {hr:02d}:00",
                "short_label": f"{wkday} {hr:02d}h",
                "actual": int(row["cnt"]),
                "predicted": int(preds[i]),
                "workingday": int(row["workingday"]),
                "temp_c": round(float(row["temp"]) * 41, 1),
                "weather": int(row["weathersit"])
            })

        _CACHED_ACTUAL_PRED = records
        return _CACHED_ACTUAL_PRED
    except Exception as e:
        logger.warning("Could not compute actual vs predicted evaluation sample: %s", e)
        return []


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Return held-out model evaluation metrics from metrics.json."""
    if not os.path.exists(METRICS_PATH):
        logger.error("Metrics artifact not found at %s", METRICS_PATH)
        return jsonify({
            "status": "error",
            "message": "Model metrics file is missing. Please run model training first."
        }), 503

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        # Include real feature importance from model artifacts if present
        if os.path.exists(IMPORTANCE_PATH):
            with open(IMPORTANCE_PATH, "r", encoding="utf-8") as f:
                metrics_data["feature_importance"] = json.load(f)

        # Include real actual-vs-predicted test evaluation sample
        actual_pred = get_actual_vs_predicted_sample()
        if actual_pred:
            metrics_data["actual_vs_predicted"] = actual_pred

        return jsonify(metrics_data), 200
    except Exception as e:
        logger.exception("Failed to read metrics file: %s", e)
        return jsonify({
            "status": "error",
            "message": f"Unable to read model metrics: {str(e)}"
        }), 500


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """Return statistical distribution summaries derived from dataset/hour.csv."""
    try:
        analytics = compute_dataset_analytics()
        return jsonify(analytics), 200
    except FileNotFoundError as e:
        logger.error("Dataset not found for analytics: %s", e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 503
    except Exception as e:
        logger.exception("Failed to generate dataset analytics: %s", e)
        return jsonify({
            "status": "error",
            "message": f"Internal error computing analytics: {str(e)}"
        }), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """Generate real-time hourly bike rental demand prediction."""
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Content-Type must be application/json."
        }), 400

    try:
        payload = request.get_json(silent=True)
    except Exception:
        return jsonify({
            "status": "error",
            "message": "Malformed JSON payload in request."
        }), 400

    if payload is None:
        return jsonify({
            "status": "error",
            "message": "Empty or invalid JSON body provided."
        }), 400

    # Strict Validation
    is_valid, error_msg, cleaned_data = validate_prediction_input(payload)
    if not is_valid:
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 400

    # Ensure model is ready
    try:
        result = predict_demand(cleaned_data)
        predicted_count = result["predictions"][0]
        raw_prediction = result["raw_predictions"][0]

        return jsonify({
            "status": "success",
            "predicted_demand": predicted_count,
            "raw_prediction": raw_prediction,
            "features_used": cleaned_data
        }), 200

    except FileNotFoundError as e:
        logger.error("Prediction failed because model artifact is missing: %s", e)
        return jsonify({
            "status": "error",
            "message": "Prediction model is currently unavailable. Ensure the model has been trained."
        }), 503
    except Exception as e:
        logger.exception("Inference error occurred: %s", e)
        return jsonify({
            "status": "error",
            "message": f"Inference execution failed: {str(e)}"
        }), 500


# ==========================================
# Error Handlers
# ==========================================

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"status": "error", "message": "API endpoint not found."}), 404
    return render_template("index.html"), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "status": "error",
        "message": f"Method {request.method} not allowed for {request.path}."
    }), 405


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error occurred."
    }), 500


# Warm up analytics and model cache upon module load if files are present
try:
    if os.path.exists(MODEL_PATH):
        load_artifacts()
    if os.path.exists(DATASET_PATH):
        compute_dataset_analytics()
except Exception as e:
    logger.warning("Startup pre-caching encountered an issue (will load on-demand): %s", e)


if __name__ == "__main__":
    logger.info("Starting RideWise Flask server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
