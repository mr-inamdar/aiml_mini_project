import json
import os
import pytest
from sklearn.pipeline import Pipeline

from model.predict import predict_demand, load_artifacts

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "bike_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_schema.json")
IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.json")


def test_model_artifacts_exist():
    """Verify that all required model artifacts were generated and saved."""
    assert os.path.exists(MODEL_PATH), "bike_model.joblib is missing"
    assert os.path.exists(METRICS_PATH), "metrics.json is missing"
    assert os.path.exists(SCHEMA_PATH), "feature_schema.json is missing"
    assert os.path.exists(IMPORTANCE_PATH), "feature_importance.json is missing"


def test_model_pipeline_loads_properly():
    """Verify that the model pipeline can be loaded and has expected components."""
    pipeline, schema = load_artifacts()
    assert isinstance(pipeline, Pipeline), "Loaded model is not a scikit-learn Pipeline"
    assert "preprocessor" in pipeline.named_steps
    assert "regressor" in pipeline.named_steps


def test_metrics_json_structure_and_values():
    """Verify that metrics.json contains evaluations for all 3 candidate models."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    assert "candidate_models" in metrics_data
    candidates = metrics_data["candidate_models"]

    for model_name in ["Linear Regression", "Decision Tree", "Random Forest"]:
        assert model_name in candidates, f"{model_name} missing from metrics"
        assert "mae" in candidates[model_name]
        assert "rmse" in candidates[model_name]
        assert "r2" in candidates[model_name]
        assert "prediction_score_pct" in candidates[model_name]
        assert candidates[model_name]["r2"] > 0, f"R2 score should be positive for {model_name}"
        assert 0 <= candidates[model_name]["prediction_score_pct"] <= 100

    best_model = metrics_data["best_model"]
    assert best_model["name"] == "Random Forest"
    assert candidates["Random Forest"]["r2"] > candidates["Linear Regression"]["r2"]


def test_feature_schema_columns():
    """Verify that the feature schema lists the 10 operational features."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_data = json.load(f)

    expected_features = [
        "season", "mnth", "hr", "weekday", "weathersit",
        "temp", "hum", "windspeed", "holiday", "workingday"
    ]
    for feat in expected_features:
        assert feat in schema_data["features"]

    # Target leakage verification
    assert "casual" not in schema_data["features"]
    assert "registered" not in schema_data["features"]
    assert "instant" not in schema_data["features"]
    assert "dteday" not in schema_data["features"]


def test_prediction_inference_valid_sample():
    """Verify inference prediction returns a non-negative integer rental count."""
    sample_input = {
        "season": 2,
        "mnth": 5,
        "hr": 8,
        "holiday": 0,
        "weekday": 2,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.55,
        "hum": 0.45,
        "windspeed": 0.15
    }

    result = predict_demand(sample_input)
    assert result["status"] == "success"
    assert len(result["predictions"]) == 1
    pred = result["predictions"][0]
    assert isinstance(pred, int)
    assert pred >= 0


def test_prediction_missing_features_raises_error():
    """Verify that inference raises a ValueError when required columns are absent."""
    incomplete_input = {
        "season": 2,
        "hr": 8
        # Missing all other required features
    }
    with pytest.raises(ValueError):
        predict_demand(incomplete_input)
