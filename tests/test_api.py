import json
import pytest
from app import app


@pytest.fixture
def client():
    """Create Flask test client fixture with testing mode enabled."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_sample_payload():
    """Valid standard feature payload for prediction."""
    return {
        "season": 3,
        "mnth": 7,
        "hr": 17,
        "holiday": 0,
        "weekday": 3,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.72,
        "hum": 0.50,
        "windspeed": 0.20
    }


def test_index_route(client):
    """Verify GET / returns 200 and renders HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"RideWise" in response.data
    assert b"Forecast Rental Demand" in response.data


def test_health_check_endpoint(client):
    """Verify GET /api/health returns healthy operational status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["dataset_available"] is True


def test_metrics_endpoint(client):
    """Verify GET /api/metrics returns real model benchmark data."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "candidate_models" in data
    assert "best_model" in data
    assert "Random Forest" in data["candidate_models"]
    assert data["best_model"]["name"] == "Random Forest"


def test_analytics_endpoint(client):
    """Verify GET /api/analytics returns aggregations computed from dataset/hour.csv."""
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "summary" in data
    assert data["summary"]["total_records"] == 17379
    assert len(data["hourly_trends"]) == 24
    assert len(data["weekday_trends"]) == 7
    assert len(data["seasonal_trends"]) == 4
    assert len(data["weather_trends"]) == 4


def test_predict_success(client, valid_sample_payload):
    """Verify POST /api/predict returns valid non-negative predicted demand."""
    response = client.post(
        "/api/predict",
        data=json.dumps(valid_sample_payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "predicted_demand" in data
    assert isinstance(data["predicted_demand"], int)
    assert data["predicted_demand"] >= 0


def test_predict_missing_feature(client, valid_sample_payload):
    """Verify missing required feature returns 400 Bad Request."""
    del valid_sample_payload["hr"]
    response = client.post(
        "/api/predict",
        data=json.dumps(valid_sample_payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "Missing required feature" in data["message"]


def test_predict_invalid_category_range(client, valid_sample_payload):
    """Verify hour > 23 or month > 12 returns 400 Bad Request."""
    valid_sample_payload["hr"] = 25
    response = client.post(
        "/api/predict",
        data=json.dumps(valid_sample_payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "Field 'hr'" in data["message"]


def test_predict_numeric_out_of_range(client, valid_sample_payload):
    """Verify temp > 1.0 or hum < 0.0 returns 400 Bad Request."""
    valid_sample_payload["temp"] = 1.45
    response = client.post(
        "/api/predict",
        data=json.dumps(valid_sample_payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "Field 'temp'" in data["message"]


def test_predict_non_json_request(client, valid_sample_payload):
    """Verify non-JSON Content-Type returns 400 Bad Request."""
    response = client.post(
        "/api/predict",
        data="season=3&hr=17",
        content_type="application/x-www-form-urlencoded"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "Content-Type must be application/json" in data["message"]


def test_api_404_handler(client):
    """Verify undefined API route returns 404 with JSON structure."""
    response = client.get("/api/unknown_endpoint")
    assert response.status_code == 404
    data = response.get_json()
    assert data["status"] == "error"
