"""RideWise — Model Training & Evaluation Pipeline
==================================================
This module trains and compares multiple regression algorithms
(Linear Regression, Decision Tree, Random Forest) on the UCI Bike Sharing
dataset using a chronological train/test split.

Outputs saved to model/:
- bike_model.joblib          : Serialized scikit-learn Pipeline of the best model
- metrics.json               : Held-out evaluation metrics (MAE, RMSE, R2) for all models
- feature_schema.json        : Exact input features, encodings, and ranges
- feature_importance.json    : Feature importances of the best tree-based model
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RideWise.Trainer")

# Feature Definitions
TARGET_COLUMN = "cnt"

# Columns deliberately excluded to prevent data leakage and index bias
EXCLUDED_COLUMNS = ["instant", "dteday", "casual", "registered", "yr"]

CATEGORICAL_FEATURES = ["season", "mnth", "hr", "weekday", "weathersit"]
NUMERIC_FEATURES = ["temp", "hum", "windspeed"]
BINARY_FEATURES = ["holiday", "workingday"]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES

FEATURE_METADATA = {
    "features": FEATURE_COLUMNS,
    "target": TARGET_COLUMN,
    "feature_groups": {
        "categorical": CATEGORICAL_FEATURES,
        "numeric": NUMERIC_FEATURES,
        "binary": BINARY_FEATURES
    },
    "descriptions": {
        "season": {
            "type": "categorical",
            "mapping": {"1": "Spring", "2": "Summer", "3": "Fall", "4": "Winter"}
        },
        "mnth": {
            "type": "categorical",
            "mapping": {
                "1": "January", "2": "February", "3": "March", "4": "April",
                "5": "May", "6": "June", "7": "July", "8": "August",
                "9": "September", "10": "October", "11": "November", "12": "December"
            }
        },
        "hr": {
            "type": "categorical",
            "range": [0, 23],
            "description": "Hour of the day (0 to 23)"
        },
        "weekday": {
            "type": "categorical",
            "mapping": {
                "0": "Sunday", "1": "Monday", "2": "Tuesday", "3": "Wednesday",
                "4": "Thursday", "5": "Friday", "6": "Saturday"
            }
        },
        "holiday": {
            "type": "binary",
            "mapping": {"0": "No", "1": "Yes"}
        },
        "workingday": {
            "type": "binary",
            "mapping": {"0": "Weekend or Holiday", "1": "Working Day"}
        },
        "weathersit": {
            "type": "categorical",
            "mapping": {
                "1": "Clear / Few Clouds",
                "2": "Mist / Cloudy",
                "3": "Light Rain / Snow",
                "4": "Heavy Rain / Storm / Snow"
            }
        },
        "temp": {
            "type": "numeric",
            "range": [0.0, 1.0],
            "unit": "Normalized Celsius (real_celsius = temp * 41)"
        },
        "hum": {
            "type": "numeric",
            "range": [0.0, 1.0],
            "unit": "Normalized Humidity (real_percent = hum * 100)"
        },
        "windspeed": {
            "type": "numeric",
            "range": [0.0, 1.0],
            "unit": "Normalized Wind Speed (real_kmh = windspeed * 67)"
        }
    }
}


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """Validate and read the UCI hour.csv file.
    
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns or valid rows are missing.
    """
    if not os.path.exists(csv_path):
        error_msg = (
            f"Dataset not found at '{csv_path}'. "
            "Please ensure 'hour.csv' is present in the dataset/ directory."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Dataset is missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if df.empty:
        error_msg = "Dataset is empty."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Check for missing values
    null_counts = df[required_cols].isnull().sum().sum()
    if null_counts > 0:
        logger.warning(f"Found {null_counts} missing values. Dropping rows with null values.")
        df = df.dropna(subset=required_cols)

    logger.info(f"Dataset successfully loaded. Total rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def split_chronological(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically to prevent temporal data leakage.
    
    The first 80% of rows are used for training and the last 20% for testing.
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    logger.info(
        f"Chronological split ({int(train_ratio*100)}/{int((1-train_ratio)*100)}): "
        f"Train set = {len(train_df)} rows, Test set = {len(test_df)} rows"
    )
    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """Create a scikit-learn ColumnTransformer for preprocessing.
    
    Categorical columns are one-hot encoded; numeric & binary columns pass through.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES
            ),
            (
                "num",
                "passthrough",
                NUMERIC_FEATURES + BINARY_FEATURES
            )
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract human-readable feature names after one-hot encoding."""
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    num_names = NUMERIC_FEATURES + BINARY_FEATURES
    return cat_names + num_names


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]], str]:
    """Train Linear Regression, Decision Tree, and Random Forest pipelines.
    
    Evaluates each on the test set using MAE, RMSE, and R2.
    Returns:
        pipelines: Dict of trained pipelines
        metrics: Dict of performance metrics per model
        best_model_name: The name of the selected best model
    """
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=15, min_samples_split=5),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=20,
            min_samples_split=4,
            n_jobs=-1
        )
    }

    pipelines: Dict[str, Pipeline] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    print("\n" + "=" * 70)
    print(" " * 22 + "MODEL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model Name':<22} | {'MAE':<10} | {'RMSE':<10} | {'R² Score':<10}")
    print("-" * 70)

    for name, model in candidate_models.items():
        logger.info(f"Training {name}...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        # Fit exclusively on training data
        pipeline.fit(X_train, y_train)
        
        # Evaluate on held-out test data
        y_pred = pipeline.predict(X_test)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        pipelines[name] = pipeline
        metrics[name] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4)
        }

        print(f"{name:<22} | {mae:<10.2f} | {rmse:<10.2f} | {r2:<10.4f}")

    print("=" * 70)

    # Model selection based on highest R2 score and lowest RMSE
    best_model_name = max(metrics, key=lambda k: (metrics[k]["r2"], -metrics[k]["rmse"]))
    logger.info(
        f"Selected Best Model: '{best_model_name}' "
        f"(R² = {metrics[best_model_name]['r2']}, RMSE = {metrics[best_model_name]['rmse']}, MAE = {metrics[best_model_name]['mae']})"
    )

    return pipelines, metrics, best_model_name


def extract_feature_importance(pipeline: Pipeline) -> Dict[str, Any]:
    """Extract feature importance from tree-based regressor."""
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(regressor, "feature_importances_"):
        return {"supported": False}

    all_names = get_feature_names(preprocessor)
    raw_importances = regressor.feature_importances_

    # Individual one-hot importances
    feature_importance_list = [
        {"feature": name, "importance": round(float(imp), 5)}
        for name, imp in sorted(zip(all_names, raw_importances), key=lambda x: x[1], reverse=True)
    ]

    # Aggregated importance by original feature column
    aggregated: Dict[str, float] = {col: 0.0 for col in FEATURE_COLUMNS}
    for item in feature_importance_list:
        feat_name = item["feature"]
        for orig_col in FEATURE_COLUMNS:
            if feat_name == orig_col or feat_name.startswith(f"{orig_col}_"):
                aggregated[orig_col] += item["importance"]
                break

    aggregated_sorted = [
        {"feature": col, "importance": round(imp, 5)}
        for col, imp in sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "supported": True,
        "top_features": feature_importance_list[:15],
        "aggregated_by_original_column": aggregated_sorted
    }


def save_artifacts(
    best_model_name: str,
    best_pipeline: Pipeline,
    metrics: Dict[str, Dict[str, float]],
    feature_importance: Dict[str, Any],
    train_count: int,
    test_count: int,
    output_dir: str
):
    """Save all model artifacts and metadata to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save Serialized Pipeline
    model_path = os.path.join(output_dir, "bike_model.joblib")
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Serialized model pipeline saved to: {model_path}")

    # 2. Save Metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    metrics_payload = {
        "split_method": "Chronological (80% train, 20% test)",
        "train_samples": train_count,
        "test_samples": test_count,
        "target": TARGET_COLUMN,
        "candidate_models": metrics,
        "best_model": {
            "name": best_model_name,
            "selection_criteria": "Highest R-squared and lowest RMSE on held-out test split",
            "metrics": metrics[best_model_name]
        }
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Evaluation metrics saved to: {metrics_path}")

    # 3. Save Feature Schema JSON
    schema_path = os.path.join(output_dir, "feature_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_METADATA, f, indent=2)
    logger.info(f"Feature schema saved to: {schema_path}")

    # 4. Save Feature Importance JSON
    importance_path = os.path.join(output_dir, "feature_importance.json")
    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2)
    logger.info(f"Feature importance saved to: {importance_path}")


def run_pipeline(data_path: str = "dataset/hour.csv", output_dir: str = "model") -> Dict[str, Any]:
    """Execute the complete end-to-end training and evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("RideWise — Starting Machine Learning Pipeline Execution")
    logger.info("=" * 60)

    # 1. Load & Validate Data
    df = load_and_validate_data(data_path)

    # 2. Chronological Split (prevents future leakage)
    train_df, test_df = split_chronological(df, train_ratio=0.8)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # 3. Build Preprocessor
    preprocessor = build_preprocessor()

    # 4. Train & Evaluate Models
    pipelines, metrics, best_model_name = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, preprocessor
    )

    # 5. Extract Feature Importance
    best_pipeline = pipelines[best_model_name]
    feature_importance = extract_feature_importance(best_pipeline)

    # 6. Save Artifacts
    save_artifacts(
        best_model_name=best_model_name,
        best_pipeline=best_pipeline,
        metrics=metrics,
        feature_importance=feature_importance,
        train_count=len(train_df),
        test_count=len(test_df),
        output_dir=output_dir
    )

    logger.info("=" * 60)
    logger.info("RideWise Pipeline Execution Completed Successfully!")
    logger.info("=" * 60)

    return {
        "best_model": best_model_name,
        "metrics": metrics,
        "artifacts_dir": output_dir
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RideWise — Bike Rental Model Training Pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset/hour.csv",
        help="Path to the hour.csv dataset (default: dataset/hour.csv)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="model",
        help="Directory to save model artifacts (default: model)"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Flag to explicitly trigger retraining"
    )

    args = parser.parse_args()

    # Normalize paths relative to project root if run from subdirectories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "model" else script_dir

    resolved_data_path = args.data_path
    if not os.path.isabs(resolved_data_path) and not os.path.exists(resolved_data_path):
        resolved_data_path = os.path.join(project_root, args.data_path)

    resolved_model_dir = args.model_dir
    if not os.path.isabs(resolved_model_dir):
        resolved_model_dir = os.path.join(project_root, args.model_dir)

    try:
        run_pipeline(data_path=resolved_data_path, output_dir=resolved_model_dir)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
