"""RideWise — Model Retraining Entrypoint
========================================
Convenience script to trigger model retraining with optional dataset updates.

Usage:
    python model/retrain.py
    python model/retrain.py --data-path dataset/hour.csv
"""

import os
import sys

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train_model import run_pipeline

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "dataset", "hour.csv")
    output_dir = os.path.join(base_dir, "model")

    print("[RideWise] Retraining initiated...")
    run_pipeline(data_path=data_path, output_dir=output_dir)
    print("[RideWise] Retraining complete. New model artifacts generated.")
