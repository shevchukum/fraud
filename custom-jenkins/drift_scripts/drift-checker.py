import pandas as pd
import numpy as np
import mlflow
import psycopg2
import os
import json
import sys
from datetime import datetime
from scipy.stats import entropy
from typing import Optional, Union

# Configuration
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.25"))
DRIFT_MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "1000"))

DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "mydatabase"),
    "user": os.getenv("POSTGRES_USER", "myuser"),
    "password": os.getenv("POSTGRES_PASSWORD", "mypassword"),
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

REFERENCE_TABLE = os.getenv("TRAINING_DATA_TABLE", "divorce_predictions")
LOG_TABLE = os.getenv("POSTGRES_TABLE", "prediction_logs")
MODEL_NAME = os.getenv("MODEL_NAME", "divorce-model")
ALIAS = "production"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

def calculate_psi(expected, actual, buckets=10):
    """Calculate PSI with edge case handling"""
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, buckets + 1))
    expected_counts = np.histogram(expected, bins=breakpoints)[0] + 1e-6
    actual_counts = np.histogram(actual, bins=breakpoints)[0] + 1e-6
    
    expected_percents = expected_counts / expected_counts.sum()
    actual_percents = actual_counts / actual_counts.sum()
    
    psi_values = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    return np.sum(psi_values)

def get_connection():
    """Get PostgreSQL connection with retry logic"""
    for attempt in range(3):
        try:
            return psycopg2.connect(**DB_PARAMS)
        except psycopg2.OperationalError as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

def fetch_data(table_name, date_column, since=None, until=None):
    """Safe SQL query with parameterized dates"""
    query = f'SELECT * FROM {table_name}'
    params = []
    conditions = []

    if since is not None:
        since_dt = pd.to_datetime(since)
        conditions.append(f'"{date_column}" >= %s')
        params.append(since_dt if "timestamp" in date_column.lower() else since_dt.date())
    
    if until is not None:
        until_dt = pd.to_datetime(until)
        conditions.append(f'"{date_column}" <= %s')
        params.append(until_dt if "timestamp" in date_column.lower() else until_dt.date())

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    print(f"[DEBUG] Executing query: {query} with params: {params}")

    with get_connection() as conn:
        return pd.read_sql(query, conn, params=params if params else None)

def get_cutoff_date():
    """Get model training date from MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        run = client.get_run(model_version.run_id)
        return run.data.tags.get("training_cutoff")
    except Exception as e:
        print(f"[ERROR] MLflow access failed: {str(e)}", file=sys.stderr)
        raise

def main():
    try:
        cutoff = get_cutoff_date()
        print(f"[INFO] Using cutoff date: {cutoff}")
        
        # Load data
        reference_df = fetch_data(REFERENCE_TABLE, "Date", until=cutoff)
        recent_df = fetch_data(LOG_TABLE, "timestamp", since=cutoff)

        if len(recent_df) < DRIFT_MIN_SAMPLES:
            print(f"[WARN] Insufficient samples ({len(recent_df)} < {DRIFT_MIN_SAMPLES})")
            sys.exit(0)  # Not enough data - neutral exit

        # Process features
        features_df = pd.json_normalize(recent_df['features'])
        recent_df = pd.concat([recent_df.drop(columns=['features']), features_df], axis=1)
        
        # Check drift
        drift_detected = False
        feature_columns = [col for col in reference_df.columns 
                         if col not in ["index", "Date", "Divorce"]]
        
        for feature in feature_columns:
            psi = calculate_psi(reference_df[feature], recent_df[feature])
            print(f"[METRIC] {feature}_psi={psi:.4f}")
            
            if psi > PSI_THRESHOLD:
                print(f"[ALERT] Drift detected in {feature} (PSI={psi:.4f})")
                drift_detected = True
                break  # Exit on first significant drift

        sys.exit(1 if drift_detected else 0)

    except Exception as e:
        print(f"[CRITICAL] {str(e)}", file=sys.stderr)
        sys.exit(2)  # Critical error code

if __name__ == "__main__":
    main()
