''' Training and logging new divorce prediction models '''

import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import mlflow
import mlflow.xgboost
import time
import warnings
import requests
import tempfile

from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError 
from scipy.stats import ttest_ind

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Switching off one warning from XGBoost
warnings.filterwarnings("ignore", message=".*Saving model in the UBJSON format as default.*")

MODEL_NAME = os.getenv("MODEL_NAME", "divorce-model")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(engine, cutoff_date, retries=10, delay=3):
    """Load and validate data from PostgreSQL with retry on failure"""
    logger.info(f"Loading data before {cutoff_date}")
    query = f"""
    SELECT * FROM divorce_predictions 
    WHERE "Date" < '{cutoff_date}'
    ORDER BY "Date" DESC
    """
    
    # trying to connect to Postgres
    for attempt in range(retries):
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            break  # success
        except OperationalError as e:
            logger.warning(f"Postgres not ready (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    else:
        logger.error(f"Failed to connect to Postgres after {retries} attempts.")
        raise

    if len(df) == 0:
        raise ValueError(f"No data found before cutoff date {cutoff_date}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')
    
def preprocess_data(df):
    """Prepare features and target, split into train/test"""
    X = df.drop(columns=['Divorce', 'Date', 'index'], errors='ignore').astype(np.float32)   
    y = df['Divorce'].values
    
    if len(np.unique(y)) < 2:
        raise ValueError("Target variable needs both classes (0 and 1)")
    
    # Partition with test samples being most recent
    n_test = int(len(df) * 0.10)
    X_train, y_train = X.iloc[:-n_test], y[:-n_test]
    X_test, y_test = X.iloc[-n_test:], y[-n_test:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, X.columns

def train_model(X_train, y_train):
    """Train XGBoost model with grid search"""
    logger.info("Starting model training with grid search")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='aucpr',
        scale_pos_weight=sum(y_train==0)/sum(y_train==1)
    )
    
    param_grid = {
        'max_depth': [3],
        'learning_rate': [0.01],
        'n_estimators': [100],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='average_precision',
        cv=3,
        verbose=1,
        n_jobs=6,
        refit=True,
        error_score='raise'
    )
    
    grid_search.fit(X_train, y_train)
    logger.info(f"Best params: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, y_pred), y_pred

def compare_with_production(X_test, y_test, feature_names):
    """Compare with current production model if exists"""
    try:
        prod_model = mlflow.xgboost.load_model("models:/divorce-model@production")
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        y_prod_pred = prod_model.predict_proba(X_test_df)[:, 1]
        prod_score = average_precision_score(y_test, y_prod_pred)
        return prod_score, y_prod_pred
    except MlflowException:
        logger.info("No production model found")
        return None, None

def log_artifacts(model, scaler, feature_names, X_test_scaled, y_pred):
    # Log scaler
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "scaler.pkl")
        joblib.dump(scaler, tmp_path)
        mlflow.log_artifact(tmp_path)
    
    # Log feature importance
    importance = model.feature_importances_
    mlflow.log_dict({k: float(v) for k, v in zip(feature_names, importance)}, "feature_importance.json")
    
    # Log model signature
    X_test = pd.DataFrame(X_test_scaled, columns=feature_names)
    signature = infer_signature(X_test, y_pred)
    
    return signature

def wait_for_mlflow(uri, retries=10, delay=3):
    logger.info(f"Waiting for MLflow server at {uri}...")
    for attempt in range(retries):
        try:
            response = requests.get(f"{uri}/health")
            if response.status_code == 200:
                logger.info("MLflow server is ready.")
                return
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}: MLflow not ready: {e}")
            time.sleep(delay)
    raise RuntimeError(f"MLflow server not available after {retries} attempts.")

def main():
    try:
        # Configuration
        DB_CONFIG = {
            "host": "postgres",
            "port": 5432,
            "user": "myuser",
            "password": "mypassword",
            "database": "mydatabase"
        }
        
        # Get cutoff date
        try:
            today = datetime.strptime(os.getenv("TODAY", ""), "%Y-%m-%d").date()
        except ValueError:
            today = datetime.today().date()
            logger.info(f"No valid TODAY env var, using current date: {today}")
        
        # Inti database 
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )

        # Inti MLFlow
        wait_for_mlflow(MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MODEL_NAME)
        
        with mlflow.start_run():
            # Load and preprocess data
            df = load_data(engine, cutoff_date=today)
            X_train, y_train, X_test, y_test, scaler, feature_names = preprocess_data(df)
            
            # Train and evaluate model
            model = train_model(X_train, y_train)
            score, y_pred = evaluate_model(model, X_test, y_test)
            
            # Compare with production
            prod_score, y_prod_pred = compare_with_production(X_test, y_test, feature_names)
            
            # Log metrics and parameters
            mlflow.log_params(model.get_params())
            mlflow.log_metric("average_precision", score)
            mlflow.log_metric("positive_class_ratio_train", y_train.mean())
            mlflow.log_metric("positive_class_ratio_test", y_test.mean())

            # Log date as parameter and tag
            today_str = today.isoformat()
            mlflow.log_param("training_date", today_str)
            mlflow.set_tag("data_cutoff_date", today_str)
            
            if prod_score is not None:
                mlflow.log_metric("production_average_precision", prod_score)
                mlflow.set_tag("comparison_result", f"{score:.4f} vs {prod_score:.4f}")
                
                # Check if we should promote
                promote_model = score > prod_score
                
                if promote_model:
                    _, p_value = ttest_ind(y_pred, y_prod_pred)
                    mlflow.log_metric("p_value", p_value)
                    mlflow.set_tag("stat_significance", p_value < 0.05)
            else:
                promote_model = True
            
            # Log artifacts and promote
            if promote_model:
                signature = log_artifacts(model, scaler, feature_names, X_test, y_pred)
                
                result = mlflow.xgboost.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name="divorce-model"
                )

                client = MlflowClient()
                latest_versions = client.search_model_versions("name='divorce-model'")
                new_model_version = str(latest_versions[0].version)
                
                # If a production model exists, mark it as deprecated
                if prod_score is not None:
                    current_prod = client.get_model_version_by_alias("divorce-model", "production")
                    client.set_registered_model_alias(
                        name="divorce-model",
                        alias="deprecated",
                        version=str(current_prod.version)  # Previous production version
                    )
                    # Remove the old "production" alias
                    client.delete_registered_model_alias("divorce-model", "production")
                
                # Promote the new model to production
                client.set_registered_model_alias(
                    name="divorce-model",
                    alias="production",
                    version=new_model_version
                )
                
                mlflow.set_tag("status", "promoted_to_production")
                mlflow.set_tag("new_version", new_model_version)
                logger.info(f"Promoted new model version {new_model_version}")
            else:
                mlflow.set_tag("status", "not_promoted_to_production")
                logger.info("Model not promoted to production")
            
            # Log dataset metadata
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.set_tag("training_cutoff", today.isoformat())
            
    except Exception as e:
        logger.exception("Training pipeline failed")
        raise

if __name__ == "__main__":
    main()