import os
import time
import tempfile
import mlflow
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import uuid
import json
import httpx

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import RedirectResponse, HTMLResponse, Response

from pydantic import BaseModel, create_model, ValidationError
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# PostgreSQL config (used for logging incoming requests)
POSTGRES_HOST = os.getenv("POSTGRES_HOST")  
POSTGRES_PORT = os.getenv("POSTGRES_PORT")      
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_TABLE = os.getenv("POSTGRES_TABLE", "prediction_logs")

# MLFlow config (used for training and loggin models)
MODEL_NAME = os.getenv("MODEL_NAME", "divorce-model")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
ALIAS = "production"
MAX_RETRIES = 3
RETRY_DELAY = 5

# Response Models
class SinglePredictionResponse(BaseModel):
    probability: float

class BatchPredictionResponse(BaseModel):
    probabilities: List[float]

class ModelUpdateResponse(BaseModel):
    message: str
    model_version: str

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    data_cutoff_date: Optional[str] = None
    status: str = "loaded"

# Model Download Utilities
def download_model_from_mlflow():
    """Download model and scaler from MLflow with proper error handling"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Getting production model version
        model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        version_number = model_version.version
        
        # Load the production model
        model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        run_id = pyfunc_model.metadata.run_id
        run = client.get_run(run_id)
        
        # Extracting training data cutoff date 
        data_cutoff_date = run.data.tags.get("data_cutoff_date")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download model
            model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/model/model.xgb",
                dst_path=tmp_dir
            )
            xgb_model = xgb.Booster()
            xgb_model.load_model(model_path)

            # Download scaler if exists
            scaler = None
            try:
                scaler_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{run_id}/scaler.pkl",
                    dst_path=tmp_dir
                )
                scaler = joblib.load(scaler_path)
            except Exception:
                print("No scaler found, proceeding without scaling")

        # Get feature names from model signature
        signature = pyfunc_model.metadata.signature
        if not signature or not signature.inputs:
            raise ValueError("Model signature missing or incomplete")
        
        input_names = [input.name for input in signature.inputs.inputs]
        
        return xgb_model, scaler, input_names, version_number, data_cutoff_date
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {str(e)}")

def wait_for_model(timeout_seconds=600, retry_interval=10):
    """Wait for model to become available in MLflow"""
    start_time = time.time()
    last_error = None
    
    while time.time() - start_time < timeout_seconds:
        try:
            return download_model_from_mlflow()
        except Exception as e:
            last_error = e
            print(f"Waiting for model... Attempt failed: {str(e)}")
            time.sleep(retry_interval)
    
    raise RuntimeError(f"Model not available after {timeout_seconds}s. Last error: {str(last_error)}")

def create_predict_request_model(input_names: List[str]):
    """Dynamically create Pydantic model for input validation"""
    fields = {name: (float, ...) for name in input_names}
    return create_model("PredictRequest", **fields)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to DB on start"""
    try:
        app.state.pg_conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        print("✅ Connection to PostgreSQL succesefful")
    except Exception as e:
        print(f"❌ PostgreSQL connection error: {str(e)}")
        raise

    # Loading the model
    model, scaler, input_names, version, data_cutoff_date = wait_for_model()
    PredictRequest = create_predict_request_model(input_names)
    
    app.state.xgb_model = model
    app.state.scaler = scaler
    app.state.input_names = input_names
    app.state.PredictRequest = PredictRequest
    app.state.model_version = version
    app.state.data_cutoff_date = data_cutoff_date

    yield

    # Closing DB connection on stop
    if hasattr(app.state, 'pg_conn'):
        app.state.pg_conn.close()
        print("🔌 Connection to PostgreSQL is closed")

# Log prediction to PostgreSQL database
async def log_prediction_to_db(
    request: Request,
    features: Dict[str, float],
    probability: float,
    endpoint: str,
    request_id: str = None
) -> None:

    print("📝 Logging prediction to DB")
    request_id = request_id or str(uuid.uuid4())
    
    cursor = None
    try:
        # Using connection from app.state
        cursor = request.app.state.pg_conn.cursor()
        
        query = sql.SQL("""
            INSERT INTO {table} 
            (request_id, timestamp, model_version, divorce_probability, features, endpoint)
            VALUES (%s, %s, %s, %s, %s, %s)
        """).format(table=sql.Identifier(POSTGRES_TABLE))
        
        cursor.execute(query, (
            request_id,
            datetime.now(timezone.utc),
            request.app.state.model_version,
            probability,
            json.dumps(features),
            endpoint
        ))
        
        # Commiting new records
        request.app.state.pg_conn.commit()

    except psycopg2.InterfaceError as e:
        print(f"Cursor error: {str(e)}")
        # Trying to re-connect
        await restore_db_connection(request.app)
    except Exception as e:
        print(f"Logging error: {str(e)}")
        if cursor and not request.app.state.pg_conn.closed:
            request.app.state.pg_conn.rollback()
        raise e
    finally:
        if cursor:
            cursor.close()

async def restore_db_connection(app: FastAPI):
    """Re-connecting to DB in case of lost connection"""
    try:
        if hasattr(app.state, 'pg_conn'):
            app.state.pg_conn.close()
            
        app.state.pg_conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        print("♻️ Connection to PostgreSQL is restored")
    except Exception as e:
        print(f"❌ Unable to restore connection to PostgreSQL: {str(e)}")
        raise

# Prepare data for the model
def prepare_input_data(validated_data, input_names, scaler=None):
    """Prepare input data for prediction"""
    features_df = pd.DataFrame([validated_data], columns=input_names)
    
    if scaler:
        features_df = pd.DataFrame(
            scaler.transform(features_df),
            columns=input_names
        )
    
    return xgb.DMatrix(
        features_df.astype(np.float32),
        feature_names=input_names
    )

  
app = FastAPI(
    lifespan=lifespan,
    title="Quick Divorce Prediction API",
    description="API for predicting quick divorce probability based on loan application",
    version="1.0.0"
)
security = HTTPBasic()

# Checking the credentials
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("MLFLOW_UI_USERNAME")
    correct_password = os.getenv("MLFLOW_UI_PASSWORD")
    
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=401,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Protected proxy for MLflow UI
@app.api_route("/mlflow/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_mlflow(full_path: str, request: Request):
    rewritten_path = full_path
    target_url = f"http://mlflow-server:5000/{rewritten_path}"

    # important headers: Authorization, Cookie, Content-Type, Accept
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    body = await request.body()

    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            params=request.query_params,
        )

    # Give away all headers except technical
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers={
            k: v for k, v in resp.headers.items()
            if k.lower() not in {
                "content-encoding", "transfer-encoding", "connection",
                "content-length"  
            }
        }
    )


# Redirect from /mlflow to /mlflow/ 
@app.get("/mlflow")
async def mlflow_redirect():
    return RedirectResponse(url="/mlflow/")

# Endpoints
@app.post("/predict", response_model=SinglePredictionResponse)
async def predict_single(request: Request, input_data: Dict):
    """Make a single prediction"""
    try:
        # Validate input
        validated = request.app.state.PredictRequest(**input_data)
        validated_dict = validated.dict()
        
        # Prepare and make prediction
        dmatrix = prepare_input_data(
            validated_dict,
            request.app.state.input_names,
            request.app.state.scaler
        )
        
        probability = float(request.app.state.xgb_model.predict(dmatrix)[0])
        
        # Logging the request and result
        await log_prediction_to_db(
            request=request,
            features=validated_dict,
            probability=probability,
            endpoint="single"
        )
        
        return {"probability": probability}
        
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"message": "Input validation failed", "errors": e.errors()}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_batch(request: Request, batch_data: Dict[str, List[Dict]]):
    """Make batch predictions"""
    try:
        if "instances" not in batch_data:
            raise ValueError("Input must contain 'instances' key")
            
        # Validate each instance
        PredictRequest = request.app.state.PredictRequest
        validated_instances = []
        
        for i, item in enumerate(batch_data["instances"]):
            try:
                validated_instances.append(PredictRequest(**item).dict())
            except ValidationError as e:
                raise ValueError(f"Instance {i} validation failed: {str(e)}")

        # Convert to numpy array
        features = np.array([list(instance.values()) for instance in validated_instances], 
                          dtype=np.float32)
        
        # Scale
        if request.app.state.scaler:
            features = request.app.state.scaler.transform(features)
        
        # Make predictions
        dmatrix = xgb.DMatrix(features, feature_names=request.app.state.input_names)
        probabilities = request.app.state.xgb_model.predict(dmatrix).tolist()
        
        # Logging batch request
        batch_id = str(uuid.uuid4())
        for instance, probability in zip(validated_instances, probabilities):
            await log_prediction_to_db(
                request=request,
                features=instance,
                probability=probability,
                endpoint="batch",
                request_id=batch_id
            )
        
        return {"probabilities": probabilities}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/update-model", response_model=ModelUpdateResponse)
async def update_model(request: Request):
    """Update model from MLflow
    
    This endpoint forces a model reload from MLflow registry
    """
    for attempt in range(MAX_RETRIES):
        try:
            model, scaler, input_names, version, data_cutoff_date = wait_for_model()
            PredictRequest = create_predict_request_model(input_names)
            
            app.state.xgb_model = model
            app.state.scaler = scaler
            app.state.input_names = input_names
            app.state.PredictRequest = PredictRequest
            app.state.model_version = version
            app.state.data_cutoff_date = data_cutoff_date
            
            return {
                "message": "Model updated successfully",
                "model_version": version
            }
            
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model update failed after {MAX_RETRIES} attempts: {str(e)}"
                )
            time.sleep(RETRY_DELAY)

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(request: Request):
    """Get information about currently loaded model"""
    return {
        "model_name": MODEL_NAME,
        "model_version": request.app.state.model_version,
        "data_cutoff_date": request.app.state.data_cutoff_date,
        "status": "loaded"
    }