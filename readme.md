# 🧠 Fraud detection in a government agency  – MLOps Stack

This project deploys a full MLOps pipeline for prediction of fruad marriages to get a marriage loan. Pipeline includes model training, experiment tracking, inference via API, data drift detection, and retraining automation. System is managed by Docker Compose. 

## 🚀 Services Overview

| Service          | Purpose                                                | Exposed Ports               |
|------------------|--------------------------------------------------------|-----------------------------|
| `postgres`       | PostgreSQL database for logs and training data         | *Internal only*             |
| `mlflow-server`  | MLflow Tracking Server for experiments and artifacts   | *Internal only*             |
| `mlflow-train`   | Image that performs training and logs models to MLflow | No ports exposed            |
| `fastapi`        | Prediction API and pretected access to MLflow UI       | `8000:8000`                 |
| `jenkins`        | CI/CD server for automated retraining and monitoring   | `8080:8080`                 |

## 📁 Volumes

| Volume         | Mounted In                                | Purpose                                  |
|----------------|-------------------------------------------|------------------------------------------|
| `pgdata`       | `/var/lib/postgresql/data`                | PostgreSQL training and logging data     |
| `jenkins_data` | `/var/jenkins_home`                       | Jenkins home directory                   |
| `./mlflow`     | `/mlflow` (in multiple containers)        | MLflow artifacts and tracking database   |

## 🔗 Internal Routes

These services are connected via the internal Docker network `myproject_network`.

| Service           | URL from inside Docker                     |
|-------------------|--------------------------------------------|
| MLflow Tracking   | `http://mlflow-server:5000`                |
| PostgreSQL        | `postgres:5432`                            |
| FastAPI           | `http://fastapi:8000`                      |

## 📌 Service Descriptions

📂 postgres: stores training data, API inputs and predictions. Initialized via SQL scripts in ./initdb.

🔬 mlflow-server: runs the MLflow Tracking UI and backend. Stores artifacts under /mlflow/mlruns and metadata in mlflow.db. 
                  Accessible through localhost:8000/mlflow with password.

🧠 mlflow-train: runs a training pipeline and logs results to MLflow.

⚡ fastapi: provides an API for real-time predictions. Loads 'production' model from MLflow and logs each prediction to PostgreSQL.

🔁 jenkins: performs CI/CD tasks like monitoring data drift, triggering retraining, and updating the deployed model.

## 🛠️ Running the System

Because all images are ready and public at Docker Hub, to start the system 

1. Copy repo.

2. docker-compose up -d

## 🔒 Security Notes

- MLflow server is accessible with password through FastAPI request: localhost:8000/mlflow/

- FastAPI is exposed publicly at localhost:8000 for external prediction requests.

- Jenkins is exposed at localhost:8080, access with password.

The following credentials are required in a `.env` file in the root directory:

POSTGRES_USER
POSTGRES_PASSWORD
MLFLOW_UI_USERNAME
MLFLOW_UI_PASSWORD
JENKINS_ADMIN_ID
JENKINS_ADMIN_PASSWORD

## 📞 API Usage (FastAPI)

There are two example requests for testing: (1) signle prediction request: request_1.json and (2) batch (1001 samples) prediction request: request_drift.json. 
The second is used for drift detection testing.

1. For single prediction request use :
   curl -X POST \
       http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d "$(cat request_1.json)" 

   In Powershell:
   Invoke-RestMethod -Uri http://localhost:8000/predict `
      -Method POST `
      -ContentType "application/json" `
      -Body (Get-Content -Raw -Path "request_1.json")

2. For batch prediction request use:
   curl -X POST \
       http://localhost:8000/batch_predict \
      -H "Content-Type: application/json" \
      -d "$(cat request_drift.json)"

   In Powershell: 
   Invoke-RestMethod -Uri http://localhost:8000/batch_predict `
      -Method POST `
      -ContentType "application/json" `
      -Body (Get-Content -Raw -Path "request_drift.json")

Return is probability of divroce within 1 month after loan. Single request example:

        probability
        -----------
        0.39910775423049927

## 📁 Folder Structure
.
├── docker-compose.yml
├── .env
├── readme.md
├── roadmap.md
├── request_1.json
├── request_drift.json
├── mlflow/
|   └── mlruns/
├── initdb/
├── custom-jenkins/
│   ├── drift_scripts/
│   ├── pipelines/
│   └── init.groovy.d/
├── fastapi/
└── data_preparation/

## ✅ Health Checks & Dependencies

1. postgres includes a health check using pg_isready.

2. fastapi waits for both PostgreSQL and mlflow-train to finish.

3. jenkins mounts all necessary training and drift monitoring scripts as read-only.

# ✨ Author & License
Developed by Dmitrii Shevchuk
License: none
   