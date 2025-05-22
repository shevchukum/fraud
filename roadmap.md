# 📍 Roadmap: Fraud Divorce Prediction System (MLOps)

## ✅ Model and Prediction API

* [X] PostgreSQL database for true data (see data_preparation folder), requests and predictions 
* [X] MLFlow service training and registering XGBoost model
* [X] FastAPI service delivering inference from latest model
* [X] Dockerfile for each service and docker-compose for docker network
* [X] Requests and predictions are logged to PostgreSQL database

## ⚙️ CI/CD: Jenkins + tests

* [X] Script for drift-monitoring
* [X] Custom jenkins container
* [X] 3 piplelines: drift-triggered, each month, 12 months
* [X] Test each pipeline
* [X] Add auth to MLFlow

## 🗃️ Documentation & Cleanup

* [X] Document all services, volumes, routes (README)
* [X] Commit to Git and Docker Hub

