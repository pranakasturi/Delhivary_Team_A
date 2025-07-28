
# CatBoost API with Logging and Monitoring

This FastAPI application serves predictions using a trained CatBoost model and integrates logging and monitoring tools.

## Features
- Logs incoming requests, responses, and errors using Python's logging module.
- Exposes metrics at `/metrics` endpoint using Prometheus client.
- Ready for Prometheus & Grafana integration.

## Setup

### Requirements
Install required packages:
```
pip install -r requirements.txt
```

### Run the API
```
uvicorn catboost_model_api:app --reload
```

### Prometheus Monitoring
To monitor the API with Prometheus:
1. Use the `prometheus.yml` config provided.
2. Start Prometheus with:
```
prometheus --config.file=prometheus.yml
```
3. Visit Prometheus UI at `http://localhost:9090`.

### Grafana (Optional)
You can connect Grafana to Prometheus at `localhost:9090` and create dashboards using the metrics exposed.

## Logging
Logs are written to `api_logs.log`. It includes:
- Request payloads
- Prediction results
- Errors and tracebacks
