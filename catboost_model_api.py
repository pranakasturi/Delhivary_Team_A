from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import joblib
import logging
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Load the model
model = joblib.load("catboost_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# ------------------ Logging Configuration ------------------ #
logging.basicConfig(
    filename="api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------ Prometheus Metrics ------------------ #
REQUEST_COUNTER = Counter("http_requests_total", "Total number of API requests")
ERROR_COUNTER = Counter("http_errors_total", "Total number of errors")

# ------------------ Request Schema ------------------ #
class InputData(BaseModel):
    cutoff_factor: float
    actual_distance_to_destination: float
    segment_osrm_distance: float
    segment_factor: float
    osrm_distance: float
    start_scan_to_end_scan: float
    source_city: int
    destination_city: int
    is_cutoff: int

# ------------------ Middleware for Counting Requests ------------------ #
@app.middleware("http")
async def count_requests(request: Request, call_next):
    REQUEST_COUNTER.inc()
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        ERROR_COUNTER.inc()
        logging.error(f"Unhandled Error: {str(e)}")
        raise e

# ------------------ Prediction Endpoint ------------------ #
@app.post("/predict")
async def predict(input_data: InputData):
    try:
        input_list = list(input_data.dict().values())
        logging.info(f"Incoming Request: {input_data.dict()}")
        
        prediction = model.predict([input_list])
        
        logging.info(f"Prediction Output: {prediction.tolist()}")
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

# ------------------ Prometheus Metrics Endpoint ------------------ #
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------------------ Root Endpoint ------------------ #
@app.get("/")
def read_root():
    return {"message": "CatBoost Model API is running"}
