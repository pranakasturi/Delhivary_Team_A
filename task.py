# trip_api.py
from fastapi import FastAPI
import pandas as pd
from sklearn.svm import LinearSVR
import joblib
import numpy as np

# 1. Sample Data Preparation (replace with your data)
data = pd.DataFrame({
    'distance': [10, 20, 30, 15, 25],
    'osrm_distance': [11, 21, 31, 16, 26],
    'actual_time': [15, 25, 35, 20, 30]
})

# 2. Train Model Immediately
X = data[['distance', 'osrm_distance']]
y = data['actual_time']
model = LinearSVR().fit(X, y)
joblib.dump(model, 'trip_model.pkl')

# 3. Create FastAPI App
app = FastAPI()

class TripRequest(BaseModel):
    distance: float
    osrm_distance: float

@app.post("/predict")
def predict(request: TripRequest):
    """Takes distance values, returns predicted time"""
    prediction = model.predict([[request.distance, request.osrm_distance]])
    return {"predicted_time": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)