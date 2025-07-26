# trip_api_advanced.py
from fastapi import FastAPI
import pandas as pd
from sklearn.svm import LinearSVR
import joblib
from pydantic import BaseModel
import uvicorn

# 1. Load and prepare your Excel data
def load_data():
    try:
        # Replace with your actual Excel path
        df = pd.read_excel("trip_data.xlsx")  
        
        # Convert time columns to minutes (example for 'actual_time')
        # Add other time conversions as needed
        df['actual_time'] = df['actual_time'].apply(lambda x: float(x.split(':')[0]) * 60 + float(x.split(':')[1]))
        
        # Select features and target
        X = df[['osrm_distance', 'actual_distance_to_destination', 'factor']]
        y = df['actual_time']
        return X, y
    
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

# 2. Train and save model
X, y = load_data()
model = LinearSVR(C=1.0, epsilon=0.1, max_iter=10000)
model.fit(X, y)
joblib.dump(model, 'trip_model.pkl')

# 3. Create FastAPI app
app = FastAPI()

class TripRequest(BaseModel):
    osrm_distance: float
    actual_distance: float
    factor: float

@app.post("/predict")
def predict(request: TripRequest):
    try:
        prediction = model.predict([[request.osrm_distance, 
                                  request.actual_distance, 
                                  request.factor]])
        return {
            "predicted_time_minutes": round(prediction[0], 2),
            "confidence": "High"  # Add actual confidence calculation if available
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)