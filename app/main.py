from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import joblib  
from app.model.lstm_model import LSTMTripDuration  # Your model class

app = FastAPI()

# Load model and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMTripDuration(num_input_features=12).to(device)
model.load_state_dict(torch.load("app/model/lstm_trip_duration.pt", map_location=device))
model.eval()

# Load scaler with joblib
scaler = joblib.load("app/model/scaler.pkl") 

# Define input schema
class TripInput(BaseModel):
    vendor_id: int
    passenger_count: int
    pickup_hour: int
    pickup_weekday: int
    pickup_month: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: int
    log_haversine_distance: float
    is_holiday: int

@app.post("/predict")
def predict_duration(data: TripInput):
    # Numeric features
    features = np.array([[  
        data.vendor_id,
        data.passenger_count,
        data.log_haversine_distance,
        data.pickup_longitude,
        data.pickup_latitude,
        data.dropoff_longitude,
        data.dropoff_latitude,
        data.store_and_fwd_flag,
        data.pickup_month,
        data.pickup_hour,
        data.is_holiday,
        data.pickup_weekday
    ]], dtype=np.float32)
        
    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    # Categorical features
    hour = torch.tensor([data.pickup_hour]).to(device)
    day = torch.tensor([data.pickup_weekday]).to(device)
    month = torch.tensor([data.pickup_month-1]).to(device)
    
    with torch.no_grad():
        pred_log = model(features_tensor, hour, day, month)
        pred_sec = torch.expm1(pred_log).cpu().item()

    pred_min = pred_sec / 60.0
    return {"predicted_duration_min": round(pred_min, 2)}