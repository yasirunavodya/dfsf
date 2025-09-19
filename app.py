# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from inference import RFPowerForecaster

app = FastAPI(title="Solar Power RF Forecaster", version="1.0.0")
forecaster = RFPowerForecaster()  # loads artifacts/model.joblib + artifacts/metadata.json

class Row(BaseModel):
    Date: str
    # Include at least the weather columns your model used and (optionally) Power Output
    # Unknown extras will be ignored
    # Example fields:
    temperature_2m_templin: Optional[float] = None
    cloud_cover_templin: Optional[float] = None
    shortwave_radiation_templin: Optional[float] = None
    diffuse_radiation_templin: Optional[float] = None
    direct_normal_irradiance_templin: Optional[float] = None


class ForecastRequest(BaseModel):
    history: List[Row]  # at least last 6 hours (more is fine)
    future_weather: Optional[List[Dict[str, Any]]] = None  # 3 rows with Date + weather cols

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_next3")
def predict_next3(req: ForecastRequest):
    # Convert to DataFrames
    hist_df = pd.DataFrame([r.dict() for r in req.history])
    fut_df = pd.DataFrame(req.future_weather) if req.future_weather else None

    preds = forecaster.forecast_next3(history_df=hist_df, future_weather_df=fut_df)
    return {"predictions": [{"Timestamp": str(idx), "Predicted_Power_Output": float(val)} 
                            for idx, val in preds["Predicted_Power_Output"].items()]}
