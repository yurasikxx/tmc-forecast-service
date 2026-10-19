from fastapi import FastAPI
from app.schemas import ForecastRequest, ForecastResponse
from app.forecast import generate_forecast

app = FastAPI(title="TMC Forecast Service")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    return generate_forecast(request)