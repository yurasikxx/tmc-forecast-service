from fastapi import FastAPI, HTTPException

from app.forecast import generate_forecast
from app.schemas import ForecastRequest, ForecastResponse

app = FastAPI(title="TMC Forecast Service", version="2.0.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    try:
        return generate_forecast(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"forecast failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=5000, reload=True)
