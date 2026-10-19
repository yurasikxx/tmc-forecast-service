from pydantic import BaseModel
from typing import List

class ForecastRequest(BaseModel):
    history: List[float]
    steps: int = 3

class ForecastPoint(BaseModel):
    step: int
    value: float
    lower_bound: float
    upper_bound: float

class ForecastResponse(BaseModel):
    forecast: List[ForecastPoint]
    method: str