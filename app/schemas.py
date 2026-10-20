from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    history: List[float] = Field(..., min_length=1)
    steps: int = Field(default=12, ge=1, le=36)


class ForecastPoint(BaseModel):
    step: int
    value: float
    lower_bound: float
    upper_bound: float


class ForecastDiagnostics(BaseModel):
    aic: Optional[float] = None
    converged: Optional[bool] = None
    ljung_box_pvalue: Optional[float] = None
    intervals_capped: Optional[bool] = None
    n_obs: Optional[int] = None
    smoothing_level: Optional[float] = None
    smoothing_trend: Optional[float] = None
    smoothing_seasonal: Optional[float] = None


class ForecastResponse(BaseModel):
    forecast: List[ForecastPoint]
    method: str
    reliability: Literal["high", "medium", "low"] = "medium"
    diagnostics: ForecastDiagnostics = Field(default_factory=ForecastDiagnostics)
