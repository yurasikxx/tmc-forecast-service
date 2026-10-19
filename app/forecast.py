import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from app.schemas import ForecastRequest, ForecastResponse, ForecastPoint

def generate_forecast(request: ForecastRequest) -> ForecastResponse:
    history = request.history
    steps = request.steps

    if len(history) < 6:
        return _simple_forecast(history, steps, "moving_average")

    try:
        model = ARIMA(history, order=(1, 1, 0))
        model_fit = model.fit()

        forecast_result = model_fit.get_forecast(steps=steps)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)

        points = []
        for i in range(steps):
            points.append(ForecastPoint(
                step=i + 1,
                value=max(0.0, float(forecast_values[i])),
                lower_bound=max(0.0, float(conf_int[i][0])),
                upper_bound=max(0.0, float(conf_int[i][1]))
            ))

        return ForecastResponse(forecast=points, method="arima")
    except Exception:
        return _simple_forecast(history, steps, "moving_average_fallback")


def _simple_forecast(history, steps, method_name):
    if len(history) == 0:
        avg = 0.0
    else:
        avg = float(np.mean(history))

    points = []
    for i in range(steps):
        points.append(ForecastPoint(
            step=i + 1,
            value=round(avg, 2),
            lower_bound=round(avg * 0.8, 2),
            upper_bound=round(avg * 1.2, 2)
        ))
    return ForecastResponse(forecast=points, method=method_name)