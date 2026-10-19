import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from app.schemas import ForecastRequest, ForecastResponse, ForecastPoint


def generate_forecast(request: ForecastRequest) -> ForecastResponse:
    history = request.history
    steps = request.steps

    if len(history) < 12:
        return _simple_forecast(history, steps, "moving_average_insufficient_data")

    if len(history) >= 24:
        try:
            model = SARIMAX(
                history,
                order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False, maxiter=200)

            forecast_result = model_fit.get_forecast(steps=steps)

            forecast_values = np.array(forecast_result.predicted_mean)
            conf_int = np.array(forecast_result.conf_int(alpha=0.05))

            points = []
            for i in range(steps):
                points.append(ForecastPoint(
                    step=i + 1,
                    value=max(0.0, float(forecast_values[i])),
                    lower_bound=max(0.0, float(conf_int[i][0])),
                    upper_bound=max(0.0, float(conf_int[i][1]))
                ))
            return ForecastResponse(forecast=points, method="sarima")
        except Exception as e:
            print(f"SARIMA failed: {e}. Falling back to Holt-Winters.")

    try:
        model = ExponentialSmoothing(
            history,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )
        model_fit = model.fit()

        forecast_values = np.array(model_fit.forecast(steps))

        points = []
        for i in range(steps):
            val = max(0.0, float(forecast_values[i]))
            points.append(ForecastPoint(
                step=i + 1,
                value=val,
                lower_bound=max(0.0, val * 0.85),
                upper_bound=val * 1.15
            ))
        return ForecastResponse(forecast=points, method="holt_winters")
    except Exception as e:
        print(f"Holt-Winters failed: {e}. Falling back to moving average.")

    return _simple_forecast(history, steps, "moving_average_fallback")


def _simple_forecast(history, steps, method_name):
    avg = float(np.mean(history)) if len(history) > 0 else 0.0

    points = []
    for i in range(steps):
        points.append(ForecastPoint(
            step=i + 1,
            value=round(avg, 2),
            lower_bound=round(avg * 0.8, 2),
            upper_bound=round(avg * 1.2, 2)
        ))
    return ForecastResponse(forecast=points, method=method_name)
