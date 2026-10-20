import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from app.schemas import (
    ForecastDiagnostics,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
)

MIN_HISTORY = 12
SEASONAL_PERIOD = 12
MAX_INTERVAL_FACTOR = 3.0
BOOTSTRAP_SIMS = 500
CONFIDENCE_ALPHA = 0.05


def generate_forecast(request: ForecastRequest) -> ForecastResponse:
    history = np.asarray(request.history, dtype=float)
    steps = int(request.steps)

    if len(history) < MIN_HISTORY:
        return _simple_forecast(history, steps, "moving_average_insufficient_data")

    series = _to_monthly_series(history)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=SEASONAL_PERIOD,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)

        if not _is_valid_fit(fit):
            return _simple_forecast(history, steps, "moving_average_fallback")

        mean = np.asarray(fit.forecast(steps), dtype=float)
        ci = _bootstrap_conf_int(fit, steps)

        upper_cap = float(np.max(history)) * MAX_INTERVAL_FACTOR
        capped = False
        points: List[ForecastPoint] = []

        for i in range(steps):
            value = max(0.0, float(mean[i]))
            lo = max(0.0, float(ci[i][0]))
            hi = float(ci[i][1])

            if hi > upper_cap:
                hi = upper_cap
                capped = True

            points.append(
                ForecastPoint(
                    step=i + 1,
                    value=round(value, 2),
                    lower_bound=round(lo, 2),
                    upper_bound=round(hi, 2),
                )
            )

        diagnostics = _build_diagnostics(fit, series, capped)
        reliability = _compute_reliability(diagnostics, points)

        return ForecastResponse(
            forecast=points,
            method="holt_winters",
            reliability=reliability,
            diagnostics=diagnostics,
        )

    except Exception:
        return _simple_forecast(history, steps, "moving_average_fallback")


def _to_monthly_series(history: np.ndarray) -> pd.Series:
    index = pd.date_range(
        end=pd.Timestamp.today().normalize().replace(day=1),
        periods=len(history),
        freq="MS",
    )
    return pd.Series(history, index=index, name="required_quantity")


def _is_valid_fit(fit) -> bool:
    try:
        aic = getattr(fit, "aic", None)
        if aic is None or not np.isfinite(aic):
            return False
        resid = np.asarray(fit.resid, dtype=float)
        if resid.size == 0 or not np.isfinite(resid).all():
            return False
        return True
    except Exception:
        return False


def _bootstrap_conf_int(fit, steps: int) -> np.ndarray:
    resid = np.asarray(fit.resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    mean_path = np.asarray(fit.forecast(steps), dtype=float)

    if resid.size < 5:
        sigma = float(np.std(resid)) if resid.size > 0 else 0.0
        z = 1.96
        return np.column_stack([mean_path - z * sigma, mean_path + z * sigma])

    resid_centered = resid - resid.mean()
    rng = np.random.default_rng(seed=42)
    noise = rng.choice(resid_centered, size=(BOOTSTRAP_SIMS, steps), replace=True)
    sims = mean_path[None, :] + noise

    lower = np.quantile(sims, CONFIDENCE_ALPHA / 2, axis=0)
    upper = np.quantile(sims, 1 - CONFIDENCE_ALPHA / 2, axis=0)
    return np.column_stack([lower, upper])


def _build_diagnostics(fit, series: pd.Series, capped: bool) -> ForecastDiagnostics:
    lb_pvalue: Optional[float] = None
    try:
        resid = np.asarray(fit.resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        if resid.size >= 10:
            lag = min(24, max(1, resid.size // 3))
            lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
            lb_pvalue = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        pass

    params = getattr(fit, "params", {}) or {}

    def _p(name: str) -> Optional[float]:
        try:
            v = params.get(name) if isinstance(params, dict) else getattr(params, name, None)
            return float(v) if v is not None else None
        except Exception:
            return None

    return ForecastDiagnostics(
        aic=round(float(fit.aic), 2),
        converged=True,
        ljung_box_pvalue=round(lb_pvalue, 4) if lb_pvalue is not None else None,
        intervals_capped=capped,
        n_obs=len(series),
        smoothing_level=_p("smoothing_level"),
        smoothing_trend=_p("smoothing_trend"),
        smoothing_seasonal=_p("smoothing_seasonal"),
    )


def _compute_reliability(
        diagnostics: ForecastDiagnostics, points: List[ForecastPoint]
) -> str:
    lb_ok = (
            diagnostics.ljung_box_pvalue is not None
            and diagnostics.ljung_box_pvalue > 0.05
    )

    widths = [(p.upper_bound - p.lower_bound) / max(p.value, 1e-6) for p in points]
    avg_width = float(np.mean(widths)) if widths else 1.0
    width_ok = avg_width < 0.25

    if lb_ok and width_ok:
        return "high"
    if lb_ok or width_ok:
        return "medium"
    return "low"


def _simple_forecast(history: np.ndarray, steps: int, method_name: str) -> ForecastResponse:
    avg = float(np.mean(history)) if history.size > 0 else 0.0
    points = [
        ForecastPoint(
            step=i + 1,
            value=round(avg, 2),
            lower_bound=round(avg * 0.8, 2),
            upper_bound=round(avg * 1.2, 2),
        )
        for i in range(steps)
    ]
    return ForecastResponse(
        forecast=points,
        method=method_name,
        reliability="low",
        diagnostics=ForecastDiagnostics(n_obs=int(history.size)),
    )
