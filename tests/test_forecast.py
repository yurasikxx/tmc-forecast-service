import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def calculate_mape(actual, forecast):
    """Вспомогательная функция для расчета MAPE."""
    actual, forecast = np.array(actual), np.array(forecast)
    return np.mean(np.abs((actual - forecast) / actual)) * 100


def test_forecast_metrics_and_length():
    """Тест 1: Проверка качества прогноза (MAPE) и корректности структуры ответа."""
    import random
    random.seed(123)
    np.random.seed(123)

    history = []
    for i in range(36):
        base = 100 + i * 2
        season = 10 if i % 12 == 11 else 0
        noise = np.random.normal(0, 3)
        history.append(base + season + noise)

    steps = 6

    response = client.post("/forecast", json={
        "history": history,
        "steps": steps
    })

    assert response.status_code == 200
    data = response.json()

    assert len(data["forecast"]) == steps
    assert data["method"] == "holt_winters"

    for point in data["forecast"]:
        assert point["lower_bound"] <= point["value"] <= point["upper_bound"]

    train_history = history[:30]
    test_actual = history[30:]

    response_train = client.post("/forecast", json={
        "history": train_history,
        "steps": 6
    })
    forecast_values = [p["value"] for p in response_train.json()["forecast"]]
    mape = calculate_mape(test_actual, forecast_values)

    print(f"\n[METRICS] MAPE на тестовой выборке: {mape:.2f}%")
    print(f"[METRICS] AIC: {data['diagnostics']['aic']}")
    print(f"[METRICS] Ljung-Box p-value: {data['diagnostics']['ljung_box_pvalue']}")
    print(f"[METRICS] Reliability: {data['reliability']}")

    assert mape <= 15.0, f"MAPE слишком высок: {mape}%"
    assert data["diagnostics"]["ljung_box_pvalue"] > 0.05


def test_forecast_fallback_on_insufficient_data():
    """Тест 2: Проверка срабатывания fallback при коротком ряде данных."""
    short_history = [50, 55, 60, 52, 58]

    response = client.post("/forecast", json={
        "history": short_history,
        "steps": 3
    })

    assert response.status_code == 200
    data = response.json()

    assert data["method"] == "moving_average_insufficient_data"
    assert len(data["forecast"]) == 3
    assert 50 <= data["forecast"][0]["value"] <= 60
