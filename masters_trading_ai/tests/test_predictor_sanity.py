import pytest

from src.inference.predictor import _sanitize_predicted_return


def test_decimal_pass_through():
    assert _sanitize_predicted_return(0.02, {"xgboost": 0.02}) == pytest.approx(0.02)


def test_percent_input_converted():
    assert _sanitize_predicted_return(2.5, {"arima": 2.5}) == pytest.approx(0.025)


def test_large_value_capped():
    assert _sanitize_predicted_return(75.0, {"arima": 75.0}) == pytest.approx(0.5)


def test_nan_returns_zero():
    assert _sanitize_predicted_return(float("nan"), {"xgboost": 0.1}) == pytest.approx(
        0.0
    )
