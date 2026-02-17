import importlib
import math

import src.inference.predictor as predictor_module


def test_default_sentiment_weight_is_point_ten(monkeypatch):
    monkeypatch.delenv("SENTIMENT_WEIGHT", raising=False)
    monkeypatch.delenv("ENSEMBLE_WEIGHT", raising=False)
    mod = importlib.reload(predictor_module)
    assert math.isclose(mod.SENTIMENT_WEIGHT, 0.10, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mod.ENSEMBLE_WEIGHT, 0.90, rel_tol=0, abs_tol=1e-9)


def test_same_day_sentiment_required_to_change_combined_prediction(monkeypatch):
    mod = predictor_module
    monkeypatch.setattr(mod, "ENSEMBLE_WEIGHT", 0.90)
    monkeypatch.setattr(mod, "SENTIMENT_WEIGHT", 0.10)

    technical = 0.02
    same_day_meta = {
        "article_count": 2,
        "weights_used": {"day0": 0.35},
        "weighted_sentiment_raw": 0.8,
        "weighted_sentiment_decimal": 0.05,
    }
    old_news_meta = {
        "article_count": 2,
        "weights_used": {"day1": 0.25, "day2": 0.15},
        "weighted_sentiment_raw": 0.8,
        "weighted_sentiment_decimal": 0.05,
    }

    blended_same_day = mod.LivePredictor._blend_strategy_and_sentiment(
        technical,
        same_day_meta,
    )
    blended_old_news = mod.LivePredictor._blend_strategy_and_sentiment(
        technical,
        old_news_meta,
    )

    assert blended_same_day[4] is True
    assert blended_same_day[2] > 0
    assert blended_same_day[0] > technical
    assert blended_old_news[4] is False
    assert math.isclose(blended_old_news[2], 0.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(blended_old_news[0], technical, rel_tol=0, abs_tol=1e-12)


def test_same_day_severe_negative_sentiment_does_not_override_strategy(monkeypatch):
    mod = predictor_module
    monkeypatch.setattr(mod, "ENSEMBLE_WEIGHT", 0.90)
    monkeypatch.setattr(mod, "SENTIMENT_WEIGHT", 0.10)
    monkeypatch.setattr(mod, "NEGATIVE_SENTIMENT_CUTOFF", -0.45)

    technical = 0.018
    meta = {
        "article_count": 1,
        "weights_used": {"day0": 0.35},
        "weighted_sentiment_raw": -0.9,
        "weighted_sentiment_decimal": -0.05,
    }
    combined, ew, sw, _sent_used, same_day = (
        mod.LivePredictor._blend_strategy_and_sentiment(
            technical,
            meta,
        )
    )
    assert same_day is True
    assert math.isclose(sw, 0.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(ew, 1.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(combined, technical, rel_tol=0, abs_tol=1e-12)
