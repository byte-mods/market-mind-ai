"""C2 verification: meta-stacker softmax + signal recovery on synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

from marketmind.ml.forecast.meta_stacker import (
    FEATURE_DIM,
    LABELS,
    MetaStacker,
    REGIME_STATES,
    _synthetic_dataset,
    feature_vector,
    get_meta_stacker,
    reset_meta_stacker_for_tests,
)


def test_feature_vector_shape() -> None:
    v = feature_vector({})
    assert v.shape == (FEATURE_DIM,)
    assert (v == 0.0).all()


def test_feature_vector_one_hot_regime() -> None:
    v = feature_vector({"regime_state": "trending bull"})
    # trending_bull is at offset 3
    assert v[3] == 1.0
    # exactly one regime dim active
    assert v[3:3 + len(REGIME_STATES)].sum() == 1.0


def test_feature_vector_unknown_regime_neutral() -> None:
    v = feature_vector({"regime_state": "atlantis"})
    # All 5 regime dims should equal 0.2
    expected = 1.0 / len(REGIME_STATES)
    assert np.allclose(v[3:3 + len(REGIME_STATES)], expected)


def test_meta_stacker_softmax_sums_to_one() -> None:
    s = get_meta_stacker()
    out = s.predict_proba({
        "forecast_return": 0.03, "forecast_vol": 0.02,
        "rl_signal_score": 0.5, "regime_state": "trending_bull",
        "sentiment_tilt": 0.2,
    })
    total = out["p_buy"] + out["p_sell"] + out["p_hold"]
    assert abs(total - 1.0) < 1e-9
    # All probs are valid
    assert 0.0 <= out["p_buy"] <= 1.0
    assert 0.0 <= out["p_sell"] <= 1.0
    assert 0.0 <= out["p_hold"] <= 1.0


def test_meta_stacker_recovers_train_signal_on_synthetic_labels() -> None:
    """The bootstrap fit on the synthetic rule should achieve well above
    chance accuracy. Random guessing would give ~0.33."""
    reset_meta_stacker_for_tests()
    s = get_meta_stacker()
    X, y = _synthetic_dataset(n=500, seed=99)  # held-out fold
    y_idx = np.asarray([LABELS.index(lbl) for lbl in y])
    score = float(s._model.score(X, y_idx))
    assert score > 0.75, f"meta-stacker held-out score {score} ≤ 0.75"


def test_meta_stacker_recommends_buy_on_strong_bullish_features() -> None:
    s = get_meta_stacker()
    out = s.predict_proba({
        "forecast_return": 0.05, "forecast_vol": 0.015,
        "rl_signal_score": 0.8, "regime_state": "trending_bull",
        "sentiment_tilt": 0.6,
    })
    # On strong bullish features, P(BUY) should dominate
    assert out["p_buy"] > out["p_sell"]
    assert out["p_buy"] > out["p_hold"]


def test_meta_stacker_recommends_sell_on_strong_bearish_features() -> None:
    s = get_meta_stacker()
    out = s.predict_proba({
        "forecast_return": -0.05, "forecast_vol": 0.04,
        "rl_signal_score": -0.7, "regime_state": "crash",
        "sentiment_tilt": -0.7,
    })
    assert out["p_sell"] > out["p_buy"]


def test_meta_stacker_recommends_hold_on_neutral_features() -> None:
    s = get_meta_stacker()
    out = s.predict_proba({
        "forecast_return": 0.001, "forecast_vol": 0.02,
        "rl_signal_score": 0.05, "regime_state": "range",
        "sentiment_tilt": 0.0,
    })
    # On neutral features, HOLD should be the modal class
    assert out["p_hold"] > out["p_buy"]
    assert out["p_hold"] > out["p_sell"]


def test_meta_stacker_predict_before_fit_raises() -> None:
    s = MetaStacker()
    with pytest.raises(RuntimeError):
        s.predict_proba({"forecast_return": 0.0})


def test_fit_from_history_validates_feature_dim() -> None:
    s = MetaStacker()
    X = np.zeros((10, FEATURE_DIM - 1))
    y = ["HOLD"] * 10
    with pytest.raises(ValueError):
        s.fit_from_history(X, y)


def test_meta_stacker_singleton_returns_same_instance() -> None:
    reset_meta_stacker_for_tests()
    a = get_meta_stacker()
    b = get_meta_stacker()
    assert a is b
