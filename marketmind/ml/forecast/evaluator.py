"""
OOS prediction-interval coverage evaluator.

The W3.1 acceptance criterion is "80% PI covers actual outcome ≥75% of the
time on held-out test." This module is the harness that measures it.

Usage:
    from marketmind.ml.forecast.ensemble import EnsembleForecaster
    from marketmind.ml.forecast.evaluator import evaluate_pi_coverage

    metrics = evaluate_pi_coverage(
        df=historical_ohlcv_df,
        forecaster_factory=EnsembleForecaster,
        horizon=5,
        n_slices=100,
        min_train_rows=120,
    )
    # metrics = {
    #   "pi80_coverage": 0.78,   # ✅ ≥ 0.75 → calibrated
    #   "pi95_coverage": 0.94,
    #   "mae": 12.3,
    #   "n_slices_evaluated": 100,
    # }

Mechanics:
    Anchored walk-forward — the same approach as ``WalkForwardBacktester``.
    For each anchor t in the held-out window: fit on df[:t], predict at
    t+horizon, compare against actual close. The factory is invoked fresh at
    each anchor so we don't leak fitted state from prior anchors.

Performance: this is a *benchmark* harness, not a request-path tool. Expect
minutes to run on full daily history × 100 slices. Run it offline to
populate the ``calibration`` field that the API surfaces.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from marketmind.ml.forecast.base import Forecaster, ForecastResult

logger = logging.getLogger(__name__)


def evaluate_pi_coverage(
    df: pd.DataFrame,
    forecaster_factory: Callable[[], Forecaster],
    horizon: int = 5,
    n_slices: int = 100,
    min_train_rows: int = 120,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Anchored walk-forward over ``df``; return coverage stats.

    Args:
        df: full historical OHLCV with at least ``min_train_rows + horizon + n_slices`` rows.
        forecaster_factory: callable returning a fresh ``Forecaster`` per anchor.
            Use ``lambda: EnsembleForecaster()`` for production calibration.
        horizon: prediction horizon in trading days.
        n_slices: number of OOS anchors to evaluate. Capped at len(df) - min_train_rows - horizon.
        min_train_rows: anchors below this train-window length are skipped.
        progress: optional callback ``(done, total)`` — for CLI progress bars.

    Returns:
        dict with ``pi80_coverage``, ``pi95_coverage``, ``mae``, ``n_slices_evaluated``,
        plus a ``failed`` count for anchors where the forecaster raised.
    """
    if "close" not in df.columns:
        raise ValueError("evaluate_pi_coverage: df missing 'close' column")
    n = len(df)
    last_anchor = n - horizon - 1
    first_anchor = max(min_train_rows, 0)
    if last_anchor <= first_anchor:
        raise ValueError(
            f"Not enough rows: need at least {min_train_rows + horizon + 1}, got {n}"
        )

    # Sample anchors evenly between [first_anchor, last_anchor]
    candidate_anchors = np.linspace(
        first_anchor, last_anchor, num=min(n_slices, last_anchor - first_anchor),
    ).astype(int)
    candidate_anchors = sorted(set(candidate_anchors.tolist()))

    inside_80 = 0
    inside_95 = 0
    abs_errors: List[float] = []
    failed = 0
    total = len(candidate_anchors)

    for i, t in enumerate(candidate_anchors):
        if progress is not None:
            progress(i, total)
        try:
            f = forecaster_factory()
            f.fit(df.iloc[:t + 1])
            r: ForecastResult = f.predict(horizon)
        except Exception as e:  # noqa: BLE001
            logger.warning("evaluator: anchor %d failed: %s", t, e)
            failed += 1
            continue
        actual = float(df["close"].iloc[t + horizon])
        if r.lower_80 <= actual <= r.upper_80:
            inside_80 += 1
        if r.lower_95 <= actual <= r.upper_95:
            inside_95 += 1
        abs_errors.append(abs(actual - r.point))

    n_eval = len(abs_errors)
    return {
        "pi80_coverage": round(inside_80 / n_eval, 4) if n_eval else 0.0,
        "pi95_coverage": round(inside_95 / n_eval, 4) if n_eval else 0.0,
        "mae": round(float(np.mean(abs_errors)), 4) if abs_errors else 0.0,
        "n_slices_evaluated": n_eval,
        "n_slices_failed": failed,
        "horizon_days": horizon,
        "calibrated_at_pi80": (inside_80 / n_eval >= 0.75) if n_eval else False,
    }
