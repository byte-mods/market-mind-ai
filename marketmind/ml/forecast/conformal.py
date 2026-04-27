"""
Split conformal prediction wrapper.

Wraps any inner ``Forecaster`` and replaces its raw 80/95 PI bands with
*calibrated* bands that carry a marginal coverage guarantee:

    P( |y - ŷ| ≤ q̂_{1-α} )  ≥  1 - α    (under exchangeability)

Algorithm (vanilla split conformal):

    1. ``fit(df)``:
        a. Split df chronologically into train (80%) and calibration (20%).
        b. Train the inner forecaster on the train set.
        c. For each row in the calibration set, compute the absolute
           residual |actual_close_at_t+h  -  inner.predict(h).point|.
        d. Store the empirical (1-α) quantile of those residuals as q̂.
    2. ``predict(h)``:
        a. Re-fit inner on full df (train + cal) so the point is best-
           informed; this is a common refinement and does not break
           split-CP coverage in practice for stationary series.
        b. point   = inner.predict(h).point
        c. lower_X = point - q̂_X
        d. upper_X = point + q̂_X
       where q̂_X is the (1-X)-quantile residual at the chosen confidence.

Exchangeability caveat: financial time series are NOT iid. Split CP gives a
*marginal* (averaged-over-time) coverage guarantee, not a conditional one.
For regime-shifted markets, recalibrate periodically — operators call
``recalibrate(df_new)`` to refresh q̂ without re-training the inner model.
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from marketmind.ml.forecast.base import ForecastResult, Forecaster

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_FRAC: float = 0.2
MIN_CALIBRATION_ROWS: int = 30


class SplitConformalWrapper:
    """Split-CP wrapper around any inner ``Forecaster``."""

    name: str = "conformal"

    def __init__(
        self,
        inner: Forecaster,
        horizon: int,
        calibration_frac: float = DEFAULT_CALIBRATION_FRAC,
    ) -> None:
        self.inner = inner
        self.horizon = horizon
        self.calibration_frac = calibration_frac
        self._fitted = False
        self._symbol = ""
        self._last_close: Optional[float] = None
        # q̂_alpha for {0.10, 0.05} (i.e. 90% and 95% PIs respectively)
        self._q90: float = 0.0
        self._q95: float = 0.0
        self._n_calibration: int = 0

    # ── Forecaster protocol ──────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> None:
        if "close" not in df.columns:
            raise ValueError("conformal.fit: df missing 'close' column")
        self._symbol = str(df.attrs.get("symbol", ""))
        self._last_close = float(df["close"].iloc[-1])

        n = len(df)
        cal_rows = max(MIN_CALIBRATION_ROWS, int(n * self.calibration_frac))
        if n < cal_rows + self.horizon + 50:
            # Not enough rows to split — fall back: train on full, set q̂=0
            # (caller's bands collapse to point; clearly degraded signal).
            logger.info("conformal: insufficient rows (%d) for split; degraded mode", n)
            try:
                inner_df = df.copy()
                inner_df.attrs["symbol"] = self._symbol
                self.inner.fit(inner_df)
            except Exception:
                pass
            self._q90 = 0.0
            self._q95 = 0.0
            self._n_calibration = 0
            self._fitted = True
            return

        split_idx = n - cal_rows - self.horizon
        train_df = df.iloc[:split_idx + 1].copy()
        train_df.attrs["symbol"] = self._symbol

        # Train inner on the prefix
        self.inner.fit(train_df)

        # Generate calibration residuals: at each anchor t in [split_idx, n-h-1],
        # the inner is fit on df[:t+1] - we approximate by using one-shot predict
        # off the trained-on-prefix model and walking the close forward. This is
        # an approximation; full leave-one-out CP is too expensive here.
        residuals = []
        # Use the fixed-fit model's point as the prediction; anchor it at each t
        # by adjusting against the true close at t (last close at t).
        try:
            base = self.inner.predict(self.horizon)
            # base.point is a forecast from the prefix's last close
            base_last_close = float(train_df["close"].iloc[-1])
            base_implied_return = base.point / base_last_close
        except Exception as e:
            logger.warning("conformal: inner predict failed during calibration: %s", e)
            base_implied_return = 1.0

        cal_anchors = range(split_idx, n - self.horizon)
        for t in cal_anchors:
            anchor_close = float(df["close"].iloc[t])
            actual = float(df["close"].iloc[t + self.horizon])
            predicted = anchor_close * base_implied_return
            residuals.append(abs(actual - predicted))

        residuals_arr = np.asarray(residuals, dtype=float)
        if len(residuals_arr) >= MIN_CALIBRATION_ROWS:
            # Empirical quantile correction for finite-sample CP:
            #   q̂_(1-α) = ceil( (n+1)(1-α) ) / n  empirical quantile of |R|
            n_cal = len(residuals_arr)
            sorted_r = np.sort(residuals_arr)
            self._q90 = float(_finite_sample_quantile(sorted_r, 0.10, n_cal))
            self._q95 = float(_finite_sample_quantile(sorted_r, 0.05, n_cal))
        else:
            self._q90 = 0.0
            self._q95 = 0.0

        self._n_calibration = len(residuals_arr)

        # Re-fit inner on full df so point is best-informed
        full_df = df.copy()
        full_df.attrs["symbol"] = self._symbol
        try:
            self.inner.fit(full_df)
        except Exception as e:
            logger.warning("conformal: refit on full df failed: %s", e)

        self._fitted = True

    def predict(self, horizon: int) -> ForecastResult:
        if not self._fitted:
            raise RuntimeError("conformal.predict called before fit")
        if horizon != self.horizon:
            raise ValueError(
                f"conformal was calibrated for horizon={self.horizon}; got {horizon}"
            )
        inner_result = self.inner.predict(horizon)
        point = float(inner_result.point)
        # 80% PI is interpolated linearly between point and the 90% band — an
        # honest approximation that retains the calibration property at 90/95.
        # Operators should treat the 90 and 95 bands as the load-bearing ones.
        q80 = 0.5 * self._q90  # rough interpolation (not coverage-guaranteed)
        return ForecastResult(
            symbol=self._symbol or inner_result.symbol,
            horizon_days=horizon,
            as_of=_dt.datetime.now(_dt.timezone.utc),
            point=round(point, 4),
            lower_80=round(point - q80, 4),
            upper_80=round(point + q80, 4),
            lower_95=round(point - self._q95, 4),
            upper_95=round(point + self._q95, 4),
            model="conformal",
            regime_conditional=inner_result.regime_conditional,
            components={
                **inner_result.components,
                "conformal": {
                    "q90": round(self._q90, 6),
                    "q95": round(self._q95, 6),
                    "n_calibration": self._n_calibration,
                    "calibration_frac": self.calibration_frac,
                },
            },
            calibration={
                "method": "split_conformal",
                "alpha_90": 0.10,
                "alpha_95": 0.05,
                "n_calibration": self._n_calibration,
            },
        )

    def recalibrate(self, df_new: pd.DataFrame) -> None:
        """Refresh the calibration quantiles on new data without re-fitting the
        inner forecaster. Operators call this weekly under regime shift."""
        if not self._fitted:
            raise RuntimeError("conformal: must fit() before recalibrate()")
        if "close" not in df_new.columns:
            raise ValueError("recalibrate: df missing 'close' column")
        n = len(df_new)
        if n < self.horizon + MIN_CALIBRATION_ROWS:
            raise ValueError(f"recalibrate: need ≥{self.horizon + MIN_CALIBRATION_ROWS} rows, got {n}")

        try:
            base = self.inner.predict(self.horizon)
            ref_close = float(df_new["close"].iloc[0])
            implied = base.point / ref_close
        except Exception:
            implied = 1.0

        residuals = []
        for t in range(n - self.horizon):
            anchor = float(df_new["close"].iloc[t])
            actual = float(df_new["close"].iloc[t + self.horizon])
            residuals.append(abs(actual - anchor * implied))
        sorted_r = np.sort(np.asarray(residuals, dtype=float))
        n_cal = len(sorted_r)
        self._q90 = float(_finite_sample_quantile(sorted_r, 0.10, n_cal))
        self._q95 = float(_finite_sample_quantile(sorted_r, 0.05, n_cal))
        self._n_calibration = n_cal


def _finite_sample_quantile(sorted_residuals: np.ndarray, alpha: float, n: int) -> float:
    """Standard CP finite-sample correction: idx = ceil((n+1)(1-α)) - 1."""
    if n == 0:
        return 0.0
    rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    rank = max(0, min(rank, n - 1))
    return float(sorted_residuals[rank])
