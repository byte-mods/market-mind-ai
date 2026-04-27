"""
Ensemble forecaster: blends Trend + PatchTST + GARCH into a unified
``ForecastResult`` with regime-conditional bull/bear branches.

Blend rules (load-bearing — read before changing):

POINT
    Direction comes from the *directional* forecasters: PatchTST and Trend.
    GARCH is a volatility model, not a directional one — it's used for bands
    only. Default weights:
        point = 0.6 * patchtst.point + 0.4 * trend.point
    If a component is in fallback, its weight transfers to the other; if both
    directional components are in fallback, point = last close (i.e. naive).

BANDS
    GARCH owns the band logic — it's a likelihood-based uncertainty model.
    The ensemble re-anchors GARCH bands at the ensemble point:
        upper_X = ensemble_point * (garch.upper_X / garch.point)
    If GARCH is in fallback, the bands fall through to the wider of the two
    directional forecasters' bands.

REGIME-CONDITIONAL
    Query the regime classifier. Build two scenarios:
        bull = ensemble_point * (1 + bull_drift)     where bull_drift = +0.005 * sqrt(horizon)
        bear = ensemble_point * (1 - bear_drift)     where bear_drift = +0.005 * sqrt(horizon)
    Bands shifted with the point. This is a heuristic — clearly labelled in
    components so the UI can show "regime-conditional (heuristic)" not
    "regime-conditional (calibrated)". Conformal-calibrated versions are W3.2.

CALIBRATION
    The ensemble does NOT compute its own calibration metrics — that's
    evaluator.py's job (F8). The ``calibration`` field is populated by the
    cache layer (F7) reading the most recent evaluator run.
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Any, Dict, Optional

import pandas as pd

from marketmind.ml.forecast.base import Band, ForecastResult, Forecaster
from marketmind.ml.forecast.garch import GarchForecaster
from marketmind.ml.forecast.patchtst import PatchTSTForecaster
from marketmind.ml.forecast.prophet_trend import TrendForecaster

logger = logging.getLogger(__name__)

DIRECTIONAL_WEIGHT_PATCHTST: float = 0.6
DIRECTIONAL_WEIGHT_TREND: float = 0.4


class EnsembleForecaster:
    """Combines three ``Forecaster`` instances. Implements ``Forecaster``.

    Sub-forecasters can be injected for testing — production usage hits the
    module-level singletons via ``get_ensemble_forecaster()``.
    """

    name: str = "ensemble"

    def __init__(
        self,
        garch: Optional[Forecaster] = None,
        trend: Optional[Forecaster] = None,
        patchtst: Optional[Forecaster] = None,
        regime_provider: Optional[callable] = None,
    ) -> None:
        self.garch = garch if garch is not None else GarchForecaster()
        self.trend = trend if trend is not None else TrendForecaster()
        self.patchtst = patchtst if patchtst is not None else PatchTSTForecaster()
        # regime_provider() returns dict like {"state": "Trending Bull", "confidence": 0.7}
        self.regime_provider = regime_provider
        self._symbol = ""
        self._last_close: Optional[float] = None
        self._fitted = False

    # ── Forecaster protocol ──────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> None:
        if "close" not in df.columns:
            raise ValueError("ensemble.fit: df missing 'close' column")
        self._symbol = str(df.attrs.get("symbol", ""))
        self._last_close = float(df["close"].iloc[-1])
        # Fit all three; per-component failures degrade to that component's
        # fallback path — they don't bubble up.
        for name, f in (("garch", self.garch),
                        ("trend", self.trend),
                        ("patchtst", self.patchtst)):
            try:
                f.fit(df)
            except Exception as e:  # noqa: BLE001
                logger.warning("ensemble: %s fit failed: %s", name, e)
        self._fitted = True

    def predict(self, horizon: int) -> ForecastResult:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self._last_close is None:
            raise RuntimeError("ensemble.predict called before fit")

        try:
            r_garch = self.garch.predict(horizon)
        except Exception as e:  # noqa: BLE001
            logger.warning("ensemble: garch.predict failed: %s", e)
            r_garch = None
        try:
            r_trend = self.trend.predict(horizon)
        except Exception as e:  # noqa: BLE001
            logger.warning("ensemble: trend.predict failed: %s", e)
            r_trend = None
        try:
            r_patch = self.patchtst.predict(horizon)
        except Exception as e:  # noqa: BLE001
            logger.warning("ensemble: patchtst.predict failed: %s", e)
            r_patch = None

        point = self._blend_point(r_trend, r_patch)
        bands = self._blend_bands(point, r_garch, r_trend, r_patch)
        regime_conditional = self._regime_branches(point, bands, horizon)

        components: Dict[str, Any] = {}
        if r_garch:
            components["garch"] = r_garch.components.get("garch", {}) | {"point": r_garch.point}
        if r_trend:
            components["trend"] = r_trend.components.get("trend", {}) | {"point": r_trend.point}
        if r_patch:
            components["patchtst"] = r_patch.components.get("patchtst", {}) | {"point": r_patch.point}

        return ForecastResult(
            symbol=self._symbol,
            horizon_days=horizon,
            as_of=_dt.datetime.now(_dt.timezone.utc),
            point=round(point, 4),
            lower_80=round(bands.lower_80, 4),
            upper_80=round(bands.upper_80, 4),
            lower_95=round(bands.lower_95, 4),
            upper_95=round(bands.upper_95, 4),
            model="ensemble",
            regime_conditional=regime_conditional,
            components=components,
            calibration={},
        )

    # ── Internal blending ────────────────────────────────────────────────
    def _blend_point(self, r_trend, r_patch) -> float:
        w_p = DIRECTIONAL_WEIGHT_PATCHTST if (r_patch and not _is_fallback(r_patch, "patchtst")) else 0.0
        w_t = DIRECTIONAL_WEIGHT_TREND if (r_trend and not _is_fallback(r_trend, "trend")) else 0.0
        if w_p + w_t == 0:
            # Neither directional component is reliable; fall back to last close.
            return float(self._last_close)
        # Re-normalise weights
        total = w_p + w_t
        w_p /= total
        w_t /= total
        p = w_p * r_patch.point if r_patch else 0.0
        p += w_t * r_trend.point if r_trend else 0.0
        return float(p)

    def _blend_bands(self, point: float, r_garch, r_trend, r_patch) -> Band:
        if r_garch and not _is_fallback(r_garch, "garch") and r_garch.point > 0:
            scale = point / r_garch.point
            return Band(
                point=point,
                lower_80=r_garch.lower_80 * scale,
                upper_80=r_garch.upper_80 * scale,
                lower_95=r_garch.lower_95 * scale,
                upper_95=r_garch.upper_95 * scale,
            )
        # GARCH unavailable / fallback — take the wider of trend/patchtst bands.
        widest = _widest(r_trend, r_patch)
        if widest is None:
            # Naïve flat ±2% bands as last resort
            sigma = 0.02
            return Band(
                point=point,
                lower_80=point * math.exp(-1.2816 * sigma),
                upper_80=point * math.exp(+1.2816 * sigma),
                lower_95=point * math.exp(-1.96 * sigma),
                upper_95=point * math.exp(+1.96 * sigma),
            )
        # Re-anchor to ensemble point preserving width
        half80 = (widest.upper_80 - widest.lower_80) / 2.0
        half95 = (widest.upper_95 - widest.lower_95) / 2.0
        return Band(
            point=point,
            lower_80=point - half80, upper_80=point + half80,
            lower_95=point - half95, upper_95=point + half95,
        )

    def _regime_branches(self, point: float, bands: Band, horizon: int) -> Dict[str, Any]:
        bull_drift = 0.005 * math.sqrt(horizon)
        bear_drift = 0.005 * math.sqrt(horizon)
        bull_point = point * (1.0 + bull_drift)
        bear_point = point * (1.0 - bear_drift)
        # Shift bands proportionally
        return {
            "bull": Band(
                point=bull_point,
                lower_80=bands.lower_80 * (1.0 + bull_drift),
                upper_80=bands.upper_80 * (1.0 + bull_drift),
                lower_95=bands.lower_95 * (1.0 + bull_drift),
                upper_95=bands.upper_95 * (1.0 + bull_drift),
            ),
            "bear": Band(
                point=bear_point,
                lower_80=bands.lower_80 * (1.0 - bear_drift),
                upper_80=bands.upper_80 * (1.0 - bear_drift),
                lower_95=bands.lower_95 * (1.0 - bear_drift),
                upper_95=bands.upper_95 * (1.0 - bear_drift),
            ),
            "_method": "heuristic_drift",
        }


def _is_fallback(result, component_key: str) -> bool:
    return bool(result.components.get(component_key, {}).get("fallback", False))


def _widest(*results):
    """Return the result with the widest 80% PI (or None if all are None)."""
    candidates = [r for r in results if r is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r.upper_80 - r.lower_80))


_singleton: Optional[EnsembleForecaster] = None


def get_ensemble_forecaster() -> EnsembleForecaster:
    global _singleton
    if _singleton is None:
        _singleton = EnsembleForecaster()
    return _singleton
