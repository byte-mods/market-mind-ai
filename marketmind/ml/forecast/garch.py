"""
GARCH(1,1) volatility forecaster.

Fits Bollerslev's classic GARCH(1,1) on log returns (in % units, the convention
``arch`` uses) and produces a multi-step conditional-variance forecast. The
``ForecastResult`` it emits represents *uncertainty around the current price* —
the GARCH model itself is not directional, so the point forecast applies the
fitted unconditional mean drift to the current close (typically very close to
zero on daily equity data).

Math:
    log_return_t ~ N(μ, σ²_t)
    σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
    cum_log_return_h ~ N(μ·h, Σ σ²_τ for τ=1..h)
    price_at_h ≈ price_0 · exp(cum_log_return_h)
    band_z(h)  = price_0 · exp(μ·h ± z · sqrt(Σ σ²_τ))   (z=1.282 / 1.960)

Fallback: if we have <60 observations or the optimiser fails to converge,
fall back to a rolling stdev annualisation. The result is still emitted but
with ``components.garch.fallback = True`` and confidence flagged so the
ensemble can re-weight or downstream UI can grey out.
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from marketmind.ml.forecast.base import ForecastResult, Forecaster

logger = logging.getLogger(__name__)

MIN_OBS: int = 60      # arch's fitter is unreliable below this
Z_80: float = 1.2816   # one-sided z for 80% PI
Z_95: float = 1.9600   # one-sided z for 95% PI


class GarchForecaster:
    """GARCH(1,1) on percent log-returns. Implements ``Forecaster``."""

    name: str = "garch"

    def __init__(self) -> None:
        self._fitted = False
        self._mu_pct: float = 0.0           # fitted mean return, percent units
        self._sigma_h2_per_step: list[float] = []  # cached forecast variances
        self._last_close: Optional[float] = None
        self._n_obs: int = 0
        self._fallback: bool = False
        self._fallback_sigma_pct: float = 0.0
        self._symbol: str = ""

    # ── Forecaster protocol ──────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> None:
        if "close" not in df.columns:
            raise ValueError("garch.fit: df missing 'close' column")
        self._symbol = str(df.attrs.get("symbol", ""))
        close = pd.Series(df["close"].astype(float).values)
        log_returns_pct = (np.log(close).diff() * 100).dropna()
        self._n_obs = len(log_returns_pct)
        self._last_close = float(close.iloc[-1])

        if self._n_obs < MIN_OBS:
            logger.info("garch fallback: %d obs < %d", self._n_obs, MIN_OBS)
            self._engage_fallback(log_returns_pct)
            return

        try:
            from arch import arch_model
            am = arch_model(log_returns_pct, mean="Constant",
                            vol="GARCH", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            self._mu_pct = float(res.params.get("mu", 0.0))
            # Cache forecast variances; horizon supplied in predict().
            self._fit_result = res
            self._fitted = True
        except Exception as e:
            logger.info("garch fit failed (%s); falling back", e)
            self._engage_fallback(log_returns_pct)

    def predict(self, horizon: int) -> ForecastResult:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self._last_close is None:
            raise RuntimeError("garch.predict called before fit")

        if self._fallback or not self._fitted:
            sigma_h_pct = self._fallback_sigma_pct * math.sqrt(horizon)
            mu_pct = 0.0
            comp_meta: Dict[str, Any] = {"fallback": True, "n_obs": self._n_obs}
        else:
            f = self._fit_result.forecast(horizon=horizon, reindex=False)
            # arch returns variance forecasts as a (1, horizon) DataFrame in
            # the LAST row of f.variance — index represents the origin date.
            var_row = f.variance.iloc[-1].to_numpy()
            cum_var = float(np.sum(var_row))
            sigma_h_pct = math.sqrt(cum_var)
            mu_pct = self._mu_pct
            comp_meta = {
                "fallback": False, "n_obs": self._n_obs,
                "mu_pct": round(mu_pct, 6),
                "sigma_1d_pct": round(math.sqrt(float(var_row[0])), 6),
            }

        # Convert percent log-return moments into price-level bands.
        z_log = lambda z: (mu_pct * horizon + z * sigma_h_pct) / 100.0  # noqa: E731
        p0 = float(self._last_close)
        point   = p0 * math.exp(z_log(0.0))
        lower80 = p0 * math.exp(z_log(-Z_80))
        upper80 = p0 * math.exp(z_log(+Z_80))
        lower95 = p0 * math.exp(z_log(-Z_95))
        upper95 = p0 * math.exp(z_log(+Z_95))

        comp_meta["sigma_h_pct"] = round(sigma_h_pct, 6)

        return ForecastResult(
            symbol=self._symbol,
            horizon_days=horizon,
            as_of=_dt.datetime.now(_dt.timezone.utc),
            point=round(point, 4),
            lower_80=round(lower80, 4), upper_80=round(upper80, 4),
            lower_95=round(lower95, 4), upper_95=round(upper95, 4),
            model="garch",
            regime_conditional=None,
            components={"garch": comp_meta},
            calibration={},
        )

    # ── Internal ─────────────────────────────────────────────────────────
    def _engage_fallback(self, log_returns_pct: pd.Series) -> None:
        # 1-day stdev in percent. If empty: use a sane equity default of 1.5%/day.
        if len(log_returns_pct) >= 2:
            self._fallback_sigma_pct = float(np.std(log_returns_pct, ddof=1))
        else:
            self._fallback_sigma_pct = 1.5
        self._mu_pct = 0.0
        self._fallback = True
        self._fitted = False


_singleton: Optional[GarchForecaster] = None


def get_garch_forecaster() -> GarchForecaster:
    """Stateless-enough to share across calls; new instance per fit if needed."""
    global _singleton
    if _singleton is None:
        _singleton = GarchForecaster()
    return _singleton
