"""
OHLCV → multivariate feature tensor for forecasters.

Input  (pandas DataFrame):  date-indexed columns ``open, high, low, close, volume``
Output (pandas DataFrame):  same index + the engineered columns below

Columns produced (deterministic, no leakage — every feature at row t uses only
information available *up to and including* row t):

    return            simple percent change vs prior close
    log_return        natural log return
    rsi_14            Wilder's RSI, 14-period
    macd_hist         MACD histogram, (fast=12, slow=26, signal=9), normalised by close
    ma20_ratio        close / SMA20  - 1
    ma50_ratio        close / SMA50  - 1
    vol20_ann         rolling 20-day stdev of log_return × sqrt(252)
    volume_z          rolling 20-day z-score of volume

The first ~50 rows will be NaN due to warm-up; callers ``.dropna()`` before training.

Why no ``ta`` library: ``ta`` is in requirements.txt for legacy reasons but its
RSI uses Wilder smoothing differently from arch/NeuralProphet conventions and
introduces a hidden dependency on numpy ABI. We compute everything ourselves
with vectorised pandas — fewer surprises across pinned versions.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

REQUIRED_COLS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
FEATURE_COLS: tuple[str, ...] = (
    "return", "log_return", "rsi_14", "macd_hist",
    "ma20_ratio", "ma50_ratio", "vol20_ann", "volume_z",
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with engineered columns appended.

    Raises ValueError if required OHLCV columns are missing.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"build_features: missing OHLCV columns {missing}")
    out = df.copy()
    close = out["close"].astype(float)

    out["return"] = close.pct_change()
    out["log_return"] = np.log(close).diff()
    out["rsi_14"] = _rsi(close, period=14)
    out["macd_hist"] = _macd_hist(close, fast=12, slow=26, signal=9)
    out["ma20_ratio"] = (close / close.rolling(20).mean() - 1.0)
    out["ma50_ratio"] = (close / close.rolling(50).mean() - 1.0)
    out["vol20_ann"] = out["log_return"].rolling(20).std() * np.sqrt(252)

    vol = out["volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std()
    out["volume_z"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

    return out


def feature_matrix(df: pd.DataFrame, cols: Sequence[str] = FEATURE_COLS) -> np.ndarray:
    """Drop NaN rows from a feature-built DF and return ``(N, F)`` numpy array.

    Use this directly as input to PatchTST.
    """
    feat = build_features(df).dropna(subset=list(cols))
    return feat[list(cols)].to_numpy(dtype=np.float32)


# ── Internal: indicator math ─────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI — exponentially smoothed gains/losses."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 over the window: RSI saturates at 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    """MACD histogram, normalised by close to keep cross-symbol comparable."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = (macd - signal_line) / close
    return hist
