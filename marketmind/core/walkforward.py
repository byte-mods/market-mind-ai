"""
MarketMind AI - Walk-Forward Backtester (W1.3)

Replaces the single-shot backtest with an anchored walk-forward evaluation:

  fold 1:  train [t0, t0+W) → test [t0+W, t0+W+H)
  fold 2:  train [t0, t0+W+H) → test [t0+W+H, t0+W+2H)
  ...

Each fold runs the same strategy on the training window (in-sample) and the
test window (out-of-sample) with realistic Indian retail costs:

  • Brokerage:                  ₹20 flat / order (Zerodha equity intraday)
  • STT (delivery):             0.10 % buy + 0.10 % sell
  • Exchange charges (NSE):     ~0.00322 % round-trip
  • GST 18 % on (brokerage + txn charges)
  • SEBI charges:               0.0001 %
  • Stamp duty:                 0.003 % on buy
  • Slippage:                   5 bps per side (configurable)

Outputs the CDF of out-of-sample Sharpe + drawdown distribution and an
in-sample / out-of-sample Sharpe gap, which is the headline overfit detector.

Bootstrap: 1000 i.i.d. resamples of in-sample daily returns → Sharpe CDF.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from marketmind.core.backtester import Backtester, get_backtester

logger = logging.getLogger(__name__)


@dataclass
class CostModel:
    """Indian retail equity cost model — delivery defaults; tune as needed."""
    brokerage_per_order: float = 20.0       # ₹ flat per executed order
    stt_buy_pct: float = 0.0010              # 0.10 %
    stt_sell_pct: float = 0.0010             # 0.10 %
    exchange_txn_pct: float = 0.0000322      # NSE round-trip approximation
    gst_pct: float = 0.18                    # on (brokerage + txn)
    sebi_pct: float = 0.000001               # 0.0001 %
    stamp_buy_pct: float = 0.00003           # 0.003 %
    slippage_bps_per_side: float = 5.0       # 5 bps each side

    def buy_cost(self, qty: float, price: float) -> float:
        notional = qty * price
        slip = notional * (self.slippage_bps_per_side / 10000)
        txn = notional * self.exchange_txn_pct / 2
        gst = (self.brokerage_per_order + txn) * self.gst_pct
        return (
            self.brokerage_per_order
            + notional * self.stt_buy_pct
            + txn
            + gst
            + notional * self.sebi_pct
            + notional * self.stamp_buy_pct
            + slip
        )

    def sell_cost(self, qty: float, price: float) -> float:
        notional = qty * price
        slip = notional * (self.slippage_bps_per_side / 10000)
        txn = notional * self.exchange_txn_pct / 2
        gst = (self.brokerage_per_order + txn) * self.gst_pct
        return (
            self.brokerage_per_order
            + notional * self.stt_sell_pct
            + txn
            + gst
            + notional * self.sebi_pct
            + slip
        )


def _annualised_sharpe(rets: np.ndarray) -> float:
    if rets.size < 2:
        return 0.0
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return mu / sd * math.sqrt(252)


def _max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    return float(np.max(dd) * 100)


def _simulate_costed(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    signals: np.ndarray,
    initial_capital: float,
    sl_pct: float,
    tp_pct: float,
    cost: CostModel,
) -> Dict:
    """Bar-by-bar simulator with the existing SL/TP logic + Indian costs."""
    capital = initial_capital
    position = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    trail_price = 0.0
    peak_in_trade = 0.0
    equity_curve = np.empty(len(closes), dtype=float)
    trade_returns: List[float] = []
    sl_pct_dec = sl_pct / 100
    tp_pct_dec = tp_pct / 100

    for i in range(len(closes)):
        price = float(closes[i])
        hi = float(highs[i])
        lo = float(lows[i])
        a = float(atr[i]) if not np.isnan(atr[i]) else price * 0.02
        sig = int(signals[i]) if i < len(signals) else 0

        if position > 0:
            peak_in_trade = max(peak_in_trade, hi)
            profit_atr = (peak_in_trade - entry_price) / a if a > 0 else 0
            if profit_atr > 2.0:
                trail_price = max(trail_price, peak_in_trade - a * 1.5)

            exit_price: Optional[float] = None
            if lo <= sl_price:
                exit_price = sl_price
            elif trail_price > sl_price and lo <= trail_price:
                exit_price = trail_price
            elif hi >= tp_price:
                exit_price = tp_price
            elif sig == -1:
                exit_price = price

            if exit_price is not None:
                proceeds = position * exit_price - cost.sell_cost(position, exit_price)
                pnl_pct = (proceeds / (entry_price * position)) - 1
                trade_returns.append(pnl_pct)
                capital += proceeds
                position = 0

        if position == 0 and sig == 1 and capital > price:
            target_qty = int(capital * 0.93 / price)
            if target_qty >= 1:
                trial_cost = cost.buy_cost(target_qty, price)
                while target_qty > 0 and target_qty * price + trial_cost > capital:
                    target_qty -= 1
                    trial_cost = cost.buy_cost(target_qty, price)
                if target_qty >= 1:
                    position = target_qty
                    entry_price = price
                    peak_in_trade = price
                    sl_atr = max(sl_pct_dec, (a * 2.0) / price)
                    tp_atr = max(tp_pct_dec, (a * 4.0) / price)
                    sl_price = price * (1 - sl_atr)
                    tp_price = price * (1 + tp_atr)
                    trail_price = sl_price
                    capital -= position * price + trial_cost

        equity_curve[i] = capital + position * price

    if position > 0:
        capital += position * float(closes[-1]) - cost.sell_cost(position, float(closes[-1]))
        equity_curve[-1] = capital
        position = 0

    rets = np.diff(equity_curve) / np.where(equity_curve[:-1] > 0, equity_curve[:-1], 1)
    return {
        "equity": equity_curve,
        "returns": rets,
        "trade_returns": np.array(trade_returns) if trade_returns else np.array([0.0]),
        "final_value": float(equity_curve[-1]),
        "sharpe": _annualised_sharpe(rets),
        "max_dd_pct": _max_drawdown_pct(equity_curve),
        "n_trades": len(trade_returns),
    }


class WalkForwardBacktester:
    """Anchored walk-forward over the existing Backtester's strategies."""

    def __init__(self):
        self._inner: Backtester = get_backtester()

    def _strategy_fn(self, name: str):
        # Reuse the Backtester's strategy dispatch table.
        return {
            "adx_trend_follow":   self._inner._adx_trend_follow,
            "rsi_pullback":       self._inner._rsi_pullback,
            "macd_histogram":     self._inner._macd_histogram,
            "donchian_breakout":  self._inner._donchian_breakout,
            "ma_ribbon":          self._inner._ma_ribbon,
            "bb_mean_reversion":  self._inner._bb_mean_reversion,
            "supertrend":         self._inner._supertrend_strategy,
            "price_action":       self._inner._price_action,
            "rsi_ma_combo":       self._inner._rsi_ma_combo,
            "swing_continuation": self._inner._swing_continuation,
        }.get(name, self._inner._adx_trend_follow)

    def run(
        self,
        symbol: str,
        strategy: str = "adx_trend_follow",
        days: int = 750,
        train_window: int = 252,        # ~1y
        test_window: int = 63,          # ~1q
        initial_capital: float = 100000.0,
        stop_loss_pct: float = 2.5,
        target_pct: float = 7.0,
        bootstrap_n: int = 1000,
        cost: Optional[CostModel] = None,
    ) -> Dict:
        symbol = symbol.upper()
        cost = cost or CostModel()
        try:
            df = self._inner._load_data(symbol, days)
            if df is None or len(df) < train_window + test_window + 30:
                return {
                    "error": (
                        f"Need at least {train_window + test_window + 30} bars "
                        f"({train_window} train + {test_window} test + 30 warmup). "
                        f"Got {0 if df is None else len(df)} for {symbol}."
                    )
                }
            df = df.reset_index(drop=True)
            ind = self._inner._precompute(df)
            fn = self._strategy_fn(strategy)
            signals_full = fn(df, ind).fillna(0).astype(int).values
            closes = df["close"].astype(float).values
            highs = df["high"].astype(float).values
            lows = df["low"].astype(float).values
            atr = ind["atr14"].values
            dates = (
                pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()
                if "date" in df.columns
                else [str(i) for i in range(len(df))]
            )

            n = len(df)
            warmup = 30  # let early indicators settle
            t_start = warmup + train_window
            folds: List[Dict] = []
            is_returns: List[float] = []
            oos_returns: List[float] = []
            oos_drawdowns: List[float] = []

            t = t_start
            while t + test_window <= n:
                train_lo = t - train_window
                train_hi = t
                test_lo = t
                test_hi = t + test_window

                is_run = _simulate_costed(
                    closes[train_lo:train_hi],
                    highs[train_lo:train_hi],
                    lows[train_lo:train_hi],
                    atr[train_lo:train_hi],
                    signals_full[train_lo:train_hi],
                    initial_capital, stop_loss_pct, target_pct, cost,
                )
                oos_run = _simulate_costed(
                    closes[test_lo:test_hi],
                    highs[test_lo:test_hi],
                    lows[test_lo:test_hi],
                    atr[test_lo:test_hi],
                    signals_full[test_lo:test_hi],
                    initial_capital, stop_loss_pct, target_pct, cost,
                )

                is_returns.extend(is_run["returns"].tolist())
                oos_returns.extend(oos_run["returns"].tolist())
                oos_drawdowns.append(oos_run["max_dd_pct"])

                folds.append({
                    "train_from": dates[train_lo],
                    "train_to":   dates[train_hi - 1],
                    "test_from":  dates[test_lo],
                    "test_to":    dates[test_hi - 1],
                    "is_sharpe":  round(is_run["sharpe"], 3),
                    "oos_sharpe": round(oos_run["sharpe"], 3),
                    "oos_return_pct": round((oos_run["final_value"] - initial_capital) / initial_capital * 100, 2),
                    "oos_max_dd_pct": round(oos_run["max_dd_pct"], 2),
                    "oos_trades": oos_run["n_trades"],
                })
                t += test_window

            if not folds:
                return {"error": "No folds generated — try a smaller train_window or longer history."}

            is_arr = np.array(is_returns)
            oos_arr = np.array(oos_returns)

            is_sharpe = _annualised_sharpe(is_arr)
            oos_sharpe = _annualised_sharpe(oos_arr)
            sharpe_gap = is_sharpe - oos_sharpe

            # Bootstrap Sharpe CDF on IS returns (i.i.d. resampling with replacement)
            rng = np.random.default_rng(42)
            if is_arr.size >= 30:
                idx = rng.integers(0, is_arr.size, size=(bootstrap_n, is_arr.size))
                samples = is_arr[idx]
                mu = samples.mean(axis=1)
                sd = samples.std(axis=1, ddof=1)
                sd = np.where(sd <= 1e-12, 1e-12, sd)
                boot_sharpe = mu / sd * math.sqrt(252)
            else:
                boot_sharpe = np.array([is_sharpe])

            pct = lambda p: float(np.percentile(boot_sharpe, p))
            cdf_xs = np.percentile(boot_sharpe, np.arange(0, 101, 5)).tolist()

            # Overfit verdict
            if sharpe_gap > 1.0:
                verdict = "Severely overfit"
            elif sharpe_gap > 0.5:
                verdict = "Likely overfit"
            elif sharpe_gap > 0.0:
                verdict = "Mild IS edge"
            else:
                verdict = "OOS ≥ IS (robust or noisy)"

            return {
                "symbol": symbol,
                "strategy": strategy,
                "bars": int(n),
                "folds": folds,
                "n_folds": len(folds),
                "train_window": train_window,
                "test_window": test_window,
                "in_sample": {
                    "sharpe": round(is_sharpe, 3),
                    "n_returns": int(is_arr.size),
                },
                "out_of_sample": {
                    "sharpe": round(oos_sharpe, 3),
                    "n_returns": int(oos_arr.size),
                    "mean_max_dd_pct": round(float(np.mean(oos_drawdowns)), 2),
                    "p95_max_dd_pct":  round(float(np.percentile(oos_drawdowns, 95)), 2),
                },
                "sharpe_gap": round(sharpe_gap, 3),
                "overfit_verdict": verdict,
                "bootstrap": {
                    "n": int(bootstrap_n),
                    "sharpe_p05": round(pct(5), 3),
                    "sharpe_p25": round(pct(25), 3),
                    "sharpe_p50": round(pct(50), 3),
                    "sharpe_p75": round(pct(75), 3),
                    "sharpe_p95": round(pct(95), 3),
                    "sharpe_cdf_xs": [round(x, 3) for x in cdf_xs],
                },
                "drawdowns": [round(d, 2) for d in oos_drawdowns],
                "cost_model": asdict(cost),
            }
        except Exception as e:
            logger.error(f"WalkForward error [{symbol}/{strategy}]: {e}", exc_info=True)
            return {"error": str(e)}


_wf: Optional[WalkForwardBacktester] = None


def get_walkforward() -> WalkForwardBacktester:
    global _wf
    if _wf is None:
        _wf = WalkForwardBacktester()
    return _wf
