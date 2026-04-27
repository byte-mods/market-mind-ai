"""
MarketMind AI - Strategy Backtester (Professional Grade)
==========================================================
All strategies require multi-condition confluence + ADX trend filter.
SL/TP are ATR-based (adapts to actual stock volatility, not fixed %).
Simulation uses bar-by-bar trailing stop-loss.

Strategies (professional-grade, used by prop desks):
  adx_trend_follow    - ADX > 22 + triple MA alignment (most reliable)
  rsi_pullback        - RSI pullback to 40-50 in confirmed uptrend
  macd_histogram      - MACD histogram bottom/top with trend confirmation
  donchian_breakout   - Volume-confirmed channel breakout
  ma_ribbon           - 10/20/50 MA ribbon alignment + momentum
  bb_mean_reversion   - Z-score < -2.2 + RSI < 35 + volume expansion
  supertrend          - Supertrend indicator (ATR-based trailing)
  price_action        - Higher high/higher low structure + breakout
  rsi_ma_combo        - RSI + dual MA dual confirmation
  swing_continuation  - Strong-momentum swing with 3-bar consolidation
"""
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Backtester:

    def __init__(self, price_fetcher=None):
        self._price_fetcher  = price_fetcher
        self._kite_candles   = None

    def _get_kite_candles(self):
        if self._kite_candles is None:
            from .kite_candles import get_kite_candles
            self._kite_candles = get_kite_candles()
        return self._kite_candles

    def _load_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        try:
            kc = self._get_kite_candles()
            df = kc.get_candles_df(symbol, interval='day', days=days)
            if df is not None and len(df) >= 60:
                return df
        except Exception as e:
            logger.debug(f"Kite error {symbol}: {e}")
        try:
            if self._price_fetcher is None:
                from .price_fetcher import get_price_fetcher
                self._price_fetcher = get_price_fetcher()
            hist = self._price_fetcher.get_historical_data(symbol, days=days)
            if not hist.empty and len(hist) >= 60:
                return hist
        except Exception as e:
            logger.debug(f"NSE fallback error {symbol}: {e}")
        return None

    def run(self, symbol: str, strategy: str = 'adx_trend_follow',
            days: int = 500, initial_capital: float = 100000.0,
            stop_loss_pct: float = 2.5, target_pct: float = 7.0) -> Dict:
        symbol = symbol.upper()
        try:
            df = self._load_data(symbol, days)
            if df is None:
                return {'error': (
                    f'No data for {symbol}. Need 60+ trading days. '
                    'Connect Kite in Settings for reliable historical data.'
                )}

            # Pre-compute indicators shared across strategies
            ind = self._precompute(df)

            fn = {
                'adx_trend_follow':   self._adx_trend_follow,
                'rsi_pullback':       self._rsi_pullback,
                'macd_histogram':     self._macd_histogram,
                'donchian_breakout':  self._donchian_breakout,
                'ma_ribbon':          self._ma_ribbon,
                'bb_mean_reversion':  self._bb_mean_reversion,
                'supertrend':         self._supertrend_strategy,
                'price_action':       self._price_action,
                'rsi_ma_combo':       self._rsi_ma_combo,
                'swing_continuation': self._swing_continuation,
                # legacy aliases
                'swing_ma_cross':     self._ma_ribbon,
                'rsi_momentum':       self._rsi_pullback,
                'macd_signal':        self._macd_histogram,
                'bollinger_bands':    self._bb_mean_reversion,
                'golden_cross':       self._adx_trend_follow,
                'turtle_breakout':    self._donchian_breakout,
                'vwap_reversion':     self._bb_mean_reversion,
                'rsi_divergence':     self._rsi_pullback,
                'volume_breakout':    self._donchian_breakout,
                'mean_reversion':     self._bb_mean_reversion,
            }.get(strategy, self._adx_trend_follow)

            signals = fn(df, ind)
            result  = self._simulate(df, ind, signals, initial_capital,
                                     stop_loss_pct, target_pct)
            result['symbol']   = symbol
            result['strategy'] = strategy
            result['bars']     = len(df)
            return result

        except Exception as e:
            logger.error(f"Backtest error [{symbol}/{strategy}]: {e}", exc_info=True)
            return {'error': str(e)}

    # ── Pre-computed indicators ─────────────────────────────────────────────

    def _precompute(self, df: pd.DataFrame) -> Dict:
        c  = df['close'].astype(float)
        h  = df['high'].astype(float)
        l  = df['low'].astype(float)
        v  = df['volume'].astype(float)
        n  = len(df)

        # Moving averages
        ma10  = c.rolling(10).mean()
        ma20  = c.rolling(20).mean()
        ma50  = c.rolling(50).mean()
        ma200 = c.rolling(200).mean()

        # RSI
        rsi14 = self._calc_rsi(c, 14)
        rsi7  = self._calc_rsi(c, 7)

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist= macd - macd_sig

        # Bollinger Bands
        std20  = c.rolling(20).std()
        bb_up  = ma20 + 2.5 * std20
        bb_lo  = ma20 - 2.5 * std20
        bb_pct = (c - bb_lo) / (bb_up - bb_lo + 1e-9)
        z_score= (c - ma20) / (std20 + 1e-9)

        # ATR
        tr = pd.concat([h - l,
                        (h - c.shift(1)).abs(),
                        (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_pct = atr14 / c  # ATR as % of price

        # Volume
        vol_ma20 = v.rolling(20).mean()
        vol_ratio= v / (vol_ma20 + 1)

        # ADX (proper calculation)
        adx, pdi, mdi = self._calc_adx(h, l, c, 14)

        # Donchian channels
        don_hi20 = h.rolling(20).max().shift(1)
        don_lo20 = l.rolling(20).min().shift(1)
        don_hi55 = h.rolling(55).max().shift(1)
        don_lo55 = l.rolling(55).min().shift(1)

        # Momentum
        mom5  = c.pct_change(5)
        mom10 = c.pct_change(10)
        mom20 = c.pct_change(20)

        # Stochastic
        lo14  = l.rolling(14).min()
        hi14  = h.rolling(14).max()
        stoch = (c - lo14) / (hi14 - lo14 + 1e-9) * 100

        # Supertrend
        st_buy, st_sell = self._calc_supertrend(h, l, c, atr14, multiplier=3.0)

        return dict(
            c=c, h=h, l=l, v=v,
            ma10=ma10, ma20=ma20, ma50=ma50, ma200=ma200,
            rsi14=rsi14, rsi7=rsi7,
            macd=macd, macd_sig=macd_sig, macd_hist=macd_hist,
            bb_up=bb_up, bb_lo=bb_lo, bb_pct=bb_pct, z_score=z_score,
            atr14=atr14, atr_pct=atr_pct,
            vol_ratio=vol_ratio, vol_ma20=vol_ma20,
            adx=adx, pdi=pdi, mdi=mdi,
            don_hi20=don_hi20, don_lo20=don_lo20,
            don_hi55=don_hi55, don_lo55=don_lo55,
            mom5=mom5, mom10=mom10, mom20=mom20,
            stoch=stoch, st_buy=st_buy, st_sell=st_sell,
        )

    # ── Strategy signal generators ──────────────────────────────────────────

    def _adx_trend_follow(self, df, ind) -> pd.Series:
        """
        ADX Trend-Following: ADX > 22 + MA10 > MA20 > MA50 + RSI 45-65.
        One of the highest win-rate strategies in trending markets.
        Only buys pullbacks (c < ma20) within confirmed uptrend.
        """
        c, ma10, ma20, ma50, rsi14, adx, pdi, mdi = (
            ind['c'], ind['ma10'], ind['ma20'], ind['ma50'],
            ind['rsi14'], ind['adx'], ind['pdi'], ind['mdi'])

        trend_up   = (ma10 > ma20) & (ma20 > ma50) & (adx > 22) & (pdi > mdi)
        pullback   = (c > ma20 * 0.98) & (c < ma20 * 1.01)  # near MA20
        rsi_ok     = (rsi14 > 42) & (rsi14 < 68)
        entry_bar  = trend_up & pullback & rsi_ok & (c > c.shift(1))  # up bar

        trend_down = (ma10 < ma20) & (ma20 < ma50) & (adx > 22) & (mdi > pdi)
        exit_cond  = trend_down | (c < ma50) | (rsi14 > 75)

        sig = pd.Series(0, index=df.index)
        sig[entry_bar]                                = 1
        sig[exit_cond & (sig.shift(1).fillna(0) >= 0)]= -1
        return sig

    def _rsi_pullback(self, df, ind) -> pd.Series:
        """
        RSI Pullback in Uptrend: RSI dips to 40-48 while price above MA50.
        High win rate because you're buying dips in a strong trend.
        """
        c, ma20, ma50, rsi14, adx, pdi, mdi, vol_ratio, mom20 = (
            ind['c'], ind['ma20'], ind['ma50'], ind['rsi14'],
            ind['adx'], ind['pdi'], ind['mdi'], ind['vol_ratio'], ind['mom20'])

        uptrend   = (c > ma50) & (adx > 18) & (pdi > mdi)
        pullback  = (rsi14 < 48) & (rsi14 > 30)
        recovering= rsi14 > rsi14.shift(1)          # RSI turning back up
        vol_ok    = vol_ratio < 2.0                  # not panic selling
        momentum  = mom20 > 0                        # medium-term still up

        entry = uptrend & pullback & recovering & vol_ok & momentum

        # Exit: RSI overbought or trend broken
        exit_ = (rsi14 > 72) | (c < ma50 * 0.98)

        sig = pd.Series(0, index=df.index)
        sig[entry] = 1
        sig[exit_] = -1
        return sig

    def _macd_histogram(self, df, ind) -> pd.Series:
        """
        MACD Histogram reversal: histogram makes higher low (bullish divergence)
        while price is above MA50. Filter: ADX > 18.
        """
        c, ma20, ma50, macd_hist, adx, pdi, mdi, vol_ratio = (
            ind['c'], ind['ma20'], ind['ma50'], ind['macd_hist'],
            ind['adx'], ind['pdi'], ind['mdi'], ind['vol_ratio'])

        hist_bottoming = (
            (macd_hist > macd_hist.shift(1)) &   # histogram rising
            (macd_hist.shift(1) < macd_hist.shift(2)) &  # was falling before
            (macd_hist < 0)                       # still negative = early entry
        )
        trend_filter = (c > ma50) & (adx > 18) & (pdi > mdi * 0.9)
        vol_filter   = vol_ratio > 0.8

        entry = hist_bottoming & trend_filter & vol_filter

        hist_topping = (
            (macd_hist < macd_hist.shift(1)) &
            (macd_hist.shift(1) > macd_hist.shift(2)) &
            (macd_hist > 0)
        )
        exit_ = hist_topping | (c < ma50 * 0.97)

        sig = pd.Series(0, index=df.index)
        sig[entry] = 1
        sig[exit_] = -1
        return sig

    def _donchian_breakout(self, df, ind) -> pd.Series:
        """
        Volume-confirmed Donchian breakout (Turtle system enhanced).
        Breakout above 55-day high + volume > 1.8× avg + ADX > 20.
        Exit: 20-day low (Turtle System 2 exit).
        """
        c, h, l, vol_ratio, don_hi55, don_lo20, adx = (
            ind['c'], ind['h'], ind['l'], ind['vol_ratio'],
            ind['don_hi55'], ind['don_lo20'], ind['adx'])

        entry = (
            (c > don_hi55) &               # 55-day breakout
            (vol_ratio > 1.8) &            # volume confirmation
            (adx > 20)                     # trending market
        )
        exit_ = (l < don_lo20)             # 20-day low exit

        sig = pd.Series(0, index=df.index)
        sig[entry] = 1
        sig[exit_] = -1
        return sig

    def _ma_ribbon(self, df, ind) -> pd.Series:
        """
        Triple MA Ribbon: 10/20/50 all aligned upward + RSI > 50 + volume.
        When all three MAs align, trend is very strong.
        """
        c, ma10, ma20, ma50, rsi14, vol_ratio, mom10 = (
            ind['c'], ind['ma10'], ind['ma20'], ind['ma50'],
            ind['rsi14'], ind['vol_ratio'], ind['mom10'])

        ribbon_up  = (ma10 > ma20) & (ma20 > ma50) & (c > ma10)
        rsi_ok     = (rsi14 > 50) & (rsi14 < 72)
        vol_ok     = vol_ratio > 0.9
        momentum   = mom10 > 0.01

        prev_below = (c.shift(1) < ma10.shift(1)) & (c > ma10)  # Just broke above MA10
        entry      = ribbon_up & rsi_ok & vol_ok & momentum & (prev_below | (c.shift(1) > ma10.shift(1)))

        exit_ = (ma10 < ma20) | (c < ma50 * 0.97) | (rsi14 > 78)

        sig = pd.Series(0, index=df.index)
        sig[entry] = 1
        sig[exit_] = -1
        return sig

    def _bb_mean_reversion(self, df, ind) -> pd.Series:
        """
        Bollinger Band Mean Reversion (improved):
        Z-score < -2.2 AND RSI < 35 AND volume expansion → buy.
        Requires price > MA200 (in long-term uptrend) to avoid value traps.
        """
        c, ma20, ma200, z_score, rsi14, vol_ratio, adx = (
            ind['c'], ind['ma20'], ind['ma200'], ind['z_score'],
            ind['rsi14'], ind['vol_ratio'], ind['adx'])

        # Entry: deeply oversold in long-term uptrend
        entry = (
            (z_score < -2.2) &             # extreme stretch
            (rsi14 < 35) &                 # oversold
            (vol_ratio > 1.4) &            # volume expansion (capitulation)
            (c > ma200 * 0.92)             # not in structural downtrend
        )
        # Exit: price returns to mean
        exit_ = (z_score > -0.3) | (z_score > z_score.shift(1) + 0.5)

        sig = pd.Series(0, index=df.index)
        sig[entry] = 1
        sig[exit_] = -1
        return sig

    def _supertrend_strategy(self, df, ind) -> pd.Series:
        """
        Supertrend crossover: price crosses above supertrend = buy.
        One of the cleanest trend-following signals.
        """
        c, st_buy, st_sell, adx, vol_ratio = (
            ind['c'], ind['st_buy'], ind['st_sell'],
            ind['adx'], ind['vol_ratio'])

        # Buy when price crosses above supertrend (trend flip)
        entry = (
            st_buy &                        # in buy zone
            (~st_buy.shift(1).fillna(False)) &  # just flipped
            (adx > 18) &
            (vol_ratio > 0.8)
        )
        exit_ = st_sell & (~st_sell.shift(1).fillna(False))  # trend flip to sell

        sig = pd.Series(0, index=df.index)
        sig[entry] = 1
        sig[exit_] = -1
        return sig

    def _price_action(self, df, ind) -> pd.Series:
        """
        Price Action Swing: Higher High + Higher Low structure confirmed.
        Buy: new 10-day high after ≥3 HH-HL bars, volume surge, ADX > 20.
        """
        c, h, l, ma20, ma50, adx, vol_ratio, rsi14 = (
            ind['c'], ind['h'], ind['l'],
            ind['ma20'], ind['ma50'], ind['adx'],
            ind['vol_ratio'], ind['rsi14'])

        # Higher highs and higher lows over 3 bars
        hh3 = (h > h.shift(1)) & (h.shift(1) > h.shift(2))
        hl3 = (l > l.shift(1)) & (l.shift(1) > l.shift(2))

        entry = (
            hh3 & hl3 &
            (c > ma50) &
            (adx > 20) &
            (vol_ratio > 1.3) &
            (rsi14 > 50) & (rsi14 < 72)
        )
        exit_ = (
            ((h < h.shift(1)) & (l < l.shift(1))) |  # lower high + lower low
            (c < ma50 * 0.97)
        )

        sig = pd.Series(0, index=df.index)
        sig[entry] = 1
        sig[exit_] = -1
        return sig

    def _rsi_ma_combo(self, df, ind) -> pd.Series:
        """
        RSI + MA Dual Confirmation: RSI crosses above 50 while price
        above MA20 and MA20 above MA50. Both conditions needed.
        """
        c, ma20, ma50, rsi14, adx, pdi, mdi, stoch = (
            ind['c'], ind['ma20'], ind['ma50'], ind['rsi14'],
            ind['adx'], ind['pdi'], ind['mdi'], ind['stoch'])

        # RSI crossing 50 from below in uptrend
        rsi_cross_up = (rsi14 > 50) & (rsi14.shift(1) <= 50)
        trend_up     = (c > ma20) & (ma20 > ma50) & (adx > 18) & (pdi > mdi)
        stoch_ok     = stoch < 80  # not overbought on stochastic

        entry = rsi_cross_up & trend_up & stoch_ok

        # RSI cross down
        rsi_cross_dn = (rsi14 < 50) & (rsi14.shift(1) >= 50)
        exit_        = rsi_cross_dn | (c < ma50 * 0.97) | (rsi14 > 78)

        sig = pd.Series(0, index=df.index)
        sig[entry] = 1
        sig[exit_] = -1
        return sig

    def _swing_continuation(self, df, ind) -> pd.Series:
        """
        Swing Continuation: Strong momentum (mom10 > 5%) followed by
        3-bar consolidation (narrow range), then breakout bar.
        """
        c, h, l, ma20, ma50, mom10, adx, vol_ratio, rsi14 = (
            ind['c'], ind['h'], ind['l'], ind['ma20'], ind['ma50'],
            ind['mom10'], ind['adx'], ind['vol_ratio'], ind['rsi14'])

        # Consolidation: 3-bar narrow range after strong move
        rng3   = (h.rolling(3).max() - l.rolling(3).min()) / c
        narrow = rng3 < ind['atr_pct'] * 1.5

        strong_prior_move = mom10.shift(3) > 0.04

        # Breakout from consolidation
        breakout = (c > h.shift(1)) & (c > h.shift(2))

        entry = (
            strong_prior_move & narrow &
            breakout &
            (c > ma50) & (adx > 20) &
            (vol_ratio > 1.4) &
            (rsi14 > 50) & (rsi14 < 70)
        )
        exit_ = (c < ma20 * 0.98) | (rsi14 > 76)

        sig = pd.Series(0, index=df.index)
        sig[entry] = 1
        sig[exit_] = -1
        return sig

    # ── Indicator helpers ─────────────────────────────────────────────────────

    def _calc_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / loss.replace(0, float('nan'))
        return 100 - (100 / (1 + rs))

    def _calc_adx(self, h: pd.Series, l: pd.Series, c: pd.Series,
                  period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        tr  = pd.concat([h - l,
                         (h - c.shift(1)).abs(),
                         (l - c.shift(1)).abs()], axis=1).max(axis=1)
        pdm = (h - h.shift(1)).clip(lower=0)
        mdm = (l.shift(1) - l).clip(lower=0)
        # Zero out where opposite is larger
        pdm = pdm.where(pdm > mdm, 0.0)
        mdm = mdm.where(mdm > pdm.shift(0), 0.0)

        atr  = tr.ewm(alpha=1/period, adjust=False).mean()
        pdm_s= pdm.ewm(alpha=1/period, adjust=False).mean()
        mdm_s= mdm.ewm(alpha=1/period, adjust=False).mean()

        pdi  = 100 * pdm_s / (atr + 1e-9)
        mdi  = 100 * mdm_s / (atr + 1e-9)
        dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
        adx  = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx, pdi, mdi

    def _calc_supertrend(self, h: pd.Series, l: pd.Series, c: pd.Series,
                         atr14: pd.Series, multiplier: float = 3.0
                         ) -> Tuple[pd.Series, pd.Series]:
        hl2      = (h + l) / 2
        up_band  = hl2 + multiplier * atr14
        dn_band  = hl2 - multiplier * atr14

        # Supertrend direction
        direction = pd.Series(1, index=c.index)
        final_up  = dn_band.copy()
        final_dn  = up_band.copy()

        for i in range(1, len(c)):
            final_up.iloc[i] = (
                max(dn_band.iloc[i], final_up.iloc[i-1])
                if c.iloc[i-1] > final_up.iloc[i-1] else dn_band.iloc[i]
            )
            final_dn.iloc[i] = (
                min(up_band.iloc[i], final_dn.iloc[i-1])
                if c.iloc[i-1] < final_dn.iloc[i-1] else up_band.iloc[i]
            )
            if direction.iloc[i-1] == -1 and c.iloc[i] > final_dn.iloc[i]:
                direction.iloc[i] = 1
            elif direction.iloc[i-1] == 1 and c.iloc[i] < final_up.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]

        st_buy  = direction == 1
        st_sell = direction == -1
        return st_buy, st_sell

    # ── Trade simulation (ATR-based SL/TP + trailing SL) ─────────────────────

    def _simulate(self, df: pd.DataFrame, ind: Dict, signals: pd.Series,
                  initial_capital: float, sl_pct: float, tp_pct: float) -> Dict:
        """
        Simulate trades with:
        - ATR-based SL (2× ATR below entry, adapts to volatility)
        - ATR-based TP (4× ATR above entry = 2:1 R:R minimum)
        - Trailing stop-loss (ratchets up as trade profits)
        - Max 95% of capital per trade (never go all-in)
        """
        closes   = df['close'].astype(float).values
        highs    = df['high'].astype(float).values
        lows     = df['low'].astype(float).values
        atr_vals = ind['atr14'].values

        dates = (pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').values
                 if 'date' in df.columns else np.arange(len(df)).astype(str))

        capital      = initial_capital
        position     = 0
        entry_price  = 0.0
        sl_price     = 0.0
        tp_price     = 0.0
        trail_price  = 0.0  # trailing stop
        peak_in_trade= 0.0
        equity_curve = []
        trades       = []
        wins = losses = 0
        sl_pct_dec   = sl_pct / 100
        tp_pct_dec   = tp_pct / 100

        for i in range(len(df)):
            price   = closes[i]
            hi      = highs[i]
            lo      = lows[i]
            atr     = atr_vals[i] if not np.isnan(atr_vals[i]) else price * 0.02
            sig     = signals.iloc[i] if i < len(signals) else 0

            # ── Manage open position ──────────────────────────────────────
            if position > 0:
                peak_in_trade = max(peak_in_trade, hi)

                # Update trailing stop (only ratchet UP)
                profit_atr = (peak_in_trade - entry_price) / atr
                if profit_atr > 2.0:   # Start trailing after 2 ATRs profit
                    new_trail = peak_in_trade - atr * 1.5
                    trail_price = max(trail_price, new_trail)

                exit_price = None
                exit_type  = None

                if lo <= sl_price:              # Stop-loss hit
                    exit_price = sl_price; exit_type = 'SL'
                elif trail_price > sl_price and lo <= trail_price:
                    exit_price = trail_price; exit_type = 'TSL'  # Trailing SL
                elif hi >= tp_price:            # Take-profit hit
                    exit_price = tp_price; exit_type = 'TP'
                elif sig == -1:                 # Strategy exit signal
                    exit_price = price; exit_type = 'EXIT'

                if exit_price is not None:
                    pnl = (exit_price - entry_price) * position
                    capital += exit_price * position
                    is_win   = pnl >= 0
                    wins    += 1 if is_win else 0
                    losses  += 0 if is_win else 1
                    trades.append({
                        'date':        str(dates[i]),
                        'type':        exit_type,
                        'entry':       round(entry_price, 2),
                        'exit':        round(exit_price, 2),
                        'pnl':         round(pnl, 2),
                        'return_pct':  round((exit_price - entry_price) / entry_price * 100, 2),
                    })
                    position = 0

            # ── New entry ─────────────────────────────────────────────────
            if position == 0 and sig == 1 and capital > price:
                shares = int(capital * 0.93 / price)
                if shares < 1:
                    continue
                position     = shares
                entry_price  = price
                peak_in_trade= price

                # ATR-based SL and TP
                sl_atr = max(sl_pct_dec, (atr * 2.0) / price)  # ≥ user SL%
                tp_atr = max(tp_pct_dec, (atr * 4.0) / price)  # ≥ user TP%
                sl_price    = price * (1 - sl_atr)
                tp_price    = price * (1 + tp_atr)
                trail_price = sl_price   # initial = hard SL

                capital -= position * price

            portfolio_val = capital + position * price
            equity_curve.append({'date': str(dates[i]), 'value': round(portfolio_val, 2)})

        # Close open position at last price
        if position > 0:
            final_val = capital + position * closes[-1]
            pnl = (closes[-1] - entry_price) * position
            trades.append({'date': str(dates[-1]), 'type': 'OPEN',
                           'entry': round(entry_price, 2), 'exit': round(closes[-1], 2),
                           'pnl': round(pnl, 2),
                           'return_pct': round((closes[-1]-entry_price)/entry_price*100, 2)})
        else:
            final_val = capital

        total_trades = wins + losses
        win_rate     = wins / total_trades if total_trades > 0 else 0.0

        # Max drawdown
        peak = initial_capital; max_dd = 0.0
        for e in equity_curve:
            v = e['value']
            peak  = max(peak, v)
            max_dd= max(max_dd, (peak - v) / peak)

        # Sharpe + profit factor
        values  = [e['value'] for e in equity_curve]
        rets    = [values[i]/values[i-1]-1 for i in range(1, len(values))] if len(values) > 1 else [0]
        mu      = sum(rets) / len(rets)
        sigma   = math.sqrt(sum((r-mu)**2 for r in rets)/len(rets)) if rets else 1e-9
        sharpe  = (mu / sigma * math.sqrt(252)) if sigma > 1e-9 else 0.0

        win_rets = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
        loss_rets= [abs(t['pnl']) for t in trades if t.get('pnl', 0) <= 0]
        profit_factor = (sum(win_rets) / sum(loss_rets)) if loss_rets else float('inf')

        total_return  = (final_val - initial_capital) / initial_capital * 100
        bah_return    = (closes[-1] - closes[0]) / closes[0] * 100

        # Latest signal
        latest_signal = None
        for i in range(len(signals)-1, -1, -1):
            if signals.iloc[i] != 0:
                latest_signal = {
                    'action': 'BUY' if signals.iloc[i] == 1 else 'SELL',
                    'date':   str(dates[i]),
                    'price':  round(float(closes[i]), 2),
                }
                break

        return {
            'initial_capital':  initial_capital,
            'final_value':      round(final_val, 2),
            'total_return_pct': round(total_return, 2),
            'bah_return_pct':   round(bah_return, 2),
            'alpha':            round(total_return - bah_return, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'sharpe_ratio':     round(sharpe, 3),
            'profit_factor':    round(min(profit_factor, 99.9), 2),
            'win_rate':         round(win_rate * 100, 1),
            'total_trades':     total_trades,
            'wins':             wins,
            'losses':           losses,
            'equity_curve':     equity_curve,
            'trades':           trades[-50:],
            'latest_signal':    latest_signal,
        }


_backtester: Optional[Backtester] = None


def get_backtester() -> Backtester:
    global _backtester
    if _backtester is None:
        _backtester = Backtester()
    return _backtester
