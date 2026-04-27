"""
MarketMind AI - Stock Trading Gym Environment
Gym-compatible environment for PPO training.

Observation: 22 normalized technical features + 3 position features
Action: Discrete(3) — 0=HOLD, 1=BUY, 2=SELL
Reward: Sharpe-like + drawdown penalty
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

ACTION_HOLD = 0
ACTION_BUY  = 1
ACTION_SELL = 2
ACTION_NAMES = ['HOLD', 'BUY', 'SELL']

OBS_COLS = [
    'rsi14', 'rsi7', 'macd_hist', 'macd', 'signal_line',
    'stoch_k', 'bb_pct',
    'c_vs_ma10', 'c_vs_ma20', 'c_vs_ma50', 'c_vs_ma200',
    'mom3', 'mom5', 'mom10', 'mom20',
    'vol_ratio', 'vol10', 'vol20', 'atr_pct', 'daily_ret',
]

# Human-readable descriptions for pattern analysis
OBS_DESCRIPTIONS = {
    'rsi14':      'RSI(14)',
    'rsi7':       'RSI(7)',
    'macd_hist':  'MACD Histogram',
    'macd':       'MACD Line',
    'signal_line':'MACD Signal',
    'stoch_k':    'Stochastic %K',
    'bb_pct':     'Bollinger %B',
    'c_vs_ma10':  'Price vs MA10',
    'c_vs_ma20':  'Price vs MA20',
    'c_vs_ma50':  'Price vs MA50',
    'c_vs_ma200': 'Price vs MA200',
    'mom3':       '3-bar momentum',
    'mom5':       '5-bar momentum',
    'mom10':      '10-bar momentum',
    'mom20':      '20-bar momentum',
    'vol_ratio':  'Volume vs 20-day avg',
    'vol10':      '10-day volatility',
    'vol20':      '20-day volatility',
    'atr_pct':    'ATR%',
    'daily_ret':  'Daily return',
}


class StockTradingEnv:
    """
    OpenAI Gym-compatible stock trading environment.
    Used by PPO trainer for deep RL.
    """

    def __init__(
        self,
        feat: pd.DataFrame,
        initial_capital: float = 100_000.0,
        stop_loss_pct: float = 2.5,
        take_profit_pct: float = 8.0,
        trailing_sl_pct: float = 2.0,
        transaction_cost: float = 0.001,
        reward_scaling: float = 100.0,
        warmup: int = 50,
    ):
        self.feat = feat.reset_index(drop=True)
        self.n = len(self.feat)
        self.initial_capital = initial_capital
        self.sl = stop_loss_pct / 100
        self.tp = take_profit_pct / 100
        self.tsl = trailing_sl_pct / 100
        self.cost = transaction_cost
        self.scale = reward_scaling
        self.warmup = warmup

        self._build_obs_matrix()
        self.obs_size = self._obs_mat.shape[1] + 3  # +position info

        self.reset()

    def _build_obs_matrix(self):
        cols = [c for c in OBS_COLS if c in self.feat.columns]
        self._obs_cols = cols
        mat = self.feat[cols].values.astype(np.float32)

        # Clip to [1%, 99%] percentile then scale to [-1, 1]
        p1  = np.nanpercentile(mat, 1, axis=0)
        p99 = np.nanpercentile(mat, 99, axis=0)
        mat = np.clip(mat, p1, p99)
        rng = p99 - p1
        rng[rng == 0] = 1.0
        self._obs_mat = ((mat - p1) / rng * 2 - 1).astype(np.float32)
        self._obs_p1  = p1
        self._obs_rng = rng

    def reset(self, start: Optional[int] = None) -> np.ndarray:
        safe_warmup     = min(self.warmup, max(0, self.n - 5))
        self.idx        = start if start is not None else safe_warmup
        self.capital    = float(self.initial_capital)
        self.holding    = False
        self.buy_price  = 0.0
        self.peak_price = 0.0
        self.buy_idx    = self.warmup
        self.trades: List[Dict]   = []
        self.equity: List[float]  = [self.initial_capital]
        self.peak_eq    = float(self.initial_capital)
        self.max_dd     = 0.0
        self._ret_buf: List[float] = []
        # Track entry observations for pattern analysis
        self._entry_obs: List[np.ndarray] = []
        self._entry_raw: List[np.ndarray] = []  # raw (unnormalized) at entry
        return self._obs()

    def _obs(self) -> np.ndarray:
        base = self._obs_mat[min(self.idx, self.n - 1)].copy()
        h_f  = 1.0 if self.holding else 0.0
        unr  = 0.0
        if self.holding and self.buy_price > 0:
            cur  = float(self.feat['close'].iloc[self.idx])
            unr  = np.clip((cur - self.buy_price) / self.buy_price, -0.3, 0.3)
        age = (self.idx - self.buy_idx) / max(self.n, 1) if self.holding else 0.0
        return np.concatenate([base, [h_f, unr, age]]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        idx_safe = min(self.idx, self.n - 1)
        cur    = float(self.feat['close'].iloc[idx_safe])
        info   = {'action': action, 'price': cur}
        reason = None

        # Validity
        if action == ACTION_BUY  and self.holding: action = ACTION_HOLD
        if action == ACTION_SELL and not self.holding: action = ACTION_HOLD

        # Risk management overrides
        if self.holding:
            unr  = (cur - self.buy_price) / self.buy_price
            self.peak_price = max(self.peak_price, cur)
            tdd  = (self.peak_price - cur) / self.peak_price
            if unr   <= -self.sl:  action = ACTION_SELL; reason = 'SL'
            elif unr >= self.tp:   action = ACTION_SELL; reason = 'TP'
            elif tdd >= self.tsl:  action = ACTION_SELL; reason = 'TSL'

        reward = 0.0

        if action == ACTION_BUY:
            self.buy_price  = cur * (1 + self.cost)
            self.peak_price = cur
            self.buy_idx    = self.idx
            self.holding    = True
            # Save entry observation for pattern analysis
            self._entry_obs.append(self._obs_mat[self.idx].copy())
            raw = self.feat[self._obs_cols].iloc[self.idx].values.astype(float)
            self._entry_raw.append(raw)

        elif action == ACTION_SELL:
            ret = (cur * (1 - self.cost) - self.buy_price) / self.buy_price
            self.capital *= (1 + ret)
            self._ret_buf.append(ret)

            # Sharpe-like reward
            reward = ret * self.scale
            if ret < 0:
                reward *= 1.5  # amplify loss penalty

            # Consistency bonus: reward low-variance profitable trading
            if len(self._ret_buf) >= 5:
                r_arr = np.array(self._ret_buf[-5:])
                if r_arr.std() > 0:
                    reward += float(r_arr.mean() / r_arr.std()) * 0.5

            trade = {
                'entry':       round(self.buy_price, 2),
                'exit':        round(cur, 2),
                'return_pct':  round(ret * 100, 3),
                'bars_held':   self.idx - self.buy_idx,
                'outcome':     'WIN' if ret > 0 else 'LOSS',
                'exit_reason': reason or 'Signal',
                'entry_idx':   self.buy_idx,
                'exit_idx':    self.idx,
                'entry_obs':   self._entry_obs[-1].tolist() if self._entry_obs else [],
                'entry_raw':   self._entry_raw[-1].tolist() if self._entry_raw else [],
            }
            self.trades.append(trade)
            self.holding   = False
            self.buy_price = 0.0

        else:  # HOLD
            if self.holding:
                unr    = (cur - self.buy_price) / self.buy_price
                reward = unr * 0.005 * self.scale
            else:
                reward = -0.0005  # tiny penalty for never trading

        # Update equity mark
        mark = self.capital
        if self.holding and self.buy_price > 0:
            mark *= (1 + (cur - self.buy_price) / self.buy_price)
        self.equity.append(round(mark, 2))
        self.peak_eq = max(self.peak_eq, mark)
        dd = (self.peak_eq - mark) / self.peak_eq
        self.max_dd = max(self.max_dd, dd)

        self.idx += 1
        done = self.idx >= self.n - 1

        if done and self.holding:
            last = float(self.feat['close'].iloc[-1])
            ret  = (last * (1 - self.cost) - self.buy_price) / self.buy_price
            self.capital *= (1 + ret)
            self.trades.append({
                'entry':       round(self.buy_price, 2),
                'exit':        round(last, 2),
                'return_pct':  round(ret * 100, 3),
                'bars_held':   self.idx - self.buy_idx,
                'outcome':     'WIN' if ret > 0 else 'LOSS',
                'exit_reason': 'End of Data',
                'entry_idx':   self.buy_idx,
                'exit_idx':    self.idx,
                'entry_obs':   self._entry_obs[-1].tolist() if self._entry_obs else [],
                'entry_raw':   self._entry_raw[-1].tolist() if self._entry_raw else [],
            })
            self.holding = False

        next_obs = self._obs() if not done else np.zeros(self.obs_size, dtype=np.float32)
        return next_obs, float(reward), done, info

    def metrics(self) -> Dict:
        total_ret = (self.capital - self.initial_capital) / self.initial_capital * 100
        wins   = [t for t in self.trades if t['outcome'] == 'WIN']
        losses = [t for t in self.trades if 'LOSS' in t['outcome']]
        wr     = len(wins) / len(self.trades) * 100 if self.trades else 0
        aw     = float(np.mean([t['return_pct'] for t in wins]))   if wins   else 0.0
        al     = float(np.mean([t['return_pct'] for t in losses])) if losses else 0.0
        pf     = abs(aw / al) if al != 0 else 99.0  # cap to avoid JSON inf

        rets = np.array([t['return_pct'] / 100 for t in self.trades]) if self.trades else np.array([0.0])
        if len(rets) < 3 or rets.std() < 1e-6:
            sharpe = 0.0
        else:
            sharpe = float((rets.mean() / rets.std()) * np.sqrt(min(252, len(rets))))

        closes = self.feat['close'].values
        bah_start = min(self.warmup, len(closes) - 2)
        bah = (closes[-1] - closes[bah_start]) / max(closes[bah_start], 1e-9) * 100

        return {
            'total_return_pct': round(total_ret, 2),
            'bah_return_pct':   round(float(bah), 2),
            'alpha':            round(total_ret - float(bah), 2),
            'win_rate':         round(wr, 1),
            'total_trades':     len(self.trades),
            'winning_trades':   len(wins),
            'losing_trades':    len(losses),
            'avg_win_pct':      round(aw, 2),
            'avg_loss_pct':     round(al, 2),
            'profit_factor':    round(float(pf), 2),
            'max_drawdown_pct': round(self.max_dd * 100, 2),
            'sharpe_ratio':     round(sharpe, 2),
            'final_capital':    round(self.capital, 2),
        }


def _norm_label(val: float) -> str:
    """Convert a [-1,1] normalized value to a qualitative label."""
    if val >  0.6: return 'very high'
    if val >  0.2: return 'high'
    if val > -0.2: return 'neutral'
    if val > -0.6: return 'low'
    return 'very low'


def _issue_for_feature(col: str, l_norm: float, w_norm: Optional[float]) -> Optional[str]:
    """
    Return a plain-English issue description for a feature given its average normalized
    value at loss entries. l_norm and w_norm are in [-1, 1].
    Returns None if no actionable issue is found.
    """
    label = OBS_DESCRIPTIONS.get(col, col)
    divergence = abs(l_norm - (w_norm or 0))

    # Direction: did losses enter when feature was higher or lower than wins?
    direction = 'higher' if (w_norm is not None and l_norm > w_norm) else 'lower'

    if col in ('rsi14', 'rsi7'):
        if l_norm > 0.3:
            return f"{label} was {_norm_label(l_norm)} at entry — bought when overbought, momentum reversal risk"
        if l_norm < -0.3:
            return f"{label} was {_norm_label(l_norm)} at entry — oversold at entry but trend still falling"
    elif col in ('macd_hist', 'macd', 'signal_line'):
        if l_norm < -0.15:
            strength = 'strongly ' if l_norm < -0.5 else ''
            return f"{label} was {strength}{_norm_label(l_norm)} at entry — entered against bearish momentum"
        if l_norm > 0.15 and w_norm is not None and w_norm < l_norm - 0.15:
            return f"{label} was {_norm_label(l_norm)} — entered when momentum was already extended"
    elif col in ('mom3', 'mom5', 'mom10', 'mom20'):
        if l_norm < -0.15:
            period = col.replace('mom', '')
            return f"{period}-bar momentum was {_norm_label(l_norm)} — entered while price was in a short-term decline"
        if l_norm > 0.4:
            period = col.replace('mom', '')
            return f"{period}-bar momentum was {_norm_label(l_norm)} — entered after a strong move, chasing price"
    elif col == 'c_vs_ma50':
        if l_norm < -0.2:
            return f"Price was {_norm_label(l_norm)} vs MA50 — entered in bearish structure below key trend line"
        if l_norm > 0.4:
            return f"Price was {_norm_label(l_norm)} vs MA50 — entered when price was overextended above MA50"
    elif col == 'c_vs_ma200':
        if l_norm < -0.2:
            return f"Price was {_norm_label(l_norm)} vs MA200 — long-term trend was bearish at entry"
        if l_norm > 0.5:
            return f"Price was {_norm_label(l_norm)} vs MA200 — price was far extended above long-term average"
    elif col == 'vol_ratio':
        if l_norm < -0.2:
            return f"Volume was {_norm_label(l_norm)} vs 20-day avg — low-conviction move, weak institutional participation"
        if l_norm > 0.4:
            return f"Volume was {_norm_label(l_norm)} — high-volume moves often exhaust quickly after entry"
    elif col == 'bb_pct':
        if l_norm > 0.4:
            return f"Price was near Bollinger upper band at entry — entered at resistance, mean-reversion risk"
        if l_norm < -0.4:
            return f"Price was near Bollinger lower band — caught a falling knife, lower band not a reliable floor"
    elif col == 'atr_pct':
        if l_norm > 0.3:
            return f"Volatility (ATR%) was {_norm_label(l_norm)} at entry — high volatility regime, wider stop needed"
    elif col in ('stoch_k',):
        if l_norm > 0.4:
            return f"Stochastic %K was {_norm_label(l_norm)} — entered in overbought stochastic zone"
        if l_norm < -0.4:
            return f"Stochastic %K was {_norm_label(l_norm)} — deep oversold but no reversal confirmation"
    elif col in ('vol10', 'vol20'):
        if l_norm > 0.3:
            return f"Price volatility was {_norm_label(l_norm)} at entry — noisy conditions, signal less reliable"
    elif col in ('c_vs_ma10', 'c_vs_ma20'):
        if l_norm < -0.25:
            ma = col.replace('c_vs_ma', 'MA')
            return f"Price was {_norm_label(l_norm)} vs {ma} — entered below short-term trend line"
        if l_norm > 0.5:
            ma = col.replace('c_vs_ma', 'MA')
            return f"Price was {_norm_label(l_norm)} vs {ma} — short-term overextension at entry"

    # Generic fallback for any feature with large divergence — show direction of difference
    if divergence >= 0.25 and w_norm is not None:
        if l_norm > w_norm:
            return f"{label} was higher at loss entries ({_norm_label(l_norm)}) than at wins ({_norm_label(w_norm)}) — wins entered at lower {label} levels"
        else:
            return f"{label} was lower at loss entries ({_norm_label(l_norm)}) than at wins ({_norm_label(w_norm)}) — wins entered at higher {label} levels"

    return None


def analyze_mistakes(trades: List[Dict], obs_cols: List[str]) -> List[Dict]:
    """
    Analyse which market conditions led to losses using normalized observations.
    Uses entry_obs (already in [-1,1]) for scale-independent comparison.
    Returns top-5 mistake patterns with plain-English descriptions.
    """
    if not trades:
        return []

    wins   = [t for t in trades if t['outcome'] == 'WIN'   and t.get('entry_obs')]
    losses = [t for t in trades if 'LOSS' in t['outcome'] and t.get('entry_obs')]

    if not losses:
        return []

    n_cols   = min(len(obs_cols), len(losses[0]['entry_obs']))
    loss_obs = np.array([t['entry_obs'][:n_cols] for t in losses])
    win_obs  = np.array([t['entry_obs'][:n_cols] for t in wins]) if wins else None

    candidates = []

    for fi, col in enumerate(obs_cols):
        if fi >= n_cols:
            break
        l_norm = float(np.mean(loss_obs[:, fi]))
        w_norm = float(np.mean(win_obs[:, fi])) if win_obs is not None and fi < win_obs.shape[1] else None
        div    = abs(l_norm - w_norm) if w_norm is not None else 0.0

        issue = _issue_for_feature(col, l_norm, w_norm)
        if issue is None:
            continue

        label    = OBS_DESCRIPTIONS.get(col, col)
        contrast = ''
        if w_norm is not None and div >= 0.1:
            contrast = f'Winning trades entered when {label} was {_norm_label(w_norm)} instead.'

        candidates.append({
            'feature':    col,
            'label':      label,
            'issue':      issue,
            'contrast':   contrast,
            'loss_level': _norm_label(l_norm),
            'win_level':  _norm_label(w_norm) if w_norm is not None else '—',
            'divergence': round(div, 3),
            'loss_count': len(losses),
            'win_count':  len(wins),
        })

    # Sort by divergence (most impactful first) and return top 5
    candidates.sort(key=lambda x: x['divergence'], reverse=True)
    return candidates[:5]
