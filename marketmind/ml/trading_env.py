"""
MarketMind AI - RL Trading Environment
Gym-like environment for RL agent training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class TradingEnvironment:
    """
    Simulated trading environment for RL agents
    Implements gym-like interface for agent training
    """

    # Action definitions
    ACTIONS = {
        0: 'BUY',      # Enter long position
        1: 'HOLD',     # Maintain current position
        2: 'SELL',     # Close position
        3: 'HEDGE',    # Reduce risk (partial sell)
        4: 'REBALANCE' # Adjust allocation
    }

    def __init__(self, historical_data: pd.DataFrame,
                 initial_capital: float = 100000,
                 position_size: float = 0.1):
        """
        Initialize trading environment

        Args:
            historical_data: DataFrame with 'date', 'open', 'high', 'low', 'close', 'volume'
            initial_capital: Starting capital in rupees
            position_size: Fraction of capital per BUY action
        """
        self.historical_data = historical_data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.position_size = position_size

        self.current_step = 0
        self.max_steps = len(historical_data) - 1

        # Account state
        self.balance = initial_capital
        self.position = 0  # Shares held
        self.entry_price = 0
        self.portfolio_value = initial_capital

        # History
        self.portfolio_history = [initial_capital]
        self.action_history = []
        self.trade_history = []

        # State size (feature vector)
        self.state_size = 50

    def reset(self) -> np.ndarray:
        """Reset environment to beginning"""
        self.current_step = 0
        self.balance = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.portfolio_value = self.initial_capital
        self.portfolio_history = [self.initial_capital]
        self.action_history = []
        self.trade_history = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current market state as feature vector"""
        if self.current_step >= len(self.historical_data):
            return np.zeros(self.state_size)

        lookback = 30
        start_idx = max(0, self.current_step - lookback)
        end_idx = self.current_step + 1

        data_slice = self.historical_data.iloc[start_idx:end_idx]
        state = []

        # Price features (normalized)
        prices = data_slice['close'].values
        if len(prices) < 2:
            prices = np.array([100.0])

        # Returns (5, 10, 20 day)
        for days in [5, 10, 20]:
            if len(prices) > days:
                ret = (prices[-1] - prices[-days-1]) / prices[-days-1]
                state.append(np.clip(ret, -1, 1))
            else:
                state.append(0)

        # Current price relative to moving averages
        if len(prices) >= 20:
            ma20 = np.mean(prices[-20:])
            ma50 = np.mean(prices[-min(50, len(prices)):]) if len(prices) >= 50 else ma20
            state.append(np.clip((prices[-1] - ma20) / ma20, -1, 1))  # Price vs MA20
            state.append(np.clip((ma20 - ma50) / ma50, -1, 1))  # MA20 vs MA50 trend
        else:
            state.extend([0, 0])

        # Technical indicators
        rsi = self._calculate_rsi(prices)
        state.append(rsi / 100.0)

        macd = self._calculate_macd(prices)
        state.append(macd / 10.0)

        # Volatility
        if len(prices) >= 14:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns[-14:])
            state.append(np.clip(volatility * 10, -1, 1))
        else:
            state.append(0)

        # Portfolio state
        state.append(self.portfolio_value / self.initial_capital - 1)  # P&L ratio
        state.append(self.position / (self.initial_capital / prices[-1]) if prices[-1] > 0 else 0)  # Position ratio
        state.append(self.balance / self.initial_capital)  # Cash ratio

        # Unrealized P&L
        if self.position > 0 and self.entry_price > 0:
            pnl_pct = (prices[-1] - self.entry_price) / self.entry_price
            state.append(np.clip(pnl_pct, -1, 1))
        else:
            state.append(0)

        # Drawdown
        max_portfolio = max(self.portfolio_history)
        drawdown = (max_portfolio - self.portfolio_value) / max_portfolio if max_portfolio > 0 else 0
        state.append(np.clip(drawdown, 0, 1))

        # Recent returns momentum
        if len(prices) >= 6:
            recent_ret = (prices[-1] - prices[-6]) / prices[-6]
            state.append(np.clip(recent_ret, -1, 1))
        else:
            state.append(0)

        # Volume trend
        if 'volume' in data_slice.columns and len(data_slice) >= 20:
            volumes = data_slice['volume'].values
            avg_vol = np.mean(volumes[-20:])
            current_vol = volumes[-1]
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            state.append(np.clip((vol_ratio - 1) / 2, -1, 1))
        else:
            state.append(0)

        # Pad to fixed state_size
        while len(state) < self.state_size:
            state.append(0)

        return np.array(state[:self.state_size], dtype=np.float32)

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return np.clip(rsi, 0, 100)

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26) -> float:
        """Calculate MACD"""
        if len(prices) < slow:
            return 0.0

        # Simplified EMA calculation
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)

        return ema_fast - ema_slow

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate exponential moving average"""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 100.0

        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action and return new state, reward, done

        Actions:
        0 = BUY (go long)
        1 = HOLD (maintain position)
        2 = SELL (close position)
        3 = HEDGE (reduce position by 50%)
        4 = REBALANCE (adjust to 60/40 stock/cash)
        """
        current_data = self.historical_data.iloc[self.current_step]
        current_price = current_data['close']

        reward = 0

        if action == 0:  # BUY
            if self.position == 0:
                capital_to_use = self.balance * self.position_size
                shares_to_buy = capital_to_use / current_price

                if shares_to_buy > 0.1:  # Minimum trade
                    self.position += shares_to_buy
                    self.balance -= capital_to_use
                    self.entry_price = current_price
                    reward = 0.05  # Small reward for taking action

        elif action == 1:  # HOLD
            reward = 0.0

        elif action == 2:  # SELL
            if self.position > 0:
                proceeds = self.position * current_price
                pnl = proceeds - (self.position * self.entry_price)
                self.balance += proceeds
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': self.position,
                    'pnl': pnl
                })
                self.position = 0
                self.entry_price = 0
                reward = 0.1 if pnl > 0 else -0.1

        elif action == 3:  # HEDGE
            if self.position > 0:
                shares_to_sell = self.position * 0.5
                proceeds = shares_to_sell * current_price
                self.balance += proceeds
                self.position -= shares_to_sell
                reward = 0.02

        elif action == 4:  # REBALANCE
            target_stock_value = self.portfolio_value * 0.6
            current_stock_value = self.position * current_price

            if current_stock_value > target_stock_value:
                excess = current_stock_value - target_stock_value
                shares_to_sell = excess / current_price
                self.position -= shares_to_sell
                self.balance += excess
                reward = 0.02
            elif current_stock_value < target_stock_value:
                deficit = target_stock_value - current_stock_value
                if self.balance >= deficit:
                    shares_to_buy = deficit / current_price
                    self.position += shares_to_buy
                    self.balance -= deficit
                    reward = 0.02

        # Calculate new portfolio value
        self.portfolio_value = self.balance + (self.position * current_price)

        # Reward for profit
        prev_value = self.portfolio_history[-1]
        if self.portfolio_value > prev_value:
            reward += 0.1 * (self.portfolio_value - prev_value) / self.initial_capital
        else:
            reward -= 0.05 * (prev_value - self.portfolio_value) / self.initial_capital

        # Penalty for large drawdown
        max_portfolio = max(self.portfolio_history)
        drawdown = (max_portfolio - self.portfolio_value) / max_portfolio
        if drawdown > 0.1:
            reward -= 0.2

        # Penalty for idle capital
        cash_ratio = self.balance / self.portfolio_value
        if cash_ratio > 0.9 and action == 1:  # Mostly idle cash and holding
            reward -= 0.01

        self.portfolio_history.append(self.portfolio_value)
        self.action_history.append(action)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_state = self._get_state()

        return next_state, reward, done

    def get_action_name(self, action: int) -> str:
        """Get name of action"""
        return self.ACTIONS.get(action, 'UNKNOWN')

    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if len(self.portfolio_history) < 2:
            return {}

        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annualized_return = total_return * (252 / len(portfolio_values)) if len(portfolio_values) > 1 else 0

        # Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (cummax - portfolio_values) / cummax
        max_drawdown = np.max(drawdowns)

        # Win rate
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        win_rate = winning_trades / len(self.trade_history) if self.trade_history else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trade_history),
            'final_value': portfolio_values[-1]
        }
