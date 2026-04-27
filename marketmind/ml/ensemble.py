"""
MarketMind AI - Portfolio Simulator Module
Monte Carlo simulations for portfolio analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class PortfolioSimulator:
    """
    Portfolio simulation engine using Monte Carlo methods
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital

    def run_monte_carlo(self, returns: np.ndarray,
                        num_simulations: int = 1000,
                        time_horizon: int = 252) -> Dict:
        """
        Run Monte Carlo simulation
        """
        results = {
            'final_values': [],
            'max_drawdowns': [],
            'sharpe_ratios': [],
            'returns': []
        }

        for _ in range(num_simulations):
            # Generate random price paths
            portfolio_values = [self.initial_capital]

            for day in range(time_horizon):
                # Random return from distribution
                daily_return = np.random.choice(returns) if len(returns) > 0 else 0

                # Add some randomness to make it more realistic
                noise = np.random.normal(0, 0.01)
                daily_return = np.clip(daily_return + noise, -0.1, 0.1)

                new_value = portfolio_values[-1] * (1 + daily_return)
                portfolio_values.append(new_value)

            portfolio_values = np.array(portfolio_values)

            # Calculate metrics
            final_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
            results['returns'].append(final_return)
            results['final_values'].append(portfolio_values[-1])

            # Max drawdown
            cummax = np.maximum.accumulate(portfolio_values)
            drawdowns = (cummax - portfolio_values) / cummax
            max_dd = np.max(drawdowns)
            results['max_drawdowns'].append(max_dd)

            # Sharpe ratio (simplified)
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            if np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe = 0
            results['sharpe_ratios'].append(sharpe)

        return results

    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate summary metrics from simulation results"""
        final_values = np.array(results['final_values'])
        returns = np.array(results['returns'])
        max_drawdowns = np.array(results['max_drawdowns'])
        sharpe_ratios = np.array(results['sharpe_ratios'])

        return {
            'expected_return': float(np.mean(returns)),
            'return_std': float(np.std(returns)),
            'best_case': float(np.percentile(final_values, 95)),
            'base_case': float(np.percentile(final_values, 50)),
            'worst_case': float(np.percentile(final_values, 5)),
            'avg_max_drawdown': float(np.mean(max_drawdowns)),
            'avg_sharpe_ratio': float(np.mean(sharpe_ratios)),
            'probability_of_profit': float(np.mean(returns > 0)),
            'probability_of_10pct_loss': float(np.mean(returns < -0.1))
        }

    def generate_scenarios(self, results: Dict) -> Dict:
        """Generate bull/base/bear scenarios"""
        final_values = np.array(results['final_values'])

        return {
            'bull_case': {
                'portfolio_value': float(np.percentile(final_values, 80)),
                'return': float(np.percentile(final_values, 80) / self.initial_capital - 1),
                'probability': '20%',
                'conditions': 'Strong earnings growth, sector tailwinds, global positive'
            },
            'base_case': {
                'portfolio_value': float(np.percentile(final_values, 50)),
                'return': float(np.percentile(final_values, 50) / self.initial_capital - 1),
                'probability': '50%',
                'conditions': 'Normal market conditions, moderate volatility'
            },
            'bear_case': {
                'portfolio_value': float(np.percentile(final_values, 20)),
                'return': float(np.percentile(final_values, 20) / self.initial_capital - 1),
                'probability': '30%',
                'conditions': 'Economic slowdown, sector headwinds, geopolitical risks'
            }
        }

    def simulate_portfolio_allocation(self,
                                      allocations: Dict[str, float],
                                      historical_returns: Dict[str, pd.DataFrame],
                                      num_simulations: int = 1000,
                                      time_horizon: int = 252) -> Dict:
        """
        Simulate portfolio with given allocations to sectors
        """
        results = {
            'final_values': [],
            'returns': [],
            'sector_values': defaultdict(list)
        }

        for _ in range(num_simulations):
            # Initialize sector values
            sector_values = {sector: self.initial_capital * alloc
                            for sector, alloc in allocations.items()}

            for day in range(time_horizon):
                for sector, alloc in allocations.items():
                    if sector in historical_returns and not historical_returns[sector].empty:
                        # Get random return for this sector
                        returns_data = historical_returns[sector]['close'].pct_change().dropna()
                        if len(returns_data) > 0:
                            ret = np.random.choice(returns_data.values)
                            noise = np.random.normal(0, 0.005)
                            ret = np.clip(ret + noise, -0.1, 0.1)
                            sector_values[sector] *= (1 + ret)

            total_value = sum(sector_values.values())
            results['final_values'].append(total_value)
            results['returns'].append((total_value - self.initial_capital) / self.initial_capital)

            for sector, value in sector_values.items():
                results['sector_values'][sector].append(value)

        return self.calculate_metrics({k: v for k, v in results.items() if k != 'sector_values'})


class RLEnhancedEnsemble:
    """
    Combines RL agent actions with other signals for ensemble decisions
    """

    def __init__(self, rl_engine):
        self.rl_engine = rl_engine

    def make_investment_decision(self, market_state: Dict) -> Dict:
        """
        Make complete investment decision combining all signals
        """
        decision = {
            'action': None,
            'position_size': 0,
            'confidence': 0,
            'reasoning': []
        }

        # Get RL decision
        if 'state_vector' in market_state:
            rl_decision = self.rl_engine.make_decision(market_state['state_vector'])
            decision['action'] = rl_decision['action']
            decision['position_size'] = rl_decision['position_size']
            decision['confidence'] = rl_decision['confidence']
            decision['reasoning'].extend(rl_decision['reasoning'])

        # Add technical analysis signal
        if 'technical_indicators' in market_state:
            indicators = market_state['technical_indicators']
            tech_signal = self._get_technical_signal(indicators)
            decision['reasoning'].append(tech_signal)

        # Add sentiment signal
        if 'sentiment_score' in market_state:
            sentiment = market_state['sentiment_score']
            sentiment_signal = f"Sentiment: {sentiment:.2f} ({'Positive' if sentiment > 0.1 else 'Negative' if sentiment < -0.1 else 'Neutral'})"
            decision['reasoning'].append(sentiment_signal)

        return decision

    def _get_technical_signal(self, indicators: Dict) -> str:
        """Get signal from technical indicators"""
        signals = []

        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi > 70:
                signals.append("RSI Overbought")
            elif rsi < 30:
                signals.append("RSI Oversold")

        if 'macd' in indicators:
            macd = indicators['macd']
            if macd > 0:
                signals.append("MACD Bullish")
            else:
                signals.append("MACD Bearish")

        return "Technical: " + ", ".join(signals) if signals else "Technical: Neutral"


# Global instance
_portfolio_simulator = None


def get_portfolio_simulator() -> PortfolioSimulator:
    """Get or create global portfolio simulator instance"""
    global _portfolio_simulator
    if _portfolio_simulator is None:
        _portfolio_simulator = PortfolioSimulator()
    return _portfolio_simulator
