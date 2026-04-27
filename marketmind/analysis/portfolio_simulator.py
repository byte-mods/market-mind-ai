"""
MarketMind AI - Portfolio Simulator
Monte Carlo portfolio simulations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random


class PortfolioSimulator:
    """
    Monte Carlo simulation engine for portfolio analysis
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital

    def run_simulations(self,
                        expected_return: float = 0.12,
                        volatility: float = 0.20,
                        num_simulations: int = 1000,
                        time_horizon_days: int = 252,
                        initial_capital: float = None) -> Dict:
        """
        Run Monte Carlo simulations for portfolio
        """
        if initial_capital is None:
            initial_capital = self.initial_capital

        daily_return = expected_return / time_horizon_days
        daily_volatility = volatility / np.sqrt(time_horizon_days)

        results = {
            'final_values': [],
            'max_drawdowns': [],
            'sharpe_ratios': [],
            'returns': [],
            'daily_returns': []
        }

        for sim in range(num_simulations):
            portfolio_values = [initial_capital]
            daily_rets = []

            for day in range(time_horizon_days):
                # Geometric Brownian Motion
                random_return = np.random.normal(daily_return, daily_volatility)
                new_value = portfolio_values[-1] * (1 + random_return)
                portfolio_values.append(new_value)
                daily_rets.append(random_return)

            portfolio_values = np.array(portfolio_values)

            # Calculate final return
            final_return = (portfolio_values[-1] - initial_capital) / initial_capital
            results['returns'].append(final_return)
            results['final_values'].append(portfolio_values[-1])

            # Max drawdown
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (running_max - portfolio_values) / running_max
            max_dd = np.max(drawdowns)
            results['max_drawdowns'].append(max_dd)

            # Sharpe ratio
            mean_daily_ret = np.mean(daily_rets)
            std_daily_ret = np.std(daily_rets)
            if std_daily_ret > 0:
                sharpe = (mean_daily_ret / std_daily_ret) * np.sqrt(252)
            else:
                sharpe = 0
            results['sharpe_ratios'].append(sharpe)

            results['daily_returns'].append(daily_rets)

        return results

    def calculate_percentiles(self, results: Dict, percentiles: List[int] = None) -> Dict:
        """Calculate percentile values from simulation results"""
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        final_values = np.array(results['final_values'])
        returns = np.array(results['returns'])
        max_drawdowns = np.array(results['max_drawdowns'])
        sharpe_ratios = np.array(results['sharpe_ratios'])

        percentiles_dict = {}

        for p in percentiles:
            percentiles_dict[f'value_p{p}'] = float(np.percentile(final_values, p))
            percentiles_dict[f'return_p{p}'] = float(np.percentile(returns, p))
            percentiles_dict[f'drawdown_p{p}'] = float(np.percentile(max_drawdowns, p))
            percentiles_dict[f'sharpe_p{p}'] = float(np.percentile(sharpe_ratios, p))

        return percentiles_dict

    def generate_scenario_analysis(self, results: Dict) -> Dict:
        """Generate bull/base/bear scenario analysis"""
        final_values = np.array(results['final_values'])
        returns = np.array(results['returns'])

        return {
            'bull_case': {
                'label': 'Bull Case',
                'final_value': float(np.percentile(final_values, 80)),
                'return': float(np.percentile(returns, 80)),
                'probability': '20%',
                'description': 'Favorable market conditions, strong momentum'
            },
            'base_case': {
                'label': 'Base Case',
                'final_value': float(np.percentile(final_values, 50)),
                'return': float(np.percentile(returns, 50)),
                'probability': '50%',
                'description': 'Normal market conditions'
            },
            'bear_case': {
                'label': 'Bear Case',
                'final_value': float(np.percentile(final_values, 20)),
                'return': float(np.percentile(returns, 20)),
                'probability': '30%',
                'description': 'Adverse market conditions, economic slowdown'
            }
        }

    def calculate_var_cvar(self, results: Dict, confidence: float = 0.95) -> Dict:
        """Calculate Value at Risk and Conditional VaR"""
        returns = np.array(results['returns'])
        final_values = np.array(results['final_values'])

        # VaR at given confidence
        var_percentile = (1 - confidence) * 100
        var_return = np.percentile(returns, var_percentile)
        var_value = final_values[np.argmin(np.abs(returns - var_return))]

        # CVaR (Expected Shortfall)
        var_indices = returns <= var_return
        cvar_return = np.mean(returns[var_indices]) if np.any(var_indices) else var_return
        cvar_value = np.mean(final_values[var_indices]) if np.any(var_indices) else var_value

        return {
            'var_95_return': float(var_return),
            'var_95_value': float(var_value),
            'cvar_95_return': float(cvar_return),
            'cvar_95_value': float(cvar_value),
            'var_95_explanation': f"5% chance portfolio drops more than {abs(var_return)*100:.1f}%"
        }

    def calculate_summary_stats(self, results: Dict) -> Dict:
        """Calculate summary statistics"""
        returns = np.array(results['returns'])
        final_values = np.array(results['final_values'])
        max_drawdowns = np.array(results['max_drawdowns'])
        sharpe_ratios = np.array(results['sharpe_ratios'])

        return {
            'expected_return': float(np.mean(returns)),
            'return_std': float(np.std(returns)),
            'expected_value': float(np.mean(final_values)),
            'avg_max_drawdown': float(np.mean(max_drawdowns)),
            'avg_sharpe_ratio': float(np.mean(sharpe_ratios)),
            'prob_profit': float(np.mean(returns > 0)),
            'prob_10pct_loss': float(np.mean(returns < -0.10)),
            'prob_20pct_loss': float(np.mean(returns < -0.20)),
            'best_return': float(np.max(returns)),
            'worst_return': float(np.min(returns)),
            'median_return': float(np.median(returns))
        }

    def simulate_sector_allocation(self,
                                    allocations: Dict[str, float],
                                    expected_returns: Dict[str, float],
                                    volatilities: Dict[str, float],
                                    correlations: Dict[Tuple[str, str], float],
                                    num_simulations: int = 1000,
                                    time_horizon_days: int = 252) -> Dict:
        """
        Simulate portfolio with specific sector allocations
        """
        sectors = list(allocations.keys())
        num_sectors = len(sectors)

        # Build correlation matrix
        corr_matrix = np.eye(num_sectors)
        for i, s1 in enumerate(sectors):
            for j, s2 in enumerate(sectors):
                if (s1, s2) in correlations:
                    corr_matrix[i, j] = correlations[(s1, s2)]
                elif (s2, s1) in correlations:
                    corr_matrix[i, j] = correlations[(s2, s1)]

        # Cholesky decomposition for correlated returns
        try:
            L = np.linalg.cholesky(corr_matrix)
        except:
            L = np.eye(num_sectors)

        results = {
            'final_values': [],
            'sector_final_values': {s: [] for s in sectors}
        }

        for _ in range(num_simulations):
            # Generate uncorrelated random returns
            uncorrelated_returns = np.random.normal(0, 1, (time_horizon_days, num_sectors))

            # Apply correlation
            correlated_returns = uncorrelated_returns @ L.T

            # Calculate sector values over time
            sector_values = {s: allocations[s] * self.initial_capital for s in sectors}

            for day in range(time_horizon_days):
                for i, sector in enumerate(sectors):
                    daily_ret = (expected_returns[sector] / time_horizon_days +
                                correlated_returns[day, i] * volatilities[sector] / np.sqrt(time_horizon_days))
                    sector_values[sector] *= (1 + daily_ret)

            total_final = sum(sector_values.values())
            results['final_values'].append(total_final)

            for sector in sectors:
                results['sector_final_values'][sector].append(sector_values[sector])

        return {
            'expected_value': float(np.mean(results['final_values'])),
            'expected_return': float(np.mean([(v - self.initial_capital) / self.initial_capital
                                              for v in results['final_values']])),
            'std_value': float(np.std(results['final_values'])),
            'sector_expected_values': {s: float(np.mean(results['sector_final_values'][s]))
                                      for s in sectors}
        }


# Global instance
_simulator = None


def get_portfolio_simulator(initial_capital: float = 100000) -> PortfolioSimulator:
    """Get or create global portfolio simulator"""
    global _simulator
    if _simulator is None:
        _simulator = PortfolioSimulator(initial_capital)
    return _simulator
