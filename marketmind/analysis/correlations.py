"""
MarketMind AI - Correlations Analysis Module
Computes sector and stock correlations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class CorrelationAnalyzer:
    """
    Analyzes correlations between sectors and stocks
    """

    def __init__(self):
        self.correlation_cache = {}

    def compute_returns(self, prices: pd.Series) -> pd.Series:
        """Compute returns from prices"""
        return prices.pct_change().dropna()

    def compute_correlation(self, returns1: pd.Series, returns2: pd.Series) -> float:
        """Compute Pearson correlation between two return series"""
        if len(returns1) < 10 or len(returns2) < 10:
            return 0.0

        # Align series
        combined = pd.concat([returns1, returns2], axis=1).dropna()
        if len(combined) < 10:
            return 0.0

        corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
        return 0.0 if pd.isna(corr) else corr

    def compute_sector_correlations(self, price_data: Dict[str, pd.DataFrame],
                                    sectors: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise correlations between sectors
        Returns dict of {(sector1, sector2): correlation}
        """
        correlations = {}

        # Get close prices for each sector
        sector_prices = {}
        for sector in sectors:
            if sector in price_data and not price_data[sector].empty:
                df = price_data[sector]
                if 'close' in df.columns:
                    sector_prices[sector] = df.set_index('date')['close']

        # Compute pairwise correlations
        sector_list = list(sector_prices.keys())
        for i, s1 in enumerate(sector_list):
            for s2 in sector_list[i+1:]:
                returns1 = self.compute_returns(sector_prices[s1])
                returns2 = self.compute_returns(sector_prices[s2])
                corr = self.compute_correlation(returns1, returns2)
                correlations[(s1, s2)] = corr
                correlations[(s2, s1)] = corr

        return correlations

    def find_correlated_pairs(self, correlations: Dict[Tuple[str, str], float],
                              threshold: float = 0.5,
                              top_n: int = 10) -> List[Tuple[str, str, float]]:
        """
        Find most correlated sector pairs
        Returns list of (sector1, sector2, correlation) tuples
        """
        pairs = []
        seen = set()

        for (s1, s2), corr in correlations.items():
            if s1 == s2:
                continue

            pair_key = tuple(sorted([s1, s2]))
            if pair_key in seen:
                continue

            if abs(corr) >= threshold:
                pairs.append((s1, s2, corr))
                seen.add(pair_key)

        # Sort by absolute correlation
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return pairs[:top_n]

    def find_uncorrelated_sectors(self, correlations: Dict[Tuple[str, str], float],
                                 threshold: float = 0.2) -> List[Tuple[str, str]]:
        """Find sector pairs with low correlation (good for diversification)"""
        uncorrelated = []
        seen = set()

        for (s1, s2), corr in correlations.items():
            if s1 == s2:
                continue

            pair_key = tuple(sorted([s1, s2]))
            if pair_key in seen:
                continue

            if abs(corr) < threshold:
                uncorrelated.append((s1, s2))
                seen.add(pair_key)

        return uncorrelated

    def compute_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Compute beta of a stock relative to market
        Beta = Cov(stock, market) / Var(market)
        """
        if len(stock_returns) < 30 or len(market_returns) < 30:
            return 1.0

        # Align series
        combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
        if len(combined) < 30:
            return 1.0

        stock_ret = combined.iloc[:, 0]
        market_ret = combined.iloc[:, 1]

        # Compute covariance and variance
        covariance = stock_ret.cov(market_ret)
        variance = market_ret.var()

        if variance == 0:
            return 1.0

        beta = covariance / variance
        return 1.0 if pd.isna(beta) else beta

    def compute_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix from returns DataFrame"""
        return returns_df.corr()

    def get_correlation_strength(self, correlation: float) -> str:
        """Get description of correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "Very Strong"
        elif abs_corr >= 0.6:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"

    def get_correlation_direction(self, correlation: float) -> str:
        """Get description of correlation direction"""
        if correlation > 0:
            return "Positive"
        elif correlation < 0:
            return "Negative"
        else:
            return "None"


# Global instance
_correlation_analyzer = None


def get_correlation_analyzer() -> CorrelationAnalyzer:
    """Get or create global correlation analyzer instance"""
    global _correlation_analyzer
    if _correlation_analyzer is None:
        _correlation_analyzer = CorrelationAnalyzer()
    return _correlation_analyzer
