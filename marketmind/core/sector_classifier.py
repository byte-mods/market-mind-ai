"""
MarketMind AI - Sector Classifier Module
Maps news to sectors and computes sector correlations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class SectorClassifier:
    """
    Classifies news into sectors and computes sector correlations
    """

    # Indian market sectors with their components
    SECTORS = {
        'IT': {
            'name': 'Information Technology',
            'indices': ['NIFTYIT'],
            'stocks': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'LTTS', 'MINDTREE', 'COFORGE', 'LUMAXIND'],
            'keywords': ['IT', 'software', 'technology', 'digital', 'cloud', 'AI', 'data center', 'SaaS', 'IT services']
        },
        'Banking': {
            'name': 'Banking & Financial Services',
            'indices': ['NIFTYBANK'],
            'stocks': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB'],
            'keywords': ['bank', 'banking', 'credit', 'loan', 'deposit', 'NBFC', 'financial services', 'mortgage']
        },
        'Auto': {
            'name': 'Automobiles',
            'indices': ['NIFTYAUTO'],
            'stocks': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT', 'TVSMOTOR', 'ASHOKLEY'],
            'keywords': ['auto', 'automobile', 'car', 'SUV', 'EV', 'electric vehicle', 'two wheeler', 'commercial vehicle']
        },
        'Pharma': {
            'name': 'Pharmaceuticals',
            'indices': ['NIFTYPHARMA'],
            'stocks': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'AUROPHARMA', 'ZYDUSLIFE', 'CADILAHC', 'GLENMARK'],
            'keywords': ['pharma', 'pharmaceutical', 'drug', 'medicine', 'API', 'generic', 'biotech', 'healthcare']
        },
        'FMCG': {
            'name': 'Fast Moving Consumer Goods',
            'indices': ['NIFTYFMCG'],
            'stocks': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'DABUR', 'COLPAL', 'MARICO', 'BRITANNIA', 'GODREJCP'],
            'keywords': ['FMCG', 'consumer goods', 'personal care', 'household', 'food', 'beverage', 'daily use']
        },
        'Metal': {
            'name': 'Metals & Mining',
            'indices': ['NIFTYMETAL'],
            'stocks': ['TATASTEEL', 'HINDALCO', 'VEDL', 'COALINDIA', 'NMDC', 'SAIL', 'JINDALSTEL', 'WELCORP'],
            'keywords': ['metal', 'steel', 'aluminum', 'copper', 'mining', 'coal', 'iron ore', 'zinc']
        },
        'Energy': {
            'name': 'Energy',
            'indices': ['NIFTYENERGY'],
            'stocks': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HINDPETRO', 'GAIL', 'OIL', 'PETRONET'],
            'keywords': ['oil', 'gas', 'petroleum', 'refinery', 'exploration', 'energy', 'power', 'renewable', 'solar']
        },
        'Realty': {
            'name': 'Real Estate',
            'indices': ['NIFTYREALTY'],
            'stocks': ['DLF', 'GODREJPROP', 'BRIGADE', 'PRESTIGE', 'OBEROIRLTY', 'SUNTECK', 'MAHLIFE'],
            'keywords': ['real estate', 'property', 'housing', 'REIT', 'commercial', 'residential', 'construction']
        },
        'Finance': {
            'name': 'Financial Services',
            'indices': ['NIFTYFIN'],
            'stocks': ['BAJFINANCE', 'BAJAJFINSV', 'HDFC', 'SBILIFE', 'ICICIPRULI', 'MAXHEALTH', 'CHOLAFIN'],
            'keywords': ['finance', 'insurance', 'mutual fund', 'SIP', 'LIC', 'NBFC', 'investment', 'wealth']
        },
        'Global': {
            'name': 'Global Markets',
            'indices': [],
            'stocks': [],
            'keywords': ['Wall Street', 'US markets', 'Dow Jones', 'NASDAQ', 'S&P 500', 'Fed', 'Federal Reserve', 'China economy']
        },
        'Geopolitical': {
            'name': 'Geopolitical',
            'indices': [],
            'stocks': [],
            'keywords': ['war', 'Russia', 'Ukraine', 'Middle East', 'US China', 'sanctions', 'OPEC', 'conflict']
        }
    }

    # Sector relationships (for correlation context)
    SECTOR_LINKS = {
        'IT': ['Finance'],  # IT benefits from finance sector investment
        'Banking': ['Finance', 'Realty'],  # Banks benefit from realty lending
        'Auto': ['Metal', 'Energy'],  # Auto needs steel and energy
        'Pharma': ['Global'],  # Pharma affected by global regulations
        'FMCG': ['Energy'],  # FMCG affected by commodity prices
        'Metal': ['Energy', 'Global'],  # Metals affected by global demand
        'Energy': ['Global'],  # Energy very sensitive to global markets
        'Realty': ['Banking', 'Finance'],  # Realty linked to interest rates
        'Finance': ['Global'],  # Finance affected by global capital flows
    }

    def __init__(self):
        self.sectors = list(self.SECTORS.keys())

    def classify_news(self, text: str) -> List[Tuple[str, float]]:
        """
        Classify news into sectors with confidence scores
        Returns list of (sector, confidence) tuples
        """
        text_lower = text.lower()
        sector_scores = defaultdict(float)

        for sector, info in self.SECTORS.items():
            # Count keyword matches
            matches = 0
            for keyword in info['keywords']:
                if keyword.lower() in text_lower:
                    matches += 1

            # Calculate confidence based on matches
            if matches > 0:
                confidence = min(1.0, matches / 3)  # Normalize
                sector_scores[sector] = confidence

        # Sort by confidence
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)

        # If no matches, default to Global
        if not sorted_sectors:
            return [('Global', 0.5)]

        return sorted_sectors

    def get_sector_news(self, news_list: List[Dict], sector: str) -> List[Dict]:
        """Filter news items for a specific sector"""
        filtered = []
        for item in news_list:
            item_sectors = item.get('sectors', [])
            if sector in item_sectors:
                filtered.append(item)
        return filtered

    def get_sector_sentiment(self, news_list: List[Dict], sector: str) -> Dict:
        """Get aggregated sentiment for a sector"""
        sector_news = self.get_sector_news(news_list, sector)

        if not sector_news:
            return {'score': 0, 'label': 'Neutral', 'count': 0}

        scores = []
        for item in sector_news:
            score = item.get('sentiment_score', 0)
            if score is not None:
                scores.append(score)

        if not scores:
            return {'score': 0, 'label': 'Neutral', 'count': 0}

        avg_score = sum(scores) / len(scores)

        if avg_score > 0.2:
            label = 'Positive'
        elif avg_score > 0.05:
            label = 'Slightly Positive'
        elif avg_score > -0.05:
            label = 'Neutral'
        elif avg_score > -0.2:
            label = 'Slightly Negative'
        else:
            label = 'Negative'

        return {
            'score': round(avg_score, 3),
            'label': label,
            'count': len(scores)
        }

    def get_all_sector_sentiments(self, news_list: List[Dict]) -> Dict[str, Dict]:
        """Get sentiment for all sectors"""
        sentiments = {}
        for sector in self.sectors:
            sentiments[sector] = self.get_sector_sentiment(news_list, sector)
        return sentiments

    def calculate_correlation_matrix(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Sector-level return correlation matrix as a labeled DataFrame
        (rows + columns = sector names; values clipped to [-1, 1]).
        """
        if not price_data:
            return pd.DataFrame()

        # Align each sector's close series on date index
        series_by_sector: Dict[str, pd.Series] = {}
        for sector, df in price_data.items():
            if df is None or df.empty or 'close' not in df.columns:
                continue
            s = df.set_index('date')['close']
            if not s.empty:
                series_by_sector[sector] = s
        if len(series_by_sector) < 2:
            return pd.DataFrame()

        combined = pd.concat(series_by_sector, axis=1).ffill().bfill()
        returns = combined.pct_change().dropna()
        if returns.empty or len(returns) < 2:
            return pd.DataFrame()

        corr = returns.corr().fillna(0.0).clip(-1.0, 1.0)
        return corr

    def get_correlation_color(self, correlation: float) -> str:
        """Get color for correlation value"""
        if correlation > 0.5:
            return '#3FB950'  # Strong positive - green
        elif correlation > 0.2:
            return '#56D364'  # Weak positive - light green
        elif correlation > -0.2:
            return '#6E7681'  # Neutral - gray
        elif correlation > -0.5:
            return '#F85149'  # Weak negative - red
        else:
            return '#DA3633'  # Strong negative - dark red

    def get_leading_sectors(self, correlations: np.ndarray) -> List[str]:
        """
        Identify leading sectors based on correlation matrix
        Sectors that have high correlation with many others may be leaders
        """
        if correlations is None or len(correlations) == 0:
            return []

        # Sum of correlations (excluding self-correlation)
        total_corr = correlations.sum(axis=1) - 1

        # Normalize
        total_corr = total_corr / (len(total_corr) - 1)

        # Get indices sorted by correlation
        sorted_indices = np.argsort(total_corr)[::-1]

        leading = [self.sectors[i] for i in sorted_indices if total_corr[i] > 0]
        return leading

    def get_sector_stocks(self, sector: str) -> List[str]:
        """Get list of stocks in a sector"""
        if sector in self.SECTORS:
            return self.SECTORS[sector]['stocks']
        return []

    def get_sector_name(self, sector: str) -> str:
        """Get full name of sector"""
        if sector in self.SECTORS:
            return self.SECTORS[sector]['name']
        return sector


# Global instance
_sector_classifier = None


def get_sector_classifier() -> SectorClassifier:
    """Get or create global sector classifier instance"""
    global _sector_classifier
    if _sector_classifier is None:
        _sector_classifier = SectorClassifier()
    return _sector_classifier
