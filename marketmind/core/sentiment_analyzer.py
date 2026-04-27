"""
MarketMind AI - Sentiment Analyzer Module
Analyzes sentiment of financial news using rule-based and ML approaches
"""

import re
import math
from typing import Dict, List, Tuple
from collections import Counter


class SentimentAnalyzer:
    """
    Financial news sentiment analyzer
    Uses lexicon-based approach with financial-specific word lists
    """

    # Positive words common in financial context
    POSITIVE_WORDS = {
        # Earnings & Growth
        'gain': 0.5, 'gains': 0.5, 'rising': 0.4, 'rise': 0.4, 'up': 0.3,
        'growth': 0.6, 'grew': 0.5, 'growing': 0.5, 'increase': 0.4,
        'increased': 0.5, 'increases': 0.4, 'increasing': 0.4,

        # Bullish signals
        'buy': 0.6, 'outperform': 0.5, 'overweight': 0.4, 'upgrade': 0.6,
        'upgraded': 0.6, 'higher': 0.3, 'high': 0.2, 'peak': 0.3,
        'rally': 0.5, 'rallied': 0.5, 'rallying': 0.5, 'surge': 0.6,
        'surged': 0.6, 'surging': 0.6, 'jump': 0.5, 'jumped': 0.5,
        'soar': 0.7, 'soared': 0.7, 'skyrocketed': 0.8,

        # Positive business news
        'profit': 0.5, 'profitable': 0.6, 'profits': 0.5, 'profitability': 0.5,
        'beat': 0.5, 'beats': 0.5, 'exceeded': 0.5, 'exceed': 0.5,
        'strong': 0.5, 'stronger': 0.6, 'strength': 0.4, 'solid': 0.5,
        'recovery': 0.5, 'recovering': 0.5, 'recovered': 0.5, 'recovery': 0.5,
        'boom': 0.7, 'booming': 0.6, 'boost': 0.4, 'boosted': 0.4,

        # Positive sentiment
        'positive': 0.5, 'optimistic': 0.5, 'hope': 0.3, 'hopeful': 0.4,
        'bright': 0.5, 'improve': 0.4, 'improved': 0.5, 'improving': 0.5,
        'better': 0.4, 'best': 0.5, 'breakthrough': 0.6, 'success': 0.5,
        'successful': 0.5, 'win': 0.4, 'won': 0.4, 'winner': 0.5,

        # Market positive
        'bullish': 0.6, 'bull': 0.5, 'uptrend': 0.5, 'upside': 0.5,
        'momentum': 0.3, 'breakout': 0.5, 'new high': 0.5, 'all-time high': 0.7,
    }

    # Negative words common in financial context
    NEGATIVE_WORDS = {
        # Losses & Decline
        'loss': -0.5, 'losses': -0.5, 'fall': -0.4, 'falls': -0.4, 'fell': -0.4,
        'falling': -0.4, 'decline': -0.4, 'declined': -0.5, 'declining': -0.5,
        'drop': -0.4, 'dropped': -0.4, 'drops': -0.4, 'dropping': -0.4,
        'plunge': -0.7, 'plunged': -0.7, 'plummet': -0.7, 'plummeted': -0.7,
        'tumble': -0.5, 'tumbled': -0.5, 'crash': -0.8, 'crashed': -0.8,
        'collapse': -0.8, 'collapsed': -0.8,

        # Bearish signals
        'sell': -0.6, 'underperform': -0.5, 'underweight': -0.4, 'downgrade': -0.6,
        'downgraded': -0.6, 'lower': -0.3, 'low': -0.2, 'bottom': -0.3,
        'weak': -0.4, 'weaker': -0.5, 'weakness': -0.4, 'weakness': -0.4,

        # Negative business news
        'loss': -0.5, 'losing': -0.5, 'miss': -0.5, 'missed': -0.5, 'misses': -0.5,
        'missed': -0.5, 'below': -0.3, 'worse': -0.4, 'worst': -0.5,
        'deficit': -0.4, 'deficits': -0.4, 'debt': -0.3, 'burden': -0.3,
        'cut': -0.3, 'cuts': -0.3, 'cutting': -0.3, 'reduce': -0.3, 'reduced': -0.3,
        'layoff': -0.5, 'layoffs': -0.5, 'job cuts': -0.5, 'close': -0.3,
        'closed': -0.3, 'closing': -0.3, 'shut': -0.4, 'shutdown': -0.5,
        'bankruptcy': -0.8, 'bankrupt': -0.8, 'insolvent': -0.7,

        # Negative sentiment
        'negative': -0.5, 'pessimistic': -0.5, 'fear': -0.4, 'fears': -0.4,
        'concern': -0.3, 'concerns': -0.3, 'concerned': -0.3, 'worry': -0.4,
        'worried': -0.4, 'uncertainty': -0.4, 'uncertain': -0.3, 'volatile': -0.3,
        'volatility': -0.3, 'risk': -0.3, 'risks': -0.3, 'risky': -0.3,
        'danger': -0.5, 'dangerous': -0.5, 'threat': -0.4, 'threats': -0.4,

        # Market negative
        'bearish': -0.6, 'bear': -0.5, 'downtrend': -0.5, 'downside': -0.4,
        'sell-off': -0.5, 'correction': -0.3, 'new low': -0.5, 'all-time low': -0.7,
    }

    # Intensifiers
    INTENSIFIERS = {
        'very': 1.5, 'extremely': 2.0, 'highly': 1.5, 'strongly': 1.5,
        'significantly': 1.5, 'substantially': 1.5, 'massively': 2.0,
        'sharply': 1.5, 'dramatically': 1.8, 'steeply': 1.5,
        'slightly': 0.5, 'marginally': 0.5, 'somewhat': 0.7,
        'moderately': 0.7,
    }

    # Negators
    NEGATORS = ['not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'hardly', 'barely', 'scarcely']

    # Financial-specific contexts
    CONTEXT_MODIFIERS = {
        'beat expectations': 0.3, 'missed expectations': -0.3,
        'raised guidance': 0.5, 'lowered guidance': -0.5,
        'cost cutting': 0.2, 'job cuts': -0.4,
        'market share gain': 0.4, 'market share loss': -0.4,
        'regulatory approval': 0.3, 'regulatory concern': -0.3,
        'analyst upgrade': 0.4, 'analyst downgrade': -0.4,
    }

    def __init__(self):
        self.initialized = True

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        Returns dict with score, label, and confidence
        """
        if not text or len(text.strip()) == 0:
            return {
                'score': 0,
                'label': 'Neutral',
                'confidence': 0,
                'positive_words': [],
                'negative_words': []
            }

        # Clean text
        text_clean = text.lower()
        text_clean = re.sub(r'[^\w\s]', ' ', text_clean)

        words = text_clean.split()

        # Find matched words
        positive_matches = []
        negative_matches = []

        for word in words:
            if word in self.POSITIVE_WORDS:
                positive_matches.append((word, self.POSITIVE_WORDS[word]))
            if word in self.NEGATIVE_WORDS:
                negative_matches.append((word, self.NEGATIVE_WORDS[word]))

        # Calculate base score
        score = 0
        word_count = len(words)

        # Apply intensifiers and negators
        for i, word in enumerate(words):
            # Check for negator before word
            negated = False
            for neg in self.NEGATORS:
                if neg in words[max(0, i-2):i+1]:
                    negated = True
                    break

            # Apply intensifier before word
            intensity = 1.0
            if i > 0 and words[i-1] in self.INTENSIFIERS:
                intensity = self.INTENSIFIERS[words[i-1]]

            # Apply to positive words
            if word in self.POSITIVE_WORDS:
                val = self.POSITIVE_WORDS[word] * intensity
                if negated:
                    val = -val * 0.5
                score += val

            # Apply to negative words
            if word in self.NEGATIVE_WORDS:
                val = self.NEGATIVE_WORDS[word] * intensity
                if negated:
                    val = -val * 0.5
                score += val

        # Check for context modifiers (phrases)
        for context, modifier in self.CONTEXT_MODIFIERS.items():
            if context in text_clean:
                score += modifier

        # Normalize by word count (with damping)
        normalized_score = score / (math.sqrt(word_count) + 1)

        # Clamp to [-1, 1] range
        normalized_score = max(-1, min(1, normalized_score))

        # Calculate confidence based on word matches
        total_matches = len(positive_matches) + len(negative_matches)
        confidence = min(1.0, total_matches / max(1, word_count / 10))

        # Determine label
        if normalized_score > 0.2:
            label = 'Positive'
        elif normalized_score > 0.05:
            label = 'Slightly Positive'
        elif normalized_score < -0.2:
            label = 'Negative'
        elif normalized_score < -0.05:
            label = 'Slightly Negative'
        else:
            label = 'Neutral'

        return {
            'score': round(normalized_score, 3),
            'label': label,
            'confidence': round(confidence, 3),
            'positive_words': positive_matches[:5],
            'negative_words': negative_matches[:5],
            'raw_score': round(score, 3)
        }

    def classify_headline(self, headline: str) -> str:
        """Quick classification of headline"""
        result = self.analyze(headline)
        return result['label']

    def batch_analyze(self, items: List[Dict]) -> List[Dict]:
        """Analyze a batch of news items"""
        results = []
        for item in items:
            text = item.get('title', '') + ' ' + item.get('content', '')
            sentiment = self.analyze(text)
            results.append({
                **item,
                'sentiment_score': sentiment['score'],
                'sentiment_label': sentiment['label'],
                'sentiment_confidence': sentiment['confidence']
            })
        return results

    def get_sentiment_color(self, score: float) -> str:
        """Get color code for sentiment score"""
        if score > 0.3:
            return '#3FB950'  # Green
        elif score > 0.1:
            return '#56D364'  # Light green
        elif score > -0.1:
            return '#8B949E'  # Gray
        elif score > -0.3:
            return '#F85149'  # Red
        else:
            return '#DA3633'  # Dark red

    def aggregate_sentiment(self, news_items: List[Dict]) -> Dict:
        """Aggregate sentiment across multiple news items"""
        if not news_items:
            return {'score': 0, 'label': 'Neutral', 'count': 0}

        scores = []
        for item in news_items:
            score = item.get('sentiment_score', 0)
            if score is not None:
                scores.append(score)

        if not scores:
            return {'score': 0, 'label': 'Neutral', 'count': 0}

        avg_score = sum(scores) / len(scores)

        # Weight recent news more heavily
        weights = [1 + (i * 0.1) for i in range(len(scores))]
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        if weighted_score > 0.2:
            label = 'Positive'
        elif weighted_score > 0.05:
            label = 'Slightly Positive'
        elif weighted_score > -0.05:
            label = 'Neutral'
        elif weighted_score > -0.2:
            label = 'Slightly Negative'
        else:
            label = 'Negative'

        return {
            'score': round(weighted_score, 3),
            'label': label,
            'count': len(scores),
            'avg_score': round(avg_score, 3)
        }


# Global instance
_sentiment_analyzer = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create global sentiment analyzer instance"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer
