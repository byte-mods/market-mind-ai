"""
MarketMind AI - News Clusterer
Groups similar headlines to remove duplicates and surface unique stories.
Uses TF-IDF cosine similarity for fast clustering without heavy dependencies.
"""

import re
import math
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'shall', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'and', 'or', 'but', 'not', 'this', 'that', 'it', 'its', 'their',
        'they', 'he', 'she', 'we', 'you', 'i', 'my', 'our', 'your',
        'after', 'before', 'about', 'above', 'across', 'up', 'down',
        'over', 'under', 'than', 'then', 'so', 'if', 'when', 'while',
    }
    return [w for w in text.split() if w and w not in stopwords and len(w) > 2]


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, float] = defaultdict(float)
    for t in tokens:
        tf[t] += 1
    n = len(tokens) or 1
    vec = {t: (c / n) * idf.get(t, 1.0) for t, c in tf.items()}
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}


def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    shared = set(v1) & set(v2)
    return sum(v1[k] * v2[k] for k in shared)


def cluster_news(news_items: List[Dict], threshold: float = 0.35) -> List[Dict]:
    """
    Cluster news items by headline similarity.
    Returns one representative item per cluster with cluster_size and all_titles.
    """
    if not news_items:
        return []

    titles = [item.get('title', '') for item in news_items]
    tokenized = [_tokenize(t) for t in titles]

    # Build IDF
    doc_freq: Dict[str, int] = defaultdict(int)
    n_docs = len(tokenized)
    for tokens in tokenized:
        for t in set(tokens):
            doc_freq[t] += 1
    idf = {t: math.log(n_docs / (df + 1)) + 1 for t, df in doc_freq.items()}

    vectors = [_tfidf_vector(tok, idf) for tok in tokenized]

    # Greedy clustering: assign each doc to first cluster it's similar to
    clusters: List[List[int]] = []
    assigned = [False] * len(news_items)

    for i in range(len(news_items)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(news_items)):
            if assigned[j]:
                continue
            sim = _cosine(vectors[i], vectors[j])
            if sim >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    # Build output: pick most-recent item in each cluster as representative
    result = []
    for cluster_indices in clusters:
        # Sort by sentiment magnitude (most impactful first) then publication time
        def score(idx: int) -> float:
            item = news_items[idx]
            sentiment_abs = abs(item.get('sentiment_score', 0))
            pub = item.get('published_at', '') or ''
            return sentiment_abs + (0.001 if pub else 0)

        best_idx = max(cluster_indices, key=score)
        rep = dict(news_items[best_idx])
        rep['cluster_size'] = len(cluster_indices)
        rep['all_titles'] = [titles[i] for i in cluster_indices]
        result.append(rep)

    # Sort by cluster_size desc, then sentiment_abs desc
    result.sort(key=lambda x: (x['cluster_size'], abs(x.get('sentiment_score', 0))), reverse=True)
    return result
