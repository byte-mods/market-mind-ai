"""
MarketMind AI - Claude-Powered News Fetcher
Uses Claude API with web search to fetch Indian market news,
stores results in MongoDB (localhost:27017, db: stock_analyst).
"""

import os
import json
import logging
import threading
import subprocess
import shutil
from datetime import datetime, timezone
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import pymongo
    _PYMONGO_OK = True
except ImportError:
    _PYMONGO_OK = False
    logger.warning("pymongo package not installed — MongoDB storage disabled")


# ── MongoDB connection ───────────────────────────────────────────────────────

_mongo_client: Optional["pymongo.MongoClient"] = None
_mongo_lock = threading.Lock()


def _get_db():
    """Return the stock_analyst database; creates connection once."""
    global _mongo_client
    if not _PYMONGO_OK:
        return None
    with _mongo_lock:
        if _mongo_client is None:
            try:
                _mongo_client = pymongo.MongoClient(
                    "localhost", 27017,
                    serverSelectionTimeoutMS=3000,
                    connectTimeoutMS=3000,
                )
                # Ping to confirm
                _mongo_client.admin.command("ping")
                logger.info("MongoDB connected — stock_analyst db ready")
            except Exception as exc:
                logger.error(f"MongoDB connection failed: {exc}")
                _mongo_client = None
                return None
    return _mongo_client["stock_analyst"] if _mongo_client else None


# ── Claude web-search news fetch ─────────────────────────────────────────────

_NEWS_PROMPT = """
Search the web for the latest Indian stock market and financial news published today or in the last 24 hours.

Focus on:
- NIFTY 500 and SENSEX movements
- Top gaining/losing NSE stocks
- RBI/SEBI announcements
- FII/DII flows
- Corporate earnings, mergers, dividends
- Sector-specific news (Banking, IT, Pharma, Auto, FMCG)
- Global market impact on Indian markets

Return a JSON array (no markdown, pure JSON) with up to 20 articles. Each article must have:
{
  "title": "...",
  "summary": "...",
  "source": "...",
  "url": "...",
  "published_at": "ISO-8601 datetime or date string",
  "sectors": ["IT", "Banking", ...],
  "stocks_mentioned": ["RELIANCE", "TCS", ...]
}
"""


def fetch_news_via_claude(api_key: str = None) -> List[Dict]:
    """
    Use the `claude` CLI (Claude Code) to retrieve Indian market news via web search.
    Falls back to Anthropic SDK if CLI is not available.
    Returns a list of news dicts.
    """
    claude_bin = shutil.which("claude")
    if claude_bin:
        return _fetch_via_claude_cli(claude_bin)

    # Fallback: Anthropic SDK
    try:
        import anthropic
    except ImportError:
        logger.warning("Neither claude CLI nor anthropic SDK available — skipping news fetch")
        return []

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping Claude news fetch")
        return []

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": _NEWS_PROMPT}],
        )
        text_parts = [block.text for block in response.content if hasattr(block, "text")]
        raw = "".join(text_parts).strip()
        raw = _strip_fences(raw)
        articles = json.loads(raw) if raw.startswith("[") else []
        logger.info(f"Claude SDK returned {len(articles)} news articles")
        return articles
    except Exception as exc:
        logger.error(f"Claude SDK news fetch error: {exc}")
        return []


def _fetch_via_claude_cli(claude_bin: str) -> List[Dict]:
    """Run `claude -p <prompt>` as a subprocess and parse JSON output."""
    try:
        result = subprocess.run(
            [claude_bin, "-p", _NEWS_PROMPT, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"Claude CLI exited {result.returncode}: {result.stderr[:200]}")
            return []
        raw = result.stdout.strip()

        # If CLI returned a JSON envelope (newer versions), extract the text result
        if raw.startswith("{"):
            try:
                envelope = json.loads(raw)
                raw = envelope.get("result") or envelope.get("content") or raw
                if isinstance(raw, list):
                    # content array — join text blocks
                    raw = " ".join(
                        b.get("text", "") for b in raw if isinstance(b, dict)
                    )
            except json.JSONDecodeError:
                pass

        raw = _strip_fences(raw)
        # Find the JSON array (Claude may prepend prose)
        start = raw.find("[")
        if start == -1:
            logger.error("Claude CLI output contains no JSON array")
            logger.debug(f"Raw output preview: {raw[:300]}")
            return []
        raw = raw[start:]
        articles = json.loads(raw)
        logger.info(f"Claude CLI returned {len(articles)} news articles")
        return articles
    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timed out after 120s")
        return []
    except json.JSONDecodeError as exc:
        logger.error(f"Claude CLI JSON parse error: {exc}")
        return []
    except Exception as exc:
        logger.error(f"Claude CLI error: {exc}")
        return []


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences from a string."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


# ── MongoDB storage ───────────────────────────────────────────────────────────

def store_news_in_mongo(articles: List[Dict]) -> int:
    """
    Upsert news articles into MongoDB.
    Collection: stock_analyst.news
    Deduplicates by title.
    Returns number of articles inserted/updated.
    """
    db = _get_db()
    if db is None:
        logger.warning("MongoDB unavailable — news not stored")
        return 0

    collection = db["news"]

    # Create a unique index on title to avoid duplicates
    try:
        collection.create_index("title", unique=True, background=True)
    except Exception:
        pass

    stored = 0
    for article in articles:
        try:
            article.setdefault("fetched_at", datetime.now(timezone.utc).isoformat())
            collection.update_one(
                {"title": article["title"]},
                {"$set": article},
                upsert=True,
            )
            stored += 1
        except pymongo.errors.DuplicateKeyError:
            pass
        except Exception as exc:
            logger.error(f"Mongo insert error: {exc}")

    logger.info(f"Stored {stored}/{len(articles)} articles in MongoDB stock_analyst.news")
    return stored


def get_recent_news_from_mongo(limit: int = 50) -> List[Dict]:
    """
    Retrieve the most recent news articles from MongoDB.
    """
    db = _get_db()
    if db is None:
        return []

    try:
        docs = list(
            db["news"]
            .find({}, {"_id": 0})
            .sort("fetched_at", pymongo.DESCENDING)
            .limit(limit)
        )
        return docs
    except Exception as exc:
        logger.error(f"Mongo read error: {exc}")
        return []


# ── High-level entry point ────────────────────────────────────────────────────

def run_claude_news_pipeline(api_key: str = None) -> List[Dict]:
    """
    Full pipeline: fetch via Claude → store in MongoDB → return articles.
    Call this on app startup or on demand.
    """
    logger.info("Starting Claude news pipeline…")
    articles = fetch_news_via_claude(api_key)

    if articles:
        store_news_in_mongo(articles)
    else:
        # Fallback: return what's in Mongo
        logger.info("No new articles from Claude; loading from MongoDB cache")
        articles = get_recent_news_from_mongo()

    return articles
