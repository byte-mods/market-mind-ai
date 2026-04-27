"""
MarketMind AI - Filings & Concall Ingester (W2.1)

Pulls NSE corporate-announcements for a symbol — the same feed that carries
quarterly results, board outcomes, concall-transcript intimations, insider
trading disclosures, and Reg-30 material events — chunks the body text, and
upserts the chunks into a persistent Chroma collection (`filings`) so the
grounded research endpoint can cite them.

This is the seed ingester for Wave 2.1. Concall PDFs and BSE annual reports
plug in here later by adding fetchers that produce the same chunk shape:

    {
      "id":      "<deterministic>",
      "text":    "...embed-friendly chunk...",
      "metadata": {
        "symbol":    "RELIANCE",
        "date":      "2026-04-26",
        "category":  "concall|results|insider|board|other",
        "source":    "nse_announcements",
        "url":       "https://nsearchives.nseindia.com/.../foo.pdf",
        "title":     "Transcript of the discussion ...",
        "industry":  "Refineries",
      }
    }
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",  # no brotli — requests can't decode
    "Origin": "https://www.nseindia.com",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
    "X-Requested-With": "XMLHttpRequest",
}


# Map NSE description keywords → coarse category for filtering
_CATEGORY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("concall",  re.compile(r"transcript|con\.?\s*call|analyst|investor meet", re.I)),
    ("results",  re.compile(r"financial result|audited|unaudited|earnings", re.I)),
    ("insider",  re.compile(r"insider|sast|reg\.?\s*7|reg\.?\s*29", re.I)),
    ("board",    re.compile(r"board meeting|board outcome|outcome of|directors", re.I)),
    ("dividend", re.compile(r"dividend|payout|record date|book closure", re.I)),
    ("ma",       re.compile(r"acquisition|merger|amalgamation|demerger|disinvest", re.I)),
    ("regulatory", re.compile(r"sebi|bse|reg\s*30|material|disclosure|investigation|sho?w cause", re.I)),
]


def _classify(text: str) -> str:
    if not text:
        return "other"
    for cat, pat in _CATEGORY_PATTERNS:
        if pat.search(text):
            return cat
    return "other"


def _normalise_date(s: str) -> str:
    """Produce an ISO yyyy-mm-dd from any of NSE's date formats."""
    if not s:
        return ""
    s = s.strip()
    fmts = ("%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%d")
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if len(s) >= 8 and s[:8].isdigit():
        try:
            return datetime.strptime(s[:8], "%d%m%Y").strftime("%Y-%m-%d")
        except ValueError:
            pass
    return s[:10]


def _chunk(text: str, target: int = 900, overlap: int = 120) -> List[str]:
    """Sentence-aware chunking. Keeps chunks ≤ ~target chars."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= target:
        return [text]
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buf = ""
    for s in sents:
        if not s:
            continue
        if len(buf) + len(s) + 1 > target:
            if buf:
                chunks.append(buf.strip())
            # keep tail of previous chunk for context overlap
            tail = buf[-overlap:] if buf else ""
            buf = (tail + " " + s).strip()
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf:
        chunks.append(buf.strip())
    # If sentences are absurdly long (some pasted PDFs lack punctuation),
    # fall back to a hard slice so chunks never exceed target+overlap.
    out: List[str] = []
    for c in chunks:
        if len(c) <= target + overlap:
            out.append(c)
            continue
        for i in range(0, len(c), target):
            out.append(c[i:i + target + overlap].strip())
    return [c for c in out if c]


class FilingsIngester:
    NSE_BASE = "https://www.nseindia.com/api"
    COLLECTION = "filings"

    def __init__(self):
        self._session: Optional[requests.Session] = None
        self._session_time: float = 0.0
        self._vs = None

    def _get_session(self) -> requests.Session:
        now = time.time()
        if self._session is None or (now - self._session_time) > 600:
            s = requests.Session()
            s.headers.update(HEADERS)
            try:
                s.get("https://www.nseindia.com", timeout=10)
                s.get(
                    "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
                    timeout=10,
                )
            except Exception as e:
                logger.debug(f"FilingsIngester warmup: {e}")
            self._session = s
            self._session_time = now
        return self._session

    def _vector_store(self):
        if self._vs is None:
            from marketmind.core.vectordb import get_vector_store
            self._vs = get_vector_store()
        return self._vs

    def fetch_announcements(self, symbol: str, days: int = 365) -> List[Dict]:
        """Pull raw rows from NSE corporate-announcements for the symbol."""
        symbol = symbol.upper().strip()
        try:
            s = self._get_session()
            today = datetime.now()
            params = {
                "index": "equities",
                "symbol": symbol,
                "from_date": (today - timedelta(days=days)).strftime("%d-%m-%Y"),
                "to_date":   today.strftime("%d-%m-%Y"),
            }
            r = s.get(f"{self.NSE_BASE}/corporate-announcements", params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.warning(f"NSE announcements fetch failed [{symbol}]: {e}")
            self._session = None
            return []

    def ingest_symbol(self, symbol: str, days: int = 365) -> Dict:
        """Fetch, chunk, embed. Idempotent thanks to deterministic IDs."""
        symbol = symbol.upper().strip()
        rows = self.fetch_announcements(symbol, days)
        if not rows:
            return {"symbol": symbol, "rows": 0, "chunks": 0, "skipped": True,
                    "reason": "NSE returned no rows"}

        documents: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []
        seen_ids: set = set()

        for row in rows:
            title = (row.get("desc") or "").strip()
            body = (row.get("attchmntText") or "").strip()
            if not title and not body:
                continue
            full = f"{title}\n\n{body}".strip()
            category = _classify(title + " " + body)
            iso_date = _normalise_date(row.get("sort_date") or row.get("an_dt") or "")
            url = row.get("attchmntFile") or ""
            seq = str(row.get("seq_id") or "")
            industry = row.get("smIndustry") or ""

            chunks = _chunk(full, target=900, overlap=120)
            for i, ck in enumerate(chunks):
                # Deterministic ID: stable across re-ingests
                cid = f"nse_{symbol}_{seq}_{i}" if seq else f"nse_{symbol}_{iso_date}_{i}_{abs(hash(ck)) % 10**8}"
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                documents.append(ck)
                metadatas.append({
                    "symbol":   symbol,
                    "date":     iso_date,
                    "category": category,
                    "source":   "nse_announcements",
                    "url":      url,
                    "title":    title[:240],
                    "industry": industry,
                    "seq":      seq,
                    "chunk_idx": i,
                })
                ids.append(cid)

        if not documents:
            return {"symbol": symbol, "rows": len(rows), "chunks": 0, "skipped": True,
                    "reason": "all rows lacked attchmntText"}
        vs = self._vector_store()
        added = vs.add(self.COLLECTION, documents=documents, metadatas=metadatas, ids=ids)
        return {
            "symbol": symbol,
            "rows": len(rows),
            "chunks": added,
            "categories": _category_counts(metadatas),
            "earliest": min((m["date"] for m in metadatas if m.get("date")), default=""),
            "latest":   max((m["date"] for m in metadatas if m.get("date")), default=""),
        }

    def stats(self) -> Dict:
        vs = self._vector_store()
        return {
            "collection": self.COLLECTION,
            "doc_count": vs.count(self.COLLECTION),
        }


def _category_counts(metas: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for m in metas:
        c = m.get("category", "other")
        out[c] = out.get(c, 0) + 1
    return out


_singleton: Optional[FilingsIngester] = None


def get_filings_ingester() -> FilingsIngester:
    global _singleton
    if _singleton is None:
        _singleton = FilingsIngester()
    return _singleton
