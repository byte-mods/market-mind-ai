"""
MarketMind AI - Vector DB Foundation (Wave 0 F2)

Thin wrapper around ChromaDB for RAG over filings, concalls, broker
reports, news, and any other doc corpus.

Defaults to a *persistent* on-disk store at ``./marketmind/vectordb/``
so embeddings survive restarts.  Embedding model:
``all-MiniLM-L6-v2`` (Chroma default — runs on CPU, ~22 MB, no API).

Designed to be used by W2.1 (Filings RAG), W2.2 (event archive),
W2.3 (alt-data signals).

Usage::

    from marketmind.core.vectordb import get_vector_store
    vs = get_vector_store()

    vs.add(
        collection="filings",
        documents=["...annual report chunk..."],
        metadatas=[{"symbol":"RELIANCE", "year":2026, "section":"MD&A"}],
        ids=["RELIANCE_2026_mda_p1"],
    )

    hits = vs.query(
        collection="filings",
        text="how is debt position trending",
        k=5,
        where={"symbol": "RELIANCE"},
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[1] / "vectordb"


class VectorStore:
    """ChromaDB-backed vector store with one collection per concept."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = str(persist_dir or DEFAULT_PERSIST_DIR)
        os.makedirs(self.persist_dir, exist_ok=True)
        self._client = None
        self._collections: Dict[str, Any] = {}

    def _client_lazy(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False, allow_reset=False),
            )
            logger.info(f"VectorStore initialised at {self.persist_dir}")
        return self._client

    def _collection(self, name: str):
        if name not in self._collections:
            client = self._client_lazy()
            self._collections[name] = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    # ── Write ──────────────────────────────────────────────────────────────
    def add(
        self,
        collection: str,
        documents: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> int:
        """Add or upsert ``len(documents)`` items. Returns count added."""
        if not documents:
            return 0
        col = self._collection(collection)
        if ids is None:
            ids = [f"{collection}_{i}_{abs(hash(doc)) % (10**12)}" for i, doc in enumerate(documents)]
        if metadatas is None:
            metadatas = [{} for _ in documents]
        col.upsert(documents=list(documents), metadatas=list(metadatas), ids=list(ids))
        return len(documents)

    # ── Read ───────────────────────────────────────────────────────────────
    def query(
        self,
        collection: str,
        text: str,
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to ``k`` hits: [{id, document, metadata, distance}]."""
        col = self._collection(collection)
        try:
            res = col.query(
                query_texts=[text],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"VectorStore.query failed [{collection}]: {e}")
            return []
        out: List[Dict[str, Any]] = []
        ids   = (res.get("ids") or [[]])[0]
        docs  = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            out.append({
                "id": doc_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": float(dists[i]) if i < len(dists) else None,
            })
        return out

    def count(self, collection: str) -> int:
        try:
            return self._collection(collection).count()
        except Exception:
            return 0

    def list_collections(self) -> List[str]:
        client = self._client_lazy()
        try:
            return [c.name for c in client.list_collections()]
        except Exception:
            return []

    def delete(
        self,
        collection: str,
        ids: Optional[Sequence[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            col = self._collection(collection)
            col.delete(ids=list(ids) if ids else None, where=where)
            return True
        except Exception as e:
            logger.warning(f"VectorStore.delete failed [{collection}]: {e}")
            return False

    def reset_collection(self, collection: str) -> bool:
        """Drop and recreate the named collection."""
        try:
            client = self._client_lazy()
            try:
                client.delete_collection(name=collection)
            except Exception:
                pass
            self._collections.pop(collection, None)
            self._collection(collection)  # recreate
            return True
        except Exception as e:
            logger.warning(f"VectorStore.reset_collection failed [{collection}]: {e}")
            return False


# ── Singleton ──────────────────────────────────────────────────────────────
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
