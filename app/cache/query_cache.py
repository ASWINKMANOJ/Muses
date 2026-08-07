# app/cache/query_cache.py
"""
Semantic Query Cache for Muses RAG.

Stores query embeddings and their corresponding retrieval + response results.
When a new user question has high cosine similarity (>= threshold) to a previously
seen question *with the same document scope*, the cache returns the cached
results instantly.
"""

import time
import math
import threading
from typing import Optional, Tuple, Dict, Any, List
from app.embedding.embedder import embed_query
from app.core.config import settings


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize_filter(document_filter: Optional[List[str]]) -> Tuple[str, ...]:
    """Canonical scope key so cache entries never leak across document filters."""
    if not document_filter:
        return ()
    return tuple(sorted({f for f in document_filter if f}))


class SemanticQueryCache:
    """Thread-safe vector similarity query cache, scoped by document filter."""

    def __init__(self, max_size: int = 200, default_threshold: float = 0.92):
        self.max_size = max_size
        self.default_threshold = default_threshold
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def get(
        self,
        query: str,
        threshold: Optional[float] = None,
        document_filter: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], float]:
        """
        Look up a query in the cache for the given document scope.

        Returns:
            (cache_hit, entry_dict, similarity_score)
        """
        if not settings.semantic_cache_enabled:
            return False, None, 0.0

        threshold = threshold or self.default_threshold
        scope = _normalize_filter(document_filter)
        query_vec = embed_query(query)

        best_score = -1.0
        best_entry = None

        with self._lock:
            for entry in self._entries:
                if entry.get("document_filter", ()) != scope:
                    continue
                score = _cosine_similarity(query_vec, entry["embedding"])
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry and best_score >= threshold:
            best_entry["hits"] += 1
            best_entry["last_accessed"] = time.time()
            return True, best_entry, best_score

        return False, None, max(0.0, best_score)

    def put(
        self,
        query: str,
        formatted_chunks: List[str],
        generated_answer: str,
        retrieval_results: Optional[List[Any]] = None,
        document_filter: Optional[List[str]] = None,
    ) -> None:
        """Add a query and its results to the cache under the given document scope."""
        if not settings.semantic_cache_enabled:
            return

        query_vec = embed_query(query)
        now = time.time()
        scope = _normalize_filter(document_filter)

        entry = {
            "query": query,
            "document_filter": scope,
            "embedding": query_vec,
            "formatted_chunks": formatted_chunks,
            "answer": generated_answer,
            "retrieval_results": retrieval_results or [],
            "hits": 0,
            "created_at": now,
            "last_accessed": now,
        }

        with self._lock:
            if len(self._entries) >= self.max_size:
                self._entries.sort(key=lambda x: (x["hits"], x["last_accessed"]))
                self._entries.pop(0)

            self._entries.append(entry)

    def clear(self) -> int:
        """Clear all entries (useful when documents are ingested or deleted)."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def stats(self) -> Dict[str, Any]:
        """Return operational statistics about the cache."""
        with self._lock:
            total_hits = sum(e["hits"] for e in self._entries)
            scopes = {e.get("document_filter", ()) for e in self._entries}
            return {
                "enabled": settings.semantic_cache_enabled,
                "cached_queries_count": len(self._entries),
                "total_cache_hits": total_hits,
                "threshold": self.default_threshold,
                "distinct_scopes": len(scopes),
            }


# Global singleton instance
_cache_instance = SemanticQueryCache(
    max_size=settings.semantic_cache_max_size,
    default_threshold=settings.semantic_cache_threshold,
)


def get_query_cache() -> SemanticQueryCache:
    return _cache_instance
