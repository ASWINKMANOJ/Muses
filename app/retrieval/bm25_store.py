# app/retrieval/bm25_store.py
"""
BM25 sparse index for hybrid retrieval.

Maintains an in-memory BM25 index that is persisted to disk (pickle) so it
survives server restarts. The index is keyed by chunk id and stores the raw
text for tokenisation.

Why BM25 alongside dense retrieval?
- Dense (semantic) search finds conceptually similar passages even with
  different wording. It struggles with exact matches.
- BM25 (keyword) search excels at exact term matching — critical for legal
  queries like "Section 302 IPC" or "Article 7(2)(b)".
- Combining both via Reciprocal Rank Fusion (RRF) consistently outperforms
  either approach alone without requiring any training.
"""

import pickle
import re
import threading
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from app.core.config import settings


# ── Simple legal tokeniser ────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[.\-'][a-zA-Z0-9]+)*")

def _tokenise(text: str) -> list[str]:
    """
    Tokenise text for BM25, preserving legal tokens like:
    - Clause numbers: "4.2.1"
    - Section refs:   "302-IPC", "7(2)(b)"
    - Defined terms:  "Material-Adverse-Change"
    Returns lowercase tokens.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ── BM25 Store ────────────────────────────────────────────────────────────────

class BM25Store:
    """
    Thread-safe BM25 index with persistence.

    Internal state:
        _ids   : list[str]  — chunk ids in corpus order
        _texts : list[str]  — raw texts in corpus order
        _index : BM25Okapi  — the BM25 model (rebuilt from _texts on load)
    """

    def __init__(self, index_path: Optional[str] = None):
        self._path = Path(index_path or settings.bm25_index_path)
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._index: Optional[BM25Okapi] = None
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load index from disk if it exists."""
        if self._path.exists():
            try:
                with open(self._path, "rb") as f:
                    state = pickle.load(f)
                self._ids = state["ids"]
                self._texts = state["texts"]
                self._index = BM25Okapi(
                    [_tokenise(t) for t in self._texts]
                )
                print(f"[bm25] Loaded {len(self._ids)} docs from {self._path}")
            except Exception as e:
                print(f"[bm25] Failed to load index ({e}), starting fresh.")
                self._reset()
        else:
            self._reset()

    def _save(self) -> None:
        """Persist current index state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump({"ids": self._ids, "texts": self._texts}, f)

    def _reset(self) -> None:
        self._ids = []
        self._texts = []
        self._index = None

    def _rebuild_index(self) -> None:
        """Rebuild BM25Okapi from current corpus. Must be called under lock."""
        if self._texts:
            tokenised = [_tokenise(t) for t in self._texts]
            self._index = BM25Okapi(tokenised)
        else:
            self._index = None

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Add chunks to the index. Skips chunks whose id already exists.
        Rebuilds the BM25 model and persists to disk.
        """
        with self._lock:
            existing = set(self._ids)
            added = 0
            for chunk in chunks:
                cid = chunk["id"]
                if cid not in existing:
                    self._ids.append(cid)
                    self._texts.append(chunk["text"])
                    existing.add(cid)
                    added += 1
            if added:
                self._rebuild_index()
                self._save()
                print(f"[bm25] Added {added} chunks. Total: {len(self._ids)}")

    def delete_by_source(self, source: str) -> int:
        """
        Remove all chunks whose id starts with `source` (matches naming convention
        `{source}_{index}` used by the chunker).
        Returns number of chunks removed.
        """
        with self._lock:
            prefix = source + "_"
            keep = [
                (cid, txt)
                for cid, txt in zip(self._ids, self._texts)
                if not cid.startswith(prefix)
            ]
            removed = len(self._ids) - len(keep)
            if removed:
                self._ids = [k[0] for k in keep]
                self._texts = [k[1] for k in keep]
                self._rebuild_index()
                self._save()
                print(f"[bm25] Removed {removed} chunks for '{source}'.")
            return removed

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(self, query: str, n: int = 20) -> list[tuple[str, float]]:
        """
        BM25 search. Returns list of (chunk_id, bm25_score) sorted best-first.
        Returns empty list if index is empty.
        """
        with self._lock:
            if self._index is None or not self._ids:
                return []
            tokens = _tokenise(query)
            scores = self._index.get_scores(tokens)
            ranked = sorted(
                zip(self._ids, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return ranked[:n]

    @property
    def size(self) -> int:
        return len(self._ids)


# ── Module-level singleton ────────────────────────────────────────────────────

_bm25_store: Optional[BM25Store] = None


def get_bm25_store() -> BM25Store:
    """Return the module-level BM25Store singleton (lazy init)."""
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store()
    return _bm25_store
