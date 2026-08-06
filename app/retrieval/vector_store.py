# app/retrieval/vector_store.py
"""
Hybrid retrieval: Dense (ChromaDB) + Sparse (BM25) fused via Reciprocal Rank Fusion (RRF).

Improvements:
- Hybrid retrieval: fetches candidates from both ChromaDB and BM25, then merges
  with RRF — parameter-free, consistently beats score-sum combination.
- Richer metadata stored per chunk: section_path, chunk_type, clause_number,
  file_hash — enables precise filtering (e.g. "find all indemnity clauses").
- Optional cross-encoder reranker for top-k precision (off by default).
- All constants come from centralized settings.
"""

import chromadb

from app.embedding.embedder import embed_query, embed_chunks
from app.retrieval.bm25_store import get_bm25_store
from app.core.config import settings


# ── ChromaDB client ───────────────────────────────────────────────────────────

client = chromadb.PersistentClient(path=settings.chroma_db_path)
collection = client.get_or_create_collection(
    name=settings.chroma_collection,
    metadata={"hnsw:space": "cosine"},   # use cosine similarity
)


# ── Storage ───────────────────────────────────────────────────────────────────

def store_embeddings(chunks: list[dict], embeddings: list) -> None:
    """
    Upsert chunks into ChromaDB and BM25 index.

    Stores richer metadata: section_path, chunk_type, clause_number, file_hash.
    """
    if not chunks:
        return

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = []

    for chunk in chunks:
        meta = {
            "heading":      chunk.get("heading", ""),
            "source":       chunk.get("source", ""),
            "page":         chunk.get("page", 1),
            "section_path": chunk.get("section_path", chunk.get("heading", "")),
            "chunk_type":   chunk.get("chunk_type", "general"),
            "clause_number": chunk.get("clause_number", ""),
            "file_hash":    chunk.get("file_hash", ""),
            "file_type":    chunk.get("file_type", ""),
        }
        # ChromaDB metadata values must be str/int/float/bool
        metadatas.append({k: str(v) if v is not None else "" for k, v in meta.items()})

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"[vector_store] Upserted {len(chunks)} chunks into ChromaDB.")

    # Also index in BM25
    bm25 = get_bm25_store()
    bm25.add_chunks(chunks)


def delete_document_chunks(source: str) -> int:
    """
    Delete all chunks for a given source from both ChromaDB and BM25.
    Returns the number of chunks deleted from ChromaDB.
    """
    existing = collection.get(where={"source": source}, include=[])
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
        print(f"[vector_store] Deleted {len(ids)} chunks from ChromaDB for '{source}'.")

    bm25 = get_bm25_store()
    bm25.delete_by_source(source)

    return len(ids)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_merge(
    dense_ranked: list[tuple[str, float]],   # (chunk_id, distance)  lower=better
    sparse_ranked: list[tuple[str, float]],  # (chunk_id, bm25_score) higher=better
    k: int = 60,
) -> list[str]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1/(k + rank_i) for each list the item appears in.
    Higher RRF score = better result.

    Args:
        k: RRF smoothing constant (60 is the standard default).
    Returns:
        List of chunk_ids ordered best-first.
    """
    scores: dict[str, float] = {}

    for rank, (cid, _) in enumerate(dense_ranked, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    for rank, (cid, _) in enumerate(sparse_ranked, start=1):
        scores[cid] = scores.get(cid, 0.0) + (settings.bm25_weight / (k + rank))

    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ── Optional cross-encoder reranker ──────────────────────────────────────────

_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(settings.cross_encoder_model)
        print(f"[vector_store] Cross-encoder loaded: {settings.cross_encoder_model}")
    return _cross_encoder


def _cross_encode(query: str, candidates: list[tuple]) -> list[tuple]:
    """Re-rank candidates (score, doc, meta) using a cross-encoder."""
    try:
        model = _get_cross_encoder()
        pairs = [(query, doc) for _, doc, _ in candidates]
        scores = model.predict(pairs)
        reranked = sorted(
            zip(scores, [doc for _, doc, _ in candidates], [m for _, _, m in candidates]),
            key=lambda x: x[0],
            reverse=True,
        )
        return reranked
    except Exception as e:
        print(f"[vector_store] Cross-encoder failed ({e}), using RRF order.")
        return candidates


# ── Query ─────────────────────────────────────────────────────────────────────

def query_similar(
    query: str,
    n_results: int | None = None,
    document_filter: list[str] | None = None,
    hyde_text: str | None = None,
) -> list[tuple[float, str, dict]]:
    """
    Hybrid retrieval: ChromaDB (dense) + BM25 (sparse), merged with RRF.

    Args:
        query:           Natural-language query string.
        n_results:       Number of final results to return.
        document_filter: Restrict retrieval to specific source filenames.
        hyde_text:       Hypothetical answer text (HyDE). If provided, its
                         embedding is averaged with the query embedding.

    Returns:
        List of (score, document_text, metadata) tuples, best-first.
        Score is the RRF score (higher = better) when hybrid, else distance.
    """
    n_results = n_results or settings.retrieval_n_candidates
    top_k = settings.retrieval_top_k

    # ── 1. Dense retrieval ────────────────────────────────────────────────────
    query_vec = embed_query(query)

    # HyDE: average query + hypothetical-answer embeddings
    if hyde_text:
        hyde_vec = embed_query(hyde_text)
        query_vec = [(q + h) / 2 for q, h in zip(query_vec, hyde_vec)]

    # Build ChromaDB filter
    where_clause = None
    if document_filter:
        if len(document_filter) == 1:
            where_clause = {"source": document_filter[0]}
        else:
            where_clause = {"source": {"$in": document_filter}}

    total = collection.count()
    if total == 0:
        return []

    # Clamp n_results to available
    available = total
    if where_clause:
        subset = collection.get(where=where_clause, include=[])
        available = len(subset.get("ids", []))

    n_fetch = max(1, min(n_results, available))

    chroma_kwargs = {
        "query_embeddings": [query_vec],
        "n_results": n_fetch,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        chroma_kwargs["where"] = where_clause

    chroma_results = collection.query(**chroma_kwargs)

    docs = chroma_results["documents"][0]
    metas = chroma_results["metadatas"][0]
    distances = chroma_results["distances"][0]
    ids_from_chroma = chroma_results.get("ids", [[]])[0]

    # Dense ranked list: (id, distance) — lower distance = better
    dense_ranked = list(zip(ids_from_chroma, distances))

    # ── 2. Sparse BM25 retrieval ──────────────────────────────────────────────
    bm25 = get_bm25_store()
    sparse_ranked_raw = bm25.search(query, n=n_results)

    # If document_filter is active, filter BM25 results to matching sources
    if document_filter:
        filter_set = set(document_filter)
        sparse_ranked_raw = [
            (cid, score) for cid, score in sparse_ranked_raw
            if any(cid.startswith(src + "_") for src in filter_set)
        ]

    # ── 3. RRF merge ──────────────────────────────────────────────────────────
    merged_ids = _rrf_merge(dense_ranked, sparse_ranked_raw)

    # ── 4. Build id→(doc, meta) lookup ───────────────────────────────────────
    id_to_doc = dict(zip(ids_from_chroma, docs))
    id_to_meta = dict(zip(ids_from_chroma, metas))

    # For BM25-only results not in ChromaDB result set, fetch from ChromaDB
    missing_ids = [cid for cid in merged_ids if cid not in id_to_doc]
    if missing_ids:
        fetch_n = min(len(missing_ids), 100)
        fetched = collection.get(ids=missing_ids[:fetch_n], include=["documents", "metadatas"])
        for fid, fdoc, fmeta in zip(
            fetched.get("ids", []),
            fetched.get("documents", []),
            fetched.get("metadatas", []),
        ):
            id_to_doc[fid] = fdoc
            id_to_meta[fid] = fmeta

    # ── 5. Assemble final result list ─────────────────────────────────────────
    results: list[tuple[float, str, dict]] = []
    seen = set()
    for rank, cid in enumerate(merged_ids):
        if cid in seen or cid not in id_to_doc:
            continue
        seen.add(cid)
        rrf_score = 1.0 / (60 + rank + 1)
        results.append((rrf_score, id_to_doc[cid], id_to_meta[cid]))

    results = results[:top_k * 2]   # keep a larger pool for cross-encoder

    # ── 6. Optional cross-encoder reranking ───────────────────────────────────
    if settings.cross_encoder_enabled and results:
        results = _cross_encode(query, results)

    return results[:top_k]
