# app/retrieval/vector_store.py

import chromadb
from app.embedding.embedder import model

client = chromadb.PersistentClient(path="db")

collection = client.get_or_create_collection(name="documents")


# ── Storage ───────────────────────────────────────────────────────────────────

def store_embeddings(chunks: list[dict], embeddings: list) -> None:
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "heading": chunk["heading"],
            "source": chunk["source"],
            "page": chunk["page"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"[vector_store] Stored {len(chunks)} chunks.")


def delete_document_chunks(source: str) -> int:
    """
    Delete all chunks whose metadata 'source' matches the given filename.
    Returns the number of chunks deleted.
    """
    existing = collection.get(
        where={"source": source},
        include=[],          # only need IDs
    )
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
        print(f"[vector_store] Deleted {len(ids)} stale chunks for '{source}'.")
    return len(ids)


# ── Reranking ─────────────────────────────────────────────────────────────────

def _keyword_overlap_score(query: str, text: str) -> float:
    """
    Generic BM25-inspired keyword overlap boost.
    Returns a value in [0, 1] representing the fraction of unique query
    terms that appear in the chunk text.  Higher = more overlap.
    """
    query_terms = set(query.lower().split())
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for term in query_terms if term in text_lower)
    return matches / len(query_terms)


def rerank_results(results: dict, query: str) -> list[tuple[float, str, dict]]:
    """
    Rerank raw ChromaDB results with a generic keyword-overlap boost.
    Lower final score = better (cosine distance minus overlap bonus).
    """
    reranked = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        overlap = _keyword_overlap_score(query, doc)
        # Subtract overlap bonus (0–0.3 range) from distance so
        # high-overlap chunks float to the top.
        score = dist - (overlap * 0.3)
        reranked.append((score, doc, meta))

    reranked.sort(key=lambda x: x[0])
    return reranked


# ── Query ─────────────────────────────────────────────────────────────────────

def query_similar(
    query: str,
    n_results: int = 10,
    document_filter: list[str] | None = None,
) -> list[tuple[float, str, dict]]:
    """
    Retrieve semantically similar chunks and rerank them.

    Args:
        query:           Natural-language query string.
        n_results:       Number of candidates to fetch from ChromaDB before reranking.
        document_filter: Optional list of source filenames to restrict retrieval to.
                         Passes a `where` clause to ChromaDB.
                         If None or empty, all documents are searched.

    Returns:
        List of (score, document_text, metadata) tuples, sorted best-first.
    """
    query_embedding = model.encode([query])[0].tolist()

    # Build optional where clause
    where_clause = None
    if document_filter:
        if len(document_filter) == 1:
            where_clause = {"source": document_filter[0]}
        else:
            where_clause = {"source": {"$in": document_filter}}

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if where_clause:
        query_kwargs["where"] = where_clause

    # Clamp n_results to the number of available items (ChromaDB errors otherwise)
    total = collection.count()
    if total == 0:
        return []
    if where_clause:
        # Count docs in filter subset
        subset = collection.get(where=where_clause, include=[])
        available = len(subset.get("ids", []))
    else:
        available = total
    query_kwargs["n_results"] = max(1, min(n_results, available))

    results = collection.query(**query_kwargs)

    return rerank_results(results, query)