# app/pipeline/query_pipeline.py
"""
Query pipeline with HyDE (Hypothetical Document Embedding) support.

Improvements:
- HyDE: generates a hypothetical legal provision via the LLM and uses it
  to augment the query embedding, improving recall for paraphrased queries.
- Passes hyde_text to query_similar so dense retrieval can leverage it.
- Uses settings.retrieval_top_k instead of a hardcoded constant.
- Richer citation format includes section_path and chunk_type.
"""

from app.retrieval.vector_store import query_similar
from app.generation.llm import stream_answer, generate_hypothetical_answer
from app.core.config import settings


def query_pipeline_stream(query: str, document_filter: list[str] | None = None):
    """
    Full query pipeline:
      1. (Optional) HyDE — generate a hypothetical legal provision
      2. Hybrid retrieval — dense + BM25 + RRF
      3. Format top-k chunks with rich citation metadata
      4. Stream LLM answer token by token

    Args:
        query:           User's natural-language question.
        document_filter: Optional list of source filenames to restrict search.

    Yields:
        str tokens from the LLM.
    """
    # ── Step 1: HyDE query expansion ─────────────────────────────────────────
    hyde_text = None
    if settings.hyde_enabled:
        print("[query_pipeline] Generating HyDE hypothetical answer…")
        hyde_text = generate_hypothetical_answer(query)
        if hyde_text:
            print(f"[query_pipeline] HyDE text ({len(hyde_text)} chars): {hyde_text[:80]}…")

    # ── Step 2: Hybrid retrieval ──────────────────────────────────────────────
    results = query_similar(
        query,
        document_filter=document_filter or None,
        hyde_text=hyde_text,
    )

    if not results:
        yield "No relevant documents found. Please ingest legal documents first."
        return

    # ── Step 3: Format context chunks ─────────────────────────────────────────
    top_chunks: list[str] = []
    for score, doc, meta in results:
        source      = meta.get("source", "Unknown")
        page        = meta.get("page", "?")
        section     = meta.get("section_path") or meta.get("heading", "")
        clause      = meta.get("clause_number", "")
        chunk_type  = meta.get("chunk_type", "")

        header_parts = [f"Doc: {source}", f"Page: {page}"]
        if clause:
            header_parts.append(f"Clause: {clause}")
        if section:
            header_parts.append(f"Section: {section}")
        if chunk_type and chunk_type != "general":
            header_parts.append(f"Type: {chunk_type}")

        formatted = f"[{' | '.join(header_parts)}]\n{doc}"
        top_chunks.append(formatted)

    # ── Step 4: Stream LLM answer ─────────────────────────────────────────────
    yield from stream_answer(query, top_chunks)
