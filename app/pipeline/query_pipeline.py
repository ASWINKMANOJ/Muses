# app/pipeline/query_pipeline.py
"""
Enterprise query pipeline with semantic cache, HyDE, hybrid retrieval, and CRAG.
"""

import time
from app.retrieval.vector_store import query_similar
from app.generation.llm import stream_answer, generate_hypothetical_answer
from app.core.config import settings
from app.cache import get_query_cache

_LOW_CONFIDENCE_REFUSAL = (
    "The retrieved excerpts are not confident enough matches for this question. "
    "I cannot reliably answer from the available documents without risking an "
    "unsupported conclusion.\n\n"
    "Try rephrasing the question, narrowing to a specific document, or ingesting "
    "additional source material.\n\n"
    "*This is document retrieval assistance, not legal advice. "
    "Consult a qualified lawyer for legal opinions.*"
)


def _is_low_confidence(results: list) -> bool:
    """True when cross-encoder is on and the best logit is below the CRAG floor."""
    if not results or not settings.cross_encoder_enabled:
        return False
    return results[0][0] < settings.crag_min_confidence


def _format_chunks(results: list) -> list[str]:
    top_chunks: list[str] = []
    for score, doc, meta in results:
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        section = meta.get("section_path") or meta.get("heading", "")
        clause = meta.get("clause_number", "")
        chunk_type = meta.get("chunk_type", "")

        header_parts = [f"Doc: {source}", f"Page: {page}"]
        if clause:
            header_parts.append(f"Clause: {clause}")
        if section:
            header_parts.append(f"Section: {section}")
        if chunk_type and chunk_type != "general":
            header_parts.append(f"Type: {chunk_type}")

        top_chunks.append(f"[{' | '.join(header_parts)}]\n{doc}")
    return top_chunks


def query_pipeline_stream(
    query: str,
    document_filter: list[str] | None = None,
    return_telemetry: bool = False,
):
    """
    Full enterprise query pipeline:
      0. Semantic Cache check (scoped by document_filter)
      1. HyDE query expansion (optional)
      2. Hybrid retrieval (Dense + BM25 RRF + Cross-Encoder Rerank)
      3. CRAG: corrective re-retrieve, then refuse if still low-confidence
      4. Format top-k chunks with rich citation metadata
      5. Stream LLM answer token by token & cache result
    """
    start_time = time.time()
    cache = get_query_cache()
    doc_filter = document_filter or None

    # ── Step 0: Semantic Cache Check (same document scope only) ───────────────
    hit, cached_entry, sim_score = cache.get(query, document_filter=doc_filter)
    if hit and cached_entry:
        elapsed_ms = (time.time() - start_time) * 1000
        print(
            f"[query_pipeline] ⚡ Semantic Cache HIT! "
            f"(similarity: {sim_score:.4f}, latency: {elapsed_ms:.1f}ms)"
        )
        if return_telemetry:
            yield (
                f"[CACHE_HIT | Similarity: {sim_score:.2f} | "
                f"Latency: {elapsed_ms:.1f}ms]\n\n"
            )
        yield cached_entry["answer"]
        return

    # ── Step 1: HyDE query expansion ─────────────────────────────────────────
    hyde_start = time.time()
    hyde_text = None
    if settings.hyde_enabled:
        print("[query_pipeline] Generating HyDE hypothetical answer…")
        hyde_text = generate_hypothetical_answer(query)
        if hyde_text:
            print(f"[query_pipeline] HyDE text ({len(hyde_text)} chars): {hyde_text[:80]}…")
    hyde_ms = (time.time() - hyde_start) * 1000

    # ── Step 2: Hybrid retrieval + Rerank ─────────────────────────────────────
    retrieval_start = time.time()
    results = query_similar(
        query,
        document_filter=doc_filter,
        hyde_text=hyde_text,
    )
    retrieval_ms = (time.time() - retrieval_start) * 1000

    if not results:
        yield "No relevant documents found. Please ingest legal documents first."
        return

    # ── Step 3: CRAG — corrective retrieval, then refuse if still weak ────────
    if _is_low_confidence(results):
        best = results[0][0]
        print(
            f"[query_pipeline] ⚠ CRAG: low confidence ({best:.3f} < "
            f"{settings.crag_min_confidence}). Running corrective retrieval…"
        )
        corrective_start = time.time()
        # Drop HyDE (can pull retrieval off-topic) and widen the candidate pool.
        results = query_similar(
            query,
            n_results=max(settings.retrieval_n_candidates * 2, 40),
            document_filter=doc_filter,
            hyde_text=None,
        )
        retrieval_ms += (time.time() - corrective_start) * 1000

        if not results or _is_low_confidence(results):
            score_note = f"{results[0][0]:.3f}" if results else "n/a"
            print(f"[query_pipeline] ⚠ CRAG: still low confidence ({score_note}). Refusing.")
            if return_telemetry:
                yield (
                    f"[CRAG_REFUSED | Score: {score_note} | "
                    f"Threshold: {settings.crag_min_confidence}]\n\n"
                )
            yield _LOW_CONFIDENCE_REFUSAL
            return

        print(
            f"[query_pipeline] ✓ CRAG corrective pass recovered "
            f"(best score: {results[0][0]:.3f})"
        )

    # ── Step 4: Format context chunks ─────────────────────────────────────────
    top_chunks = _format_chunks(results)

    # ── Step 5: Stream LLM answer & Cache output ──────────────────────────────
    full_answer_acc: list[str] = []

    for token in stream_answer(query, top_chunks):
        full_answer_acc.append(token)
        yield token

    total_ms = (time.time() - start_time) * 1000
    print(
        f"[query_pipeline] Query execution completed in {total_ms:.1f}ms "
        f"(HyDE: {hyde_ms:.1f}ms, Retrieval: {retrieval_ms:.1f}ms)"
    )

    complete_answer = "".join(full_answer_acc)
    if complete_answer:
        cache.put(
            query,
            top_chunks,
            complete_answer,
            results,
            document_filter=doc_filter,
        )
