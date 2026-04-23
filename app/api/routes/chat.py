# app/api/routes/chat.py
import json
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.retrieval.vector_store import query_similar
from app.generation.llm import stream_answer

router = APIRouter()

# Number of reranked chunks to pass to the LLM
TOP_K = 5


class ChatRequest(BaseModel):
    query: str
    document_filter: list[str] = []   # empty = search all documents


def _build_citations(results: list) -> list:
    """Extract unique citation metadata from retrieval results."""
    seen = set()
    citations = []
    for _score, _doc, meta in results:
        key = (meta.get("source", ""), meta.get("page", ""), meta.get("heading", ""))
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": meta.get("source", ""),
                "page": meta.get("page", ""),
                "heading": meta.get("heading", ""),
            })
    return citations


async def _sse_generator(
    query: str,
    document_filter: list[str],
) -> AsyncGenerator[str, None]:
    """
    Async SSE generator:
    1. Retrieves relevant chunks (optionally filtered to selected documents).
    2. Streams LLM tokens as   data: {"token": "..."}
    3. Sends a final event:    data: {"done": true, "citations": [...]}
    """
    # ── Retrieve & rerank ────────────────────────────────────────────────────
    results = query_similar(
        query,
        n_results=10,
        document_filter=document_filter if document_filter else None,
    )
    top_results = results[:TOP_K]

    top_chunks: list[str] = []
    for _score, doc, meta in top_results:
        formatted = (
            f"[Source: {meta['source']} | Page: {meta['page']} "
            f"| Section: {meta['heading']}]\n{doc}"
        )
        top_chunks.append(formatted)

    citations = _build_citations(top_results)

    # ── Stream LLM tokens ────────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    token_gen = stream_answer(query, top_chunks)

    def _next_token():
        try:
            return next(token_gen)
        except StopIteration:
            return None

    while True:
        token = await loop.run_in_executor(None, _next_token)
        if token is None:
            break
        yield f"data: {json.dumps({'token': token})}\n\n"

    # ── Final event with citations ───────────────────────────────────────────
    yield f"data: {json.dumps({'done': True, 'citations': citations})}\n\n"


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Stream an answer for the given query using SSE.

    Optionally restrict retrieval to a subset of documents via `document_filter`
    (list of source filenames).  Empty list = search all documents.

    Response format (Server-Sent Events):
      data: {"token": "<partial text>"}          — repeated for each token
      data: {"done": true, "citations": [...]}   — final event
    """
    return StreamingResponse(
        _sse_generator(request.query, request.document_filter),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
