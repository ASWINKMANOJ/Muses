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


class ChatRequest(BaseModel):
    query: str


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


async def _sse_generator(query: str) -> AsyncGenerator[str, None]:
    """
    Async SSE generator that:
    1. Streams LLM tokens as  data: {"token": "..."}
    2. Sends a final event:   data: {"done": true, "citations": [...]}
    """
    # Retrieve relevant chunks
    results = query_similar(query)
    top_results = results[:3]

    top_chunks = []
    for score, doc, meta in top_results:
        formatted = (
            f"[Source: {meta['source']} | Page: {meta['page']} "
            f"| Section: {meta['heading']}]\n{doc}"
        )
        top_chunks.append(formatted)

    citations = _build_citations(top_results)

    # Stream tokens from LLM (stream_answer is a sync generator — run in executor)
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
        payload = json.dumps({"token": token})
        yield f"data: {payload}\n\n"

    # Final event with citations
    done_payload = json.dumps({"done": True, "citations": citations})
    yield f"data: {done_payload}\n\n"


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Stream an answer for the given query using SSE.

    Response format (Server-Sent Events):
      data: {"token": "<partial text>"}   — repeated for each token
      data: {"done": true, "citations": [...]}  — final event
    """
    return StreamingResponse(
        _sse_generator(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
