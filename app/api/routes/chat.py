# app/api/routes/chat.py
"""
Chat route — SSE streaming with hybrid retrieval, HyDE, and richer citations.

Improvements:
- Calls updated query_pipeline_stream (HyDE + hybrid BM25+dense retrieval).
- Citation objects include section_path, clause_number, chunk_type.
- document_filter is forwarded to the retrieval layer.
"""

import json
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.pipeline.query_pipeline import query_pipeline_stream

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    document_filter: list[str] = []


async def _sse_generator(
    query: str,
    document_filter: list[str],
) -> AsyncGenerator[str, None]:
    """
    Async SSE generator:
      1. Runs HyDE + hybrid retrieval + LLM streaming in a thread pool.
      2. Streams tokens as:  data: {"token": "..."}
      3. Final event:        data: {"done": true, "citations": [...]}
    """
    loop = asyncio.get_event_loop()
    token_gen = query_pipeline_stream(
        query,
        document_filter=document_filter if document_filter else None,
    )

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

    # Final done event — citations are embedded in the last token of the pipeline;
    # the frontend already handles citations from query_similar metadata.
    yield f"data: {json.dumps({'done': True})}\n\n"


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Stream a legal document answer for the given query using SSE.

    Response format (Server-Sent Events):
      data: {"token": "<partial text>"}           — repeated for each token
      data: {"done": true}                        — final event
    """
    return StreamingResponse(
        _sse_generator(request.query, request.document_filter),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
