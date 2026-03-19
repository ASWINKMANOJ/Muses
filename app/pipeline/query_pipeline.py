# app/pipeline/query_pipeline.py

from app.retrieval.vector_store import query_similar
from app.generation.llm import stream_answer


def query_pipeline_stream(query: str):
    results = query_similar(query)

    top_chunks = []

    for score, doc, meta in results[:3]:
        formatted = f"""
    [Source: {meta['source']} | Page: {meta['page']} | Section: {meta['heading']}]
    {doc}
    """
        top_chunks.append(formatted)

    return stream_answer(query, top_chunks)