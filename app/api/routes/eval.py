# app/api/routes/eval.py
"""
Evaluation and Cache statistics routes.

Exposes endpoints for running automated benchmark evaluations
and monitoring/clearing the semantic query cache.
"""

from fastapi import APIRouter
from app.evaluation.evaluator import RAGBenchmarkEvaluator
from app.cache import get_query_cache

router = APIRouter()


@router.get("/eval/benchmark")
async def run_benchmark(top_k: int = 5):
    """
    Run automated benchmark evaluation across test query datasets.
    Returns Hit Rate @ K, MRR, average retrieval confidence, and latency.
    """
    evaluator = RAGBenchmarkEvaluator()
    report = evaluator.evaluate(top_k=top_k)
    return report


@router.get("/cache/stats")
async def cache_stats():
    """Get operational statistics for the semantic query cache."""
    cache = get_query_cache()
    return cache.stats()


@router.post("/cache/clear")
async def clear_cache():
    """Clear all entries in the semantic query cache."""
    cache = get_query_cache()
    cleared = cache.clear()
    return {"status": "success", "cleared_entries": cleared}
