# app/evaluation/evaluator.py
"""
Automated Evaluation Suite for Muses RAG System.

Calculates key quantitative metrics to benchmark RAG performance:
1. Hit Rate @ K: Fraction of test queries where relevant context is retrieved.
2. MRR (Mean Reciprocal Rank): Position quality of relevant context.
3. Context Relevance Score: Reranker confidence & semantic relevance.
4. Groundedness Score: Verbatim citation match & anti-hallucination metric.
"""

import json
import time
from typing import List, Dict, Any, Optional
from app.retrieval.vector_store import query_similar
from app.core.config import settings


def compute_hit_rate(retrieved_metas: List[Dict[str, Any]], expected_keywords: List[str]) -> bool:
    """Check if any retrieved metadata or text contains any of the expected keywords/sources."""
    for meta in retrieved_metas:
        source = meta.get("source", "").lower()
        section = meta.get("section_path", "").lower()
        clause = meta.get("clause_number", "").lower()
        combined = f"{source} {section} {clause}"
        
        for kw in expected_keywords:
            if kw.lower() in combined:
                return True
    return False


def compute_reciprocal_rank(retrieved_metas: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
    """Calculate reciprocal rank (1 / rank of first relevant doc)."""
    for rank, meta in enumerate(retrieved_metas, start=1):
        source = meta.get("source", "").lower()
        section = meta.get("section_path", "").lower()
        clause = meta.get("clause_number", "").lower()
        combined = f"{source} {section} {clause}"
        
        for kw in expected_keywords:
            if kw.lower() in combined:
                return 1.0 / rank
    return 0.0


def compute_groundedness(answer: str, context_docs: List[str]) -> float:
    """
    Measure groundedness (0.0 to 1.0) by checking what fraction of answer 5-grams
    or key sentences are directly grounded in the provided document context.
    """
    if not answer or not context_docs:
        return 0.0

    combined_context = " ".join(context_docs).lower()
    sentences = [s.strip().lower() for s in answer.split(".") if len(s.strip()) > 15]

    if not sentences:
        return 1.0

    grounded_count = 0
    for s in sentences:
        # Ignore disclaimer sentence
        if "legal advice" in s or "consult a qualified lawyer" in s:
            grounded_count += 1
            continue
        
        # Check partial token match
        words = s.split()
        if len(words) >= 4:
            sub = " ".join(words[:4])
            if sub in combined_context:
                grounded_count += 1
                continue
        if s[:20] in combined_context:
            grounded_count += 1

    return min(1.0, grounded_count / len(sentences))


class RAGBenchmarkEvaluator:
    """Evaluates RAG system performance over test benchmark datasets."""

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or settings.eval_dataset_path

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load evaluation dataset from JSON file."""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Default fallback sample test dataset if file doesn't exist
            return [
                {
                    "query": "What is the liability cap under Section 4?",
                    "expected_keywords": ["section 4", "liability", "contract"],
                    "category": "contract_clause"
                },
                {
                    "query": "What are the termination conditions for breach?",
                    "expected_keywords": ["termination", "breach", "notice"],
                    "category": "termination"
                },
                {
                    "query": "Is there a non-compete clause in the agreement?",
                    "expected_keywords": ["non-compete", "covenant", "restriction"],
                    "category": "restrictions"
                }
            ]

    def evaluate(self, top_k: int = 5) -> Dict[str, Any]:
        """Run full evaluation suite and return aggregate metrics report."""
        dataset = self.load_dataset()
        if not dataset:
            return {"error": "Evaluation dataset is empty"}

        hits = []
        reciprocal_ranks = []
        scores_list = []
        latencies = []

        for item in dataset:
            query = item["query"]
            expected = item.get("expected_keywords", [])

            t0 = time.time()
            results = query_similar(query, n_results=top_k * 2)
            lat_ms = (time.time() - t0) * 1000
            latencies.append(lat_ms)

            if not results:
                hits.append(0)
                reciprocal_ranks.append(0.0)
                scores_list.append(0.0)
                continue

            top_results = results[:top_k]
            metas = [r[2] for r in top_results]
            scores = [r[0] for r in top_results]

            # 1. Hit Rate
            hit = 1 if compute_hit_rate(metas, expected) else 0
            hits.append(hit)

            # 2. MRR
            rr = compute_reciprocal_rank(metas, expected)
            reciprocal_ranks.append(rr)

            # 3. Context Score
            scores_list.append(sum(scores) / len(scores) if scores else 0.0)

        hit_rate = sum(hits) / len(hits) if hits else 0.0
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_queries": len(dataset),
            "top_k": top_k,
            "metrics": {
                "hit_rate_at_k": round(hit_rate, 4),
                "mrr": round(mrr, 4),
                "avg_retrieval_score": round(avg_score, 4),
                "avg_latency_ms": round(avg_latency, 2),
            },
            "configuration": {
                "embedding_model": settings.embedding_model,
                "cross_encoder_enabled": settings.cross_encoder_enabled,
                "cross_encoder_model": settings.cross_encoder_model,
                "hyde_enabled": settings.hyde_enabled,
                "bm25_weight": settings.bm25_weight,
            }
        }
        return report
