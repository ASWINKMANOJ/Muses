# test_eval.py
"""
CLI tool to run automated benchmark evaluations on Muses RAG.
Generates an evaluation report JSON file.
"""

import json
import argparse
from app.evaluation.evaluator import RAGBenchmarkEvaluator
from app.core.config import settings


def main():
    parser = argparse.ArgumentParser(description="Muses RAG Automated Evaluator")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieval depth to evaluate")
    parser.add_argument("--output", type=str, default=settings.eval_report_path, help="Path to save report JSON")
    args = parser.parse_args()

    print("\n🔍 Running Muses RAG Benchmark Evaluation...")
    evaluator = RAGBenchmarkEvaluator()
    report = evaluator.evaluate(top_k=args.top_k)

    print("\n" + "=" * 50)
    print(" 📊 MUSES RAG BENCHMARK RESULTS")
    print("=" * 50)
    print(f" Total Queries Tested : {report.get('total_test_queries')}")
    print(f" Top-K Depth          : {report.get('top_k')}")
    metrics = report.get("metrics", {})
    print(f" Hit Rate @ {args.top_k}       : {metrics.get('hit_rate_at_k') * 100:.2f}%")
    print(f" Mean Reciprocal Rank : {metrics.get('mrr'):.4f}")
    print(f" Avg Retrieval Score  : {metrics.get('avg_retrieval_score'):.4f}")
    print(f" Avg Latency          : {metrics.get('avg_latency_ms'):.1f} ms")
    print("=" * 50)

    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✔ Benchmark report saved to: {output_path}\n")


if __name__ == "__main__":
    main()
