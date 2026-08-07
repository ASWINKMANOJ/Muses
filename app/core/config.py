# app/core/config.py
"""
Centralized configuration for Muses RAG.

All tunable parameters live here. Values are read from the environment
(or a .env file loaded by python-dotenv) and fall back to sensible defaults.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM (Ollama - GPU) ───────────────────────────────────────────────────
    ollama_url: str = "http://localhost:11434/api/generate"
    llm_model: str = "gemma3:4b"
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_num_ctx: int = 4096
    llm_num_predict: int = 512
    llm_num_gpu: int = 99   # Offload LLM layers to GPU VRAM

    # ── Embedding Model (CPU to save VRAM for Ollama) ─────────────────────────
    # Use a trained bi-encoder (BGE), not an MLM like Legal-BERT.
    # Changing this requires re-ingesting docs into a fresh collection.
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cpu"  # 'cpu' prevents CUDA OOM with Ollama
    embedding_batch_size: int = 32
    embedding_normalize: bool = True

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ── Retrieval & Reranking ───────────────────────────────────────────────────
    retrieval_n_candidates: int = 20
    retrieval_top_k: int = 5
    bm25_weight: float = 0.5
    hyde_enabled: bool = True
    cross_encoder_enabled: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    crag_min_confidence: float = -5.0  # Minimum rerank logit threshold for CRAG check

    # ── Semantic Cache ────────────────────────────────────────────────────────
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.92
    semantic_cache_max_size: int = 200

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_dataset_path: str = "tests/eval_dataset.json"
    eval_report_path: str = "eval_report.json"

    # ── Vector DB ─────────────────────────────────────────────────────────────
    chroma_db_path: str = "db"
    # Bumped when switching to BGE so old Legal-BERT vectors are not mixed in.
    chroma_collection: str = "documents_bge_base"

    # ── Paths ─────────────────────────────────────────────────────────────────
    uploads_dir: str = "uploads"
    bm25_index_path: str = "db/bm25_index.pkl"
    ingest_manifest_path: str = "uploads/manifest.json"

    # ── API ───────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["*"]
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
