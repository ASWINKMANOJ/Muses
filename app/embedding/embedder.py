# app/embedding/embedder.py
"""
Embedding module with retrieval-optimized bi-encoder support.

Using embedding_device="cpu" prevents CUDA VRAM competition with Ollama
on 4 GB GPUs, while maintaining sub-20ms embedding speed.
"""

from __future__ import annotations
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from app.core.config import settings

# BGE short-query → long-passage models expect this instruction on queries only.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _is_bge_model(model_name: str) -> bool:
    name = model_name.lower()
    return "bge-" in name or "/bge" in name


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Return a cached SentenceTransformer model.
    """
    device = settings.embedding_device
    print(f"[embedder] Loading embedding model '{settings.embedding_model}' on device: {device.upper()}")
    model = SentenceTransformer(
        settings.embedding_model,
        device=device,
        trust_remote_code=False,
    )
    print(f"[embedder] Model loaded on {device.upper()}. Embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of raw text strings (passages / chunks — no query instruction).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=len(texts) > 10,
        normalize_embeddings=settings.embedding_normalize,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    texts = [chunk["text"] for chunk in chunks]
    return embed_texts(texts)


def embed_query(query: str) -> list[float]:
    """
    Embed a search query. Applies the BGE query instruction when using a BGE model.
    """
    text = query
    if _is_bge_model(settings.embedding_model):
        text = f"{_BGE_QUERY_PREFIX}{query}"
    return embed_texts([text])[0]


class _LazyModel:
    def __getattr__(self, name):
        return getattr(get_model(), name)

    def encode(self, *args, **kwargs):
        return get_model().encode(*args, **kwargs)


model = _LazyModel()
