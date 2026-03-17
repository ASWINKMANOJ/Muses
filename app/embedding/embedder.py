# app/embedding/embedder.py

from sentence_transformers import SentenceTransformer

# Load once (global, fast reuse)
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """
    Convert chunk texts into embeddings.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings