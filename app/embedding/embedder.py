from sentence_transformers import SentenceTransformer

# Global model (shared everywhere)
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings