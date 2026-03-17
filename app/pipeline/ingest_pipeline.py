# app/pipeline/ingest_pipeline.py

from app.ingestion.loaders.pdf_parser import extract_sections_from_pdf
from app.ingestion.chunking.chunker import chunk_sections
from app.embedding.embedder import embed_chunks
from app.retrieval.vector_store import store_embeddings


def ingest_pdf_pipeline(file_path: str):
    """
    Full ingestion pipeline:
    PDF → Sections → Chunks → Embeddings → DB
    """

    # Step 1: Extract sections
    sections = extract_sections_from_pdf(file_path)

    # Step 2: Chunk
    chunks = chunk_sections(sections)

    # Step 3: Embed
    embeddings = embed_chunks(chunks)

    # Step 4: Store
    store_embeddings(chunks, embeddings)

    return {
        "status": "success",
        "chunks": len(chunks)
    }