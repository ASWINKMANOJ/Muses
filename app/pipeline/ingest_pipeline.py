import os
from pathlib import Path

from app.ingestion.loaders.pdf_parser import extract_sections_from_pdf
from app.ingestion.loaders.text_parser import extract_sections_from_txt
from app.ingestion.loaders.docx_parser import extract_sections_from_docx
from app.ingestion.loaders.image_parser import extract_sections_from_image
from app.ingestion.chunking.chunker import chunk_sections
from app.embedding.embedder import embed_chunks
from app.retrieval.vector_store import store_embeddings, delete_document_chunks


def ingest_file_pipeline(file_path: str) -> dict:
    """
    Full ingestion pipeline:  File → Sections → Chunks → Embeddings → DB

    Implements an upsert pattern: any existing chunks for this source file
    are deleted before the new chunks are stored, preventing duplicates on
    re-upload.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Step 1: Extract sections based on file type
    if ext == ".pdf":
        sections = extract_sections_from_pdf(file_path)
    elif ext == ".txt":
        sections = extract_sections_from_txt(file_path)
    elif ext == ".docx":
        sections = extract_sections_from_docx(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        sections = extract_sections_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not sections:
        return {"status": "error", "message": "No content extracted", "chunks": 0}

    # Step 2: Chunk
    chunks = chunk_sections(sections)

    # Step 3: Upsert — remove stale chunks for this source before storing
    filename = Path(file_path).name
    deleted = delete_document_chunks(filename)
    if deleted:
        print(f"[ingest_pipeline] Replaced {deleted} stale chunks for '{filename}'.")

    # Step 4: Embed
    embeddings = embed_chunks(chunks)

    # Step 5: Store
    store_embeddings(chunks, embeddings)

    return {
        "status": "success",
        "chunks": len(chunks),
        "replaced": deleted,
    }


# Backwards-compatibility alias
def ingest_pdf_pipeline(file_path: str) -> dict:
    return ingest_file_pipeline(file_path)