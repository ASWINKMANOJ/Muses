import os
from app.ingestion.loaders.pdf_parser import extract_sections_from_pdf
from app.ingestion.loaders.text_parser import extract_sections_from_txt
from app.ingestion.loaders.docx_parser import extract_sections_from_docx
from app.ingestion.loaders.image_parser import extract_sections_from_image
from app.ingestion.chunking.chunker import chunk_sections
from app.embedding.embedder import embed_chunks
from app.retrieval.vector_store import store_embeddings

def ingest_file_pipeline(file_path: str):
    """
    Full ingestion pipeline:
    File → Sections → Chunks → Embeddings → DB
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

    # Step 3: Embed
    embeddings = embed_chunks(chunks)

    # Step 4: Store
    store_embeddings(chunks, embeddings)

    return {
        "status": "success",
        "chunks": len(chunks)
    }

# Keep old name for backwards compatibility temporarily if needed, 
# although we will update main.py next anyway.
def ingest_pdf_pipeline(file_path: str):
    return ingest_file_pipeline(file_path)