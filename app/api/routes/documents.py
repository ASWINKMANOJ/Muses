# app/api/routes/documents.py
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.retrieval.vector_store import collection

router = APIRouter()

UPLOADS_DIR = Path("uploads")
MIME_MAP = {
    ".pdf":  "application/pdf",
    ".txt":  "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
}


@router.get("/documents")
async def list_documents():
    """
    Return a list of all unique source documents currently in the vector store.
    """
    try:
        # Fetch all metadata from ChromaDB
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])

        # Aggregate chunk counts per source file
        doc_stats: dict[str, int] = {}
        for meta in metadatas:
            source = meta.get("source", "unknown")
            doc_stats[source] = doc_stats.get(source, 0) + 1

        docs = [
            {"filename": source, "chunks": count}
            for source, count in sorted(doc_stats.items())
        ]
        return {"documents": docs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")


@router.get("/documents/{filename}/download")
async def download_document(filename: str):
    """
    Download an uploaded source document by filename.
    The filename must match the original uploaded name (path traversal is blocked).
    """
    # Sanitise: strip any directory component
    safe_name = Path(filename).name
    file_path = UPLOADS_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{safe_name}' not found in uploads.",
        )

    ext = file_path.suffix.lower()
    media_type = MIME_MAP.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=safe_name,
    )
