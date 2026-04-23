# app/api/routes/documents.py
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.retrieval.vector_store import collection, delete_document_chunks

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
    Each entry includes the filename and its chunk count.
    """
    try:
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])

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
    Path traversal is blocked by stripping directory components.
    """
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


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Permanently delete a document:
    1. Remove all its chunks from the vector store.
    2. Delete the source file from the uploads directory.

    Returns the number of chunks removed and whether the file was deleted.
    """
    safe_name = Path(filename).name

    # 1. Remove chunks from ChromaDB
    try:
        deleted_chunks = delete_document_chunks(safe_name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove chunks for '{safe_name}': {e}",
        )

    # 2. Remove file from disk (best-effort — not an error if missing)
    file_path = UPLOADS_DIR / safe_name
    file_deleted = False
    if file_path.exists():
        try:
            file_path.unlink()
            file_deleted = True
        except Exception as e:
            # Log but don't fail the request
            print(f"[documents] Warning: could not delete file '{safe_name}': {e}")

    return {
        "status": "deleted",
        "filename": safe_name,
        "chunks_removed": deleted_chunks,
        "file_deleted": file_deleted,
    }
