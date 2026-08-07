# app/api/routes/documents.py
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.retrieval.vector_store import collection, delete_document_chunks
from app.pipeline.ingest_pipeline import remove_manifest_entries_for_filename
from app.cache import get_query_cache

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
    2. Remove matching ingest-manifest entries (so re-upload re-indexes).
    3. Delete the source file from the uploads directory.
    4. Clear the semantic query cache (answers may reference the doc).

    Returns the number of chunks removed and whether the file was deleted.
    """
    safe_name = Path(filename).name

    # 1. Remove chunks from ChromaDB + BM25
    try:
        deleted_chunks = delete_document_chunks(safe_name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove chunks for '{safe_name}': {e}",
        )

    # 2. Drop stale dedupe hashes so identical re-uploads are not skipped
    manifest_removed = remove_manifest_entries_for_filename(safe_name)

    # 3. Remove file from disk (best-effort — not an error if missing)
    file_path = UPLOADS_DIR / safe_name
    file_deleted = False
    if file_path.exists():
        try:
            file_path.unlink()
            file_deleted = True
        except Exception as e:
            print(f"[documents] Warning: could not delete file '{safe_name}': {e}")

    # 4. Invalidate cached answers that may cite this document
    get_query_cache().clear()

    return {
        "status": "deleted",
        "filename": safe_name,
        "chunks_removed": deleted_chunks,
        "manifest_entries_removed": manifest_removed,
        "file_deleted": file_deleted,
    }
