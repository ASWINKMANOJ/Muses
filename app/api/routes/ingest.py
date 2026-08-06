# app/api/routes/ingest.py
"""
Ingest API route — runs ingestion off the event loop via asyncio.to_thread.
Returns a JSON payload with a `results` array compatible with frontend app.js.
"""

import asyncio
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.pipeline.ingest_pipeline import ingest_file_pipeline
from app.core.config import settings

router = APIRouter()

UPLOADS_DIR = Path(settings.uploads_dir)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_MB = 100


async def _ingest_single_file(upload: UploadFile) -> dict:
    """Save upload to disk and execute ingestion in a thread pool."""
    ext = Path(upload.filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return {
            "filename": upload.filename,
            "status": "error",
            "message": f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            "chunks": 0,
            "replaced": 0,
        }

    dest_path = UPLOADS_DIR / upload.filename

    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
    except Exception as e:
        return {
            "filename": upload.filename,
            "status": "error",
            "message": f"Failed to save file: {e}",
            "chunks": 0,
            "replaced": 0,
        }
    finally:
        await upload.close()

    # File size validation
    size_mb = dest_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        dest_path.unlink(missing_ok=True)
        return {
            "filename": upload.filename,
            "status": "error",
            "message": f"File too large ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB limit).",
            "chunks": 0,
            "replaced": 0,
        }

    # Run ingestion pipeline in thread pool to prevent blocking event loop
    try:
        res = await asyncio.to_thread(ingest_file_pipeline, str(dest_path))
        return {
            "filename": upload.filename,
            "status": "success" if res.get("status") in ("success", "skipped") else "error",
            "chunks": res.get("chunks", 0),
            "replaced": res.get("replaced", 0),
            "message": res.get("message", ""),
        }
    except Exception as e:
        return {
            "filename": upload.filename,
            "status": "error",
            "message": str(e),
            "chunks": 0,
            "replaced": 0,
        }


@router.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents and ingest them into the RAG pipeline.
    Runs off the event loop via asyncio.to_thread.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Process all files concurrently off the event loop
    results = await asyncio.gather(*[_ingest_single_file(f) for f in files])

    return JSONResponse(content={"results": results})
