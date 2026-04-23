# app/api/routes/ingest.py
import os
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.pipeline.ingest_pipeline import ingest_file_pipeline

router = APIRouter()

UPLOADS_DIR = Path("uploads")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"}


@router.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents and ingest them into the RAG pipeline.
    Accepts: PDF, TXT, DOCX, PNG, JPG, JPEG
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for upload in files:
        ext = Path(upload.filename).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            results.append({
                "filename": upload.filename,
                "status": "error",
                "message": f"Unsupported file type '{ext}'. "
                           f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                "chunks": 0,
            })
            continue

        dest_path = UPLOADS_DIR / upload.filename

        # Save uploaded file to disk
        try:
            with dest_path.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
        except Exception as e:
            results.append({
                "filename": upload.filename,
                "status": "error",
                "message": f"Failed to save file: {e}",
                "chunks": 0,
            })
            continue
        finally:
            await upload.close()

        # Run ingestion pipeline
        try:
            result = ingest_file_pipeline(str(dest_path))
            results.append({
                "filename": upload.filename,
                "status": result.get("status", "success"),
                "chunks": result.get("chunks", 0),
                "message": result.get("message", ""),
            })
        except Exception as e:
            results.append({
                "filename": upload.filename,
                "status": "error",
                "message": str(e),
                "chunks": 0,
            })

    return JSONResponse(content={"results": results})
