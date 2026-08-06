# app/pipeline/ingest_pipeline.py
"""
Ingestion pipeline with deduplication and progress callbacks.

Improvements:
- SHA256 deduplication: if the exact file has been ingested before, skip
  re-processing and return cached result immediately.
- Progress callback: allows the API layer to stream progress to the client.
- Manifest file (JSON): tracks file_hash → {filename, chunks, timestamp}.
- Async-safe: all I/O is synchronous but can be run in a thread pool by the API.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from app.ingestion.loaders.pdf_parser   import extract_sections_from_pdf
from app.ingestion.loaders.text_parser  import extract_sections_from_txt
from app.ingestion.loaders.docx_parser  import extract_sections_from_docx
from app.ingestion.loaders.image_parser import extract_sections_from_image
from app.ingestion.chunking.chunker     import chunk_sections
from app.embedding.embedder             import embed_chunks
from app.retrieval.vector_store         import store_embeddings, delete_document_chunks
from app.core.config                    import settings


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    path = Path(settings.ingest_manifest_path)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_manifest(manifest: dict) -> None:
    path = Path(settings.ingest_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def _file_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def ingest_file_pipeline(
    file_path: str,
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """
    Full ingestion pipeline: File → Sections → Chunks → Embeddings → DB

    Args:
        file_path:   Absolute or relative path to the file.
        progress_cb: Optional callback(stage: str, pct: int) for progress reporting.
                     Called with stages: 'hashing', 'parsing', 'chunking',
                     'embedding', 'storing', 'done'.

    Returns:
        dict with keys: status, chunks, replaced, skipped, file_hash
    """
    def _progress(stage: str, pct: int):
        if progress_cb:
            progress_cb(stage, pct)

    _progress("hashing", 0)

    # ── 1. Hash & deduplication check ────────────────────────────────────────
    file_hash = _file_sha256(file_path)
    filename  = Path(file_path).name
    manifest  = _load_manifest()

    if file_hash in manifest:
        cached = manifest[file_hash]
        print(f"[ingest_pipeline] '{filename}' already ingested "
              f"({cached['chunks']} chunks, hash={file_hash[:8]}…). Skipping.")
        _progress("done", 100)
        return {
            "status": "skipped",
            "message": "File already ingested (identical content).",
            "chunks": cached["chunks"],
            "replaced": 0,
            "skipped": True,
            "file_hash": file_hash,
        }

    ext = os.path.splitext(file_path)[1].lower()

    # ── 2. Parse ──────────────────────────────────────────────────────────────
    _progress("parsing", 10)
    print(f"[ingest_pipeline] Parsing '{filename}' ({ext})…")

    if ext == ".pdf":
        sections = extract_sections_from_pdf(file_path)
    elif ext == ".txt":
        sections = extract_sections_from_txt(file_path)
    elif ext == ".docx":
        sections = extract_sections_from_docx(file_path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        sections = extract_sections_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not sections:
        return {"status": "error", "message": "No content extracted", "chunks": 0,
                "replaced": 0, "skipped": False, "file_hash": file_hash}

    # ── 3. Chunk ──────────────────────────────────────────────────────────────
    _progress("chunking", 25)
    chunks = chunk_sections(sections)
    print(f"[ingest_pipeline] Generated {len(chunks)} chunks.")

    # ── 4. Upsert: delete stale, then store fresh ─────────────────────────────
    _progress("embedding", 40)
    deleted = delete_document_chunks(filename)
    if deleted:
        print(f"[ingest_pipeline] Replaced {deleted} stale chunks for '{filename}'.")

    # ── 5. Embed ──────────────────────────────────────────────────────────────
    embeddings = embed_chunks(chunks)
    _progress("storing", 80)

    # ── 6. Store ──────────────────────────────────────────────────────────────
    store_embeddings(chunks, embeddings)

    # ── 7. Update manifest ────────────────────────────────────────────────────
    manifest[file_hash] = {
        "filename": filename,
        "chunks": len(chunks),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_manifest(manifest)

    _progress("done", 100)
    print(f"[ingest_pipeline] Done — {len(chunks)} chunks stored for '{filename}'.")

    return {
        "status": "success",
        "chunks": len(chunks),
        "replaced": deleted,
        "skipped": False,
        "file_hash": file_hash,
    }


# Backwards-compatibility alias
def ingest_pdf_pipeline(file_path: str) -> dict:
    return ingest_file_pipeline(file_path)
