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
import threading
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

_manifest_lock = threading.Lock()

def _load_manifest_unlocked() -> dict:
    path = Path(settings.ingest_manifest_path)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def _save_manifest_unlocked(manifest: dict) -> None:
    path = Path(settings.ingest_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))

def _load_manifest() -> dict:
    with _manifest_lock:
        return _load_manifest_unlocked()

def _save_manifest(manifest: dict) -> None:
    with _manifest_lock:
        _save_manifest_unlocked(manifest)


def remove_manifest_entries_for_filename(filename: str) -> int:
    """
    Remove all manifest entries whose recorded filename matches.

    Needed so DELETE /documents/{filename} does not leave a stale hash
    that would cause a later re-upload of the same bytes to be skipped.
    Returns the number of manifest keys removed.
    """
    safe_name = Path(filename).name
    with _manifest_lock:
        manifest = _load_manifest_unlocked()
        to_remove = [
            h for h, meta in manifest.items()
            if isinstance(meta, dict) and meta.get("filename") == safe_name
        ]
        for h in to_remove:
            del manifest[h]
        if to_remove:
            _save_manifest_unlocked(manifest)
        return len(to_remove)


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
        cached_name = cached.get("filename", filename)
        # Guard: if vectors were deleted without clearing the manifest, re-ingest.
        from app.retrieval.vector_store import collection as _chroma
        still_indexed = _chroma.get(where={"source": cached_name}, include=[]).get("ids", [])
        if still_indexed:
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
        print(f"[ingest_pipeline] Stale manifest for '{filename}' "
              f"(hash={file_hash[:8]}…, 0 chunks in store). Re-ingesting.")
        remove_manifest_entries_for_filename(cached_name)

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
    with _manifest_lock:
        fresh_manifest = _load_manifest_unlocked()
        fresh_manifest[file_hash] = {
            "filename": filename,
            "chunks": len(chunks),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest_unlocked(fresh_manifest)

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
