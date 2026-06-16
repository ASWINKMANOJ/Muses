"""
test_image_chunking.py
──────────────────────
Diagnostic script: runs the full image chunking pipeline on YOUR images
    Image → extract_sections_from_image → chunk_sections → embed_chunks

USAGE
─────
  # Test with one or more of your own images:
  python test_image_chunking.py path/to/photo.png
  python test_image_chunking.py img1.jpg img2.png uploads/scan.jpeg

  # Run without arguments → uses a built-in synthetic test image:
  python test_image_chunking.py

OPTIONS
  --no-embed      Skip the embedding step (faster, no model download needed)
  --chunk-size N  Override chunk size  (default: 600)
  --overlap N     Override chunk overlap (default: 80)

Run from project root:
  python test_image_chunking.py [image_path ...] [--no-embed] [--chunk-size N] [--overlap N]
"""

import sys
import os
import textwrap
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── project root on path ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion.loaders.image_parser import extract_sections_from_image
from app.ingestion.chunking.chunker import chunk_sections

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

DIVIDER   = "═" * 72
THIN_LINE = "─" * 72
SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def log_header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def log_section_block(label: str, value: str = "") -> None:
    print(f"\n{THIN_LINE}")
    print(f"  ▶ {label}")
    if value:
        print(THIN_LINE)
        print(textwrap.indent(value, "    "))


def log_chunk(idx: int, chunk: dict) -> None:
    print(f"\n  ┌── Chunk #{idx}  (global_idx={chunk['global_chunk_index']}, chunk_idx={chunk['chunk_index']})")
    print(f"  │  id       : {chunk['id']}")
    print(f"  │  source   : {chunk['source']}")
    print(f"  │  heading  : {chunk['heading']}")
    print(f"  │  page     : {chunk['page']}")
    print(f"  │  type     : {chunk['file_type']}")
    print(f"  │  length   : {len(chunk['text'])} chars")
    preview = chunk["text"][:220].replace("\n", "↵")
    print(f"  │  preview  : {preview!r}")
    print(f"  └{'─' * 68}")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC FALLBACK IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def create_synthetic_image(path: str) -> None:
    """Creates a 900×700 white PNG with legible sample text for Tesseract."""
    W, H = 900, 700
    img  = Image.new("RGB", (W, H), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_body  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font_title = ImageFont.load_default()
        font_body  = font_title

    lines = [
        ("Muses — Multimodal RAG System",                                        font_title, 40,  40),
        ("Introduction",                                                           font_title, 40, 110),
        ("Muses is a retrieval-augmented generation system that processes",        font_body,  40, 155),
        ("multiple document types: PDF, DOCX, plain text, and images.",           font_body,  40, 185),
        ("Architecture",                                                           font_title, 40, 230),
        ("The ingestion pipeline extracts sections from raw files,",               font_body,  40, 275),
        ("splits them into overlapping chunks, embeds each chunk with",            font_body,  40, 305),
        ("sentence-transformers (all-MiniLM-L6-v2), and stores vectors",          font_body,  40, 335),
        ("in a ChromaDB collection for fast approximate-nearest-neighbour",        font_body,  40, 365),
        ("retrieval at query time.",                                                font_body,  40, 395),
        ("Image Chunking",                                                         font_title, 40, 440),
        ("Images are processed by Tesseract OCR (pytesseract). The",              font_body,  40, 485),
        ("extracted text is treated as a single section and then chunked",         font_body,  40, 515),
        ("by RecursiveCharacterTextSplitter with chunk_size=600 and",             font_body,  40, 545),
        ("chunk_overlap=80, matching the behaviour of other file types.",          font_body,  40, 575),
        ("Page 1 of 1",                                                            font_body,  40, 650),
    ]
    for text, font, x, y in lines:
        draw.text((x, y), text, fill="black", font=font)

    img.save(path)
    print(f"  [synthetic] Test image saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-IMAGE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(image_path: str, chunk_size: int, overlap: int, embed: bool) -> dict:
    """Run the full diagnostic pipeline for one image. Returns a result summary dict."""

    log_header(f"IMAGE  →  {image_path}")

    # ── Image info ─────────────────────────────────────────────────────────────
    try:
        with Image.open(image_path) as img:
            print(f"\n  Format  : {img.format or Path(image_path).suffix.upper()}")
            print(f"  Mode    : {img.mode}")
            print(f"  Size    : {img.size[0]} × {img.size[1]} px")
            file_kb = os.path.getsize(image_path) / 1024
            print(f"  File    : {file_kb:.1f} KB")
    except Exception as e:
        print(f"\n  ⚠  Could not open image for info: {e}")

    # ── STEP 1: Extract ────────────────────────────────────────────────────────
    log_header("STEP 1 — extract_sections_from_image()")
    sections = extract_sections_from_image(image_path)

    print(f"\n  Sections returned : {len(sections)}")
    for idx, sec in enumerate(sections):
        log_section_block(
            f"Section [{idx}] | heading={sec['heading']!r} | "
            f"page={sec['page']} | file_type={sec['file_type']}",
            sec["text"],
        )
        char_count = len(sec["text"])
        print(f"\n  Total characters extracted : {char_count}")
        if char_count == 0 or sec["text"] == "No text found in image.":
            print("  ⚠  OCR returned no text — check image quality / Tesseract install.")

    if not sections:
        print("  ❌  No sections returned.")
        return {"status": "error", "image": image_path, "sections": 0, "chunks": 0}

    # ── STEP 2: Chunk ──────────────────────────────────────────────────────────
    log_header(f"STEP 2 — chunk_sections()  [chunk_size={chunk_size}, overlap={overlap}]")
    chunks = chunk_sections(sections, chunk_size=chunk_size, chunk_overlap=overlap)

    print(f"\n  Total chunks produced : {len(chunks)}")
    for idx, chunk in enumerate(chunks):
        log_chunk(idx, chunk)

    # ── STEP 3: Summary ────────────────────────────────────────────────────────
    log_header("STEP 3 — Summary")
    total_chars    = sum(len(c["text"]) for c in chunks)
    unique_sources = {c["source"] for c in chunks}
    unique_types   = {c["file_type"] for c in chunks}
    print(f"""
  Image path      : {image_path}
  Sections        : {len(sections)}
  Chunks          : {len(chunks)}
  Total chars     : {total_chars}
  Unique sources  : {unique_sources}
  File types      : {unique_types}
""")

    if len(chunks) == 0:
        print("  ⚠  Zero chunks — OCR may have returned empty text.")
        ok = False
    elif all(c["file_type"] == "image" for c in chunks):
        print("  ✅  All chunks correctly tagged as file_type='image'.")
        ok = True
    else:
        print("  ⚠  Some chunks have unexpected file_type values.")
        ok = False

    # ── STEP 4: Embed (optional) ───────────────────────────────────────────────
    embed_ok = None
    if embed:
        log_header("STEP 4 — embed_chunks() smoke-test (no DB write)")
        try:
            from app.embedding.embedder import embed_chunks
            print("\n  Loading sentence-transformer model…")
            embeddings = embed_chunks(chunks)
            print(f"\n  Embeddings shape : {len(embeddings)} × {len(embeddings[0])}")
            print(f"  First 5 values   : {[float(f'{v:.6f}') for v in embeddings[0][:5]]}")
            print("\n  ✅  Embedding step OK.")
            embed_ok = True
        except Exception as e:
            print(f"\n  ⚠  Embedding failed: {e}")
            embed_ok = False

    return {
        "status": "ok" if ok else "warn",
        "image": image_path,
        "sections": len(sections),
        "chunks": len(chunks),
        "total_chars": total_chars,
        "embed_ok": embed_ok,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-IMAGE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_final_report(results: list[dict]) -> None:
    log_header("FINAL REPORT — All Images")
    col = 38
    print(f"\n  {'Image':<{col}} {'Sections':>9} {'Chunks':>7} {'Chars':>7} {'Embed':>6} {'Status':>7}")
    print(f"  {'─'*col} {'─'*9} {'─'*7} {'─'*7} {'─'*6} {'─'*7}")
    all_ok = True
    for r in results:
        name   = Path(r["image"]).name
        embed  = {"True": "✅", "False": "❌", "None": "—"}.get(str(r["embed_ok"]), "—")
        status = "✅ OK" if r["status"] == "ok" else ("⚠ WARN" if r["status"] == "warn" else "❌ ERR")
        if r["status"] != "ok":
            all_ok = False
        print(f"  {name:<{col}} {r['sections']:>9} {r['chunks']:>7} {r.get('total_chars', 0):>7} {embed:>6} {status:>7}")
    print()
    if all_ok:
        print("  ✅  All images passed the chunking pipeline.")
    else:
        print("  ⚠  Some images had warnings — check per-image logs above.")
    print(f"\n{DIVIDER}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING & ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the Muses image chunking pipeline on your own images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_image_chunking.py                            # synthetic image
  python test_image_chunking.py photo.png                 # single image
  python test_image_chunking.py img1.jpg img2.png         # multiple images
  python test_image_chunking.py scan.jpg --no-embed       # skip embedding
  python test_image_chunking.py doc.png --chunk-size 400 --overlap 50
        """,
    )
    parser.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help="Path(s) to image file(s). Supported: .png .jpg .jpeg .bmp .tiff .webp",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip the embedding step (faster, no model download).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        metavar="N",
        help="Chunk size in characters (default: 600).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=80,
        metavar="N",
        help="Chunk overlap in characters (default: 80).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embed = not args.no_embed

    # ── Resolve image paths ────────────────────────────────────────────────────
    image_paths: list[str] = []

    if not args.images:
        # No images supplied → create and use a synthetic one
        log_header("No images specified — using built-in synthetic test image")
        synthetic = "/tmp/muses_test_image.png"
        create_synthetic_image(synthetic)
        image_paths = [synthetic]
    else:
        for raw in args.images:
            p = Path(raw)
            if not p.exists():
                print(f"  ❌  File not found: {raw}")
                continue
            if p.suffix.lower() not in SUPPORTED:
                print(f"  ⚠  Skipping unsupported format: {raw}  (supported: {', '.join(SUPPORTED)})")
                continue
            image_paths.append(str(p.resolve()))

        if not image_paths:
            print("\n  ❌  No valid image files found. Exiting.")
            sys.exit(1)

    print(f"\n  Images to process : {len(image_paths)}")
    print(f"  Chunk size        : {args.chunk_size}")
    print(f"  Chunk overlap     : {args.overlap}")
    print(f"  Embedding         : {'enabled' if embed else 'disabled (--no-embed)'}")

    # ── Run pipeline for each image ────────────────────────────────────────────
    results = []
    for path in image_paths:
        result = run_pipeline(path, args.chunk_size, args.overlap, embed)
        results.append(result)

    # ── Final multi-image report ───────────────────────────────────────────────
    if len(results) > 1:
        print_final_report(results)
    else:
        print(f"\n{DIVIDER}")
        print("  ✅  Image chunking diagnostic complete.")
        print(DIVIDER + "\n")


if __name__ == "__main__":
    main()
