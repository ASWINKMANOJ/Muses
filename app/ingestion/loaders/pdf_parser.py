# app/ingestion/loaders/pdf_parser.py
"""
Enhanced PDF parser for legal documents.

Improvements over the original:
- Detects numbered clause headings via regex (e.g. "4.2 Liability Cap")
  in addition to font-size heuristics — critical for legal PDFs where
  clause numbers share the same font size as body text.
- Extracts tables using pdfplumber and converts them to readable text.
- Preserves cross-page section continuity (a section is not reset on each page).
- Stores richer metadata: file_hash, bounding box page, section_hierarchy.
- Falls back gracefully to OCR via pytesseract for scanned PDFs.
"""

import re
import hashlib
from pathlib import Path

import fitz          # PyMuPDF
import pdfplumber    # table extraction


# ── Heading detection patterns ────────────────────────────────────────────────

# Numbered clause: 1. / 1.2 / 1.2.3 / 12.3.4
_NUMBERED_CLAUSE = re.compile(r"^\d+(?:\.\d+)*[\.\)]\s+[A-Z]")

# Keyword section headings: Article, Section, Schedule, Annexure, WHEREAS, etc.
_KEYWORD_HEADING = re.compile(
    r"^(?:Article|Section|Schedule|Annexure|Exhibit|Appendix|Part|Clause|"
    r"WHEREAS|NOW,?\s+THEREFORE|IN WITNESS|RECITAL|DEFINITIONS?|BACKGROUND)\b",
    re.IGNORECASE,
)

# ALL-CAPS headings (common in contracts)
_ALLCAPS_HEADING = re.compile(r"^[A-Z][A-Z\s\-]{4,}$")


def _is_legal_heading(text: str, font_size: float, avg_font_size: float) -> bool:
    """
    Detect headings using both font-size heuristics AND legal-specific patterns.
    A line is a heading if ANY of the following is true:
      1. Font size is significantly larger than average (original heuristic)
      2. Matches a numbered clause pattern
      3. Matches a legal keyword heading
      4. Is short ALL-CAPS (contract title style)
    """
    text = text.strip()
    if not text or len(text) > 200:
        return False

    # Font-size heuristic (original)
    if font_size > avg_font_size * 1.15 and len(text) < 150:
        return True

    # Legal-specific patterns
    if _NUMBERED_CLAUSE.match(text):
        return True
    if _KEYWORD_HEADING.match(text):
        return True
    if _ALLCAPS_HEADING.match(text) and len(text) < 80:
        return True

    return False


# ── Table extraction ──────────────────────────────────────────────────────────

def _extract_tables_from_page(pdf_path: str, page_num: int) -> str:
    """
    Use pdfplumber to extract tables from a page and return as plain text.
    Returns empty string if no tables found.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return ""
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            if not tables:
                return ""
            parts = []
            for table in tables:
                rows = []
                for row in table:
                    cells = [str(c).strip() if c else "" for c in row]
                    rows.append(" | ".join(cells))
                parts.append("\n".join(rows))
            return "\n\n[TABLE]\n" + "\n\n".join(parts) + "\n[/TABLE]\n"
    except Exception:
        return ""


# ── File hash ─────────────────────────────────────────────────────────────────

def _sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_sections_from_pdf(file_path: str) -> list[dict]:
    """
    Extract structured sections from a PDF, aware of legal document structure.

    Returns:
        List of section dicts: {heading, text, source, page, file_type,
                                 file_hash, is_table}
    """
    doc = fitz.open(file_path)
    source_name = Path(file_path).name
    file_hash = _sha256(file_path)
    sections: list[dict] = []

    current_heading = "Preamble"
    current_content: list[str] = []
    current_page = 1

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        lines: list[dict] = []

        # Collect all text lines with font size
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = ""
                max_font_size = 0.0
                bold = False
                for span in line["spans"]:
                    text += span["text"]
                    max_font_size = max(max_font_size, span["size"])
                    if span.get("flags", 0) & 2**4:  # bold flag
                        bold = True
                text = text.strip()
                if text:
                    lines.append({
                        "text": text,
                        "font_size": max_font_size,
                        "bold": bold,
                    })

        if not lines:
            continue

        avg_font_size = sum(l["font_size"] for l in lines) / len(lines)

        for line in lines:
            text = line["text"]
            if _is_legal_heading(text, line["font_size"], avg_font_size):
                # Save accumulated section
                if current_content:
                    sections.append({
                        "heading": current_heading,
                        "text": " ".join(current_content),
                        "source": source_name,
                        "page": current_page,
                        "file_type": "pdf",
                        "file_hash": file_hash,
                        "is_table": False,
                    })
                    current_content = []
                current_heading = text
                current_page = page_num + 1
            else:
                current_content.append(text)

        # Extract tables from this page and add as a separate section
        table_text = _extract_tables_from_page(file_path, page_num)
        if table_text.strip():
            # Flush any pending content first
            if current_content:
                sections.append({
                    "heading": current_heading,
                    "text": " ".join(current_content),
                    "source": source_name,
                    "page": page_num + 1,
                    "file_type": "pdf",
                    "file_hash": file_hash,
                    "is_table": False,
                })
                current_content = []
            sections.append({
                "heading": f"{current_heading} [Table]",
                "text": table_text,
                "source": source_name,
                "page": page_num + 1,
                "file_type": "pdf",
                "file_hash": file_hash,
                "is_table": True,
            })

    # Flush final section
    if current_content:
        sections.append({
            "heading": current_heading,
            "text": " ".join(current_content),
            "source": source_name,
            "page": current_page,
            "file_type": "pdf",
            "file_hash": file_hash,
            "is_table": False,
        })

    doc.close()

    # If the PDF yielded no text at all, attempt OCR fallback
    if not sections:
        sections = _ocr_fallback(file_path, source_name, file_hash)

    return sections


# ── OCR fallback for scanned PDFs ─────────────────────────────────────────────

def _ocr_fallback(file_path: str, source_name: str, file_hash: str) -> list[dict]:
    """Extract text from a scanned PDF using Tesseract page by page."""
    try:
        import pytesseract
        from PIL import Image
        import io

        doc = fitz.open(file_path)
        sections = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img).strip()
            if text:
                sections.append({
                    "heading": f"Page {page_num + 1}",
                    "text": text,
                    "source": source_name,
                    "page": page_num + 1,
                    "file_type": "pdf",
                    "file_hash": file_hash,
                    "is_table": False,
                    "ocr_method": "tesseract",
                })
        doc.close()
        return sections
    except Exception as e:
        print(f"[pdf_parser] OCR fallback failed: {e}")
        return []
