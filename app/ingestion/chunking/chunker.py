# app/ingestion/chunking/chunker.py
"""
Legal-aware text chunker.

Improvements over the original:
- Larger chunk_size (1000) and overlap (150) to keep legal clauses intact.
- Legal-boundary-aware separators: prefer splitting on clause/section markers
  (e.g. \n1., \nSection, WHEREAS) before falling back to generic whitespace.
- Each chunk carries a full section_path breadcrumb for precise citations.
- Chunk type classification: clause, definition, recital, schedule, general.
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings


# ── Regex patterns ────────────────────────────────────────────────────────────

# Matches numbered clause markers that are good split points, e.g.:
#   \n1.   \n2.1   \n12.3.4   \n(a)   \n(i)
_CLAUSE_BOUNDARY = re.compile(
    r"\n(?=\d+(?:\.\d+)*[\.\s]|\([a-zA-Z0-9]+\)\s)"
)

# Matches legal section-level keywords
_SECTION_KEYWORD = re.compile(
    r"\n(?=(?:Article|Section|Schedule|Annexure|Exhibit|Appendix|Part|Clause|Recital|WHEREAS|NOW,?\s+THEREFORE|IN WITNESS)\b)",
    re.IGNORECASE,
)

# Legal separators ordered from most preferred to least preferred
LEGAL_SEPARATORS = [
    "\n\nArticle ",
    "\n\nSection ",
    "\n\nSchedule ",
    "\n\nAnnexure ",
    "\n\nExhibit ",
    "\n\nAppendix ",
    "\n\nPart ",
    "\n\nClause ",
    "\n\nWHEREAS",
    "\n\nNOW, THEREFORE",
    "\n\nIN WITNESS",
    "\n\n",        # paragraph break
    "\n",
    ". ",
    " ",
]

# ── Chunk type classifier ─────────────────────────────────────────────────────

_DEFINITION_RE = re.compile(
    r"^(?:Definitions?|\"[^\"]+\"|'[^']+'|means|shall mean|is defined)",
    re.IGNORECASE,
)
_RECITAL_RE = re.compile(r"^(?:WHEREAS|RECITAL|BACKGROUND)", re.IGNORECASE)
_SCHEDULE_RE = re.compile(
    r"^(?:Schedule|Annexure|Exhibit|Appendix|Attachment)\b", re.IGNORECASE
)
_CLAUSE_RE = re.compile(r"^\d+(?:\.\d+)*[\.\s]")


def _classify_chunk(heading: str, text: str) -> str:
    combined = (heading + " " + text[:200]).strip()
    if _RECITAL_RE.search(combined):
        return "recital"
    if _SCHEDULE_RE.search(combined):
        return "schedule"
    if _DEFINITION_RE.search(combined):
        return "definition"
    if _CLAUSE_RE.match(combined):
        return "clause"
    return "general"


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize whitespace for legal text while preserving structural cues
    (clause numbering, paragraph breaks) that aid chunking.
    """
    text = re.sub(r"[ \t]+", " ", text)       # collapse horizontal whitespace
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # max 3 consecutive newlines
    # Fix common OCR artefacts in legal docs
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)   # hyphenated line breaks
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)   # spaced decimal numbers
    return text.strip()


# ── Main chunker ──────────────────────────────────────────────────────────────

def chunk_sections(
    sections: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """
    Split a list of section dicts into smaller chunks suitable for embedding.

    Each output chunk dict contains:
        id, text, heading, source, page, chunk_index, global_chunk_index,
        file_type, section_path, chunk_type, clause_number
    Plus optional: ocr_method (from image sections)

    Args:
        sections:      Output of any loader (pdf_parser, docx_parser, etc.)
        chunk_size:    Override settings.chunk_size
        chunk_overlap: Override settings.chunk_overlap
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=LEGAL_SEPARATORS,
        keep_separator=True,   # keep the separator in the chunk so clause numbers aren't lost
    )

    chunks: list[dict] = []
    global_chunk_index = 0

    for section in sections:
        cleaned = clean_text(section["text"])
        splits = splitter.split_text(cleaned)

        # Build section breadcrumb path for citation
        section_path = _build_section_path(section)

        for i, split in enumerate(splits):
            # Prepend heading so each chunk is self-contained
            full_text = f"{section['heading']}\n{split}"
            chunk_type = _classify_chunk(section["heading"], split)
            clause_number = _extract_clause_number(section["heading"])

            chunk: dict = {
                "id": f"{section['source']}_{global_chunk_index}",
                "text": full_text,
                "heading": section["heading"],
                "source": section["source"],
                "page": section["page"],
                "chunk_index": i,
                "global_chunk_index": global_chunk_index,
                "file_type": section["file_type"],
                "section_path": section_path,
                "chunk_type": chunk_type,
                "clause_number": clause_number,
            }

            # Forward optional OCR metadata
            if "ocr_method" in section:
                chunk["ocr_method"] = section["ocr_method"]
            if "doc_date" in section:
                chunk["doc_date"] = section["doc_date"]

            chunks.append(chunk)
            global_chunk_index += 1

    return chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_section_path(section: dict) -> str:
    """Build a breadcrumb string like 'Part II > Section 4 > Clause 4.2'."""
    parts = []
    for key in ("part", "article", "section", "heading"):
        val = section.get(key, "").strip()
        if val and val not in parts:
            parts.append(val)
    return " > ".join(parts) if parts else section.get("heading", "")


def _extract_clause_number(heading: str) -> str:
    """Extract the leading clause number from a heading, e.g. '4.2.1' from '4.2.1 Liability'."""
    m = re.match(r"^(\d+(?:\.\d+)*)", heading.strip())
    return m.group(1) if m else ""
