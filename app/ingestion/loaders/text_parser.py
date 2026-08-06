# app/ingestion/loaders/text_parser.py
import hashlib
import re
from pathlib import Path


def _sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# Legal keyword heading detector for plain text
_LEGAL_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*[\.\)]\s+[A-Z]|"            # numbered: 1. Clause
    r"(?:Article|Section|Schedule|Clause|Part)\s+\w+|"  # keyword headings
    r"[A-Z][A-Z\s]{4,}$)",                          # ALL CAPS
    re.MULTILINE,
)


def extract_sections_from_txt(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    source_name = Path(file_path).name
    file_hash = _sha256(file_path)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    if not paragraphs:
        return []

    sections = []
    for para in paragraphs:
        lines = para.split("\n")
        first_line = lines[0].strip()

        # Heading: first line is short, or matches legal heading pattern
        if len(first_line) < 120 or _LEGAL_HEADING.match(first_line):
            heading = first_line
        else:
            heading = "Text Section"

        sections.append({
            "heading": heading,
            "text": para,
            "source": source_name,
            "page": 1,
            "file_type": "txt",
            "file_hash": file_hash,
        })

    return sections
