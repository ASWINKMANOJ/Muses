# app/ingestion/loaders/docx_parser.py
import hashlib
from docx import Document
from pathlib import Path


def _sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_sections_from_docx(file_path: str) -> list[dict]:
    doc = Document(file_path)
    sections = []
    source_name = Path(file_path).name
    file_hash = _sha256(file_path)

    current_heading = "Document Start"
    current_content: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style_name = para.style.name if para.style else ""
        if style_name.startswith("Heading"):
            if current_content:
                sections.append({
                    "heading": current_heading,
                    "text": " ".join(current_content),
                    "source": source_name,
                    "page": 1,
                    "file_type": "docx",
                    "file_hash": file_hash,
                })
                current_content = []
            current_heading = text
        else:
            current_content.append(text)

    if current_content:
        sections.append({
            "heading": current_heading,
            "text": " ".join(current_content),
            "source": source_name,
            "page": 1,
            "file_type": "docx",
            "file_hash": file_hash,
        })

    return sections
