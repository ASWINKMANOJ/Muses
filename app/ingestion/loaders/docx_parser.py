from docx import Document
from pathlib import Path

def extract_sections_from_docx(file_path: str) -> list[dict]:
    doc = Document(file_path)
    sections = []
    source_name = Path(file_path).name
    
    current_heading = "Document Start"
    current_content = []
    
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
                    "file_type": "docx"
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
            "file_type": "docx"
        })
        
    return sections
