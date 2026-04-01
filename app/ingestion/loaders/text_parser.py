from pathlib import Path

def extract_sections_from_txt(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by double newline for simple section logic
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    sections = []
    source_name = Path(file_path).name
    
    if not paragraphs:
        return sections

    for i, para in enumerate(paragraphs):
        lines = para.split("\n")
        heading = "Text Section"
        if len(lines[0]) < 100:
            heading = lines[0].strip()
        
        sections.append({
            "heading": heading,
            "text": para,
            "source": source_name,
            "page": 1,
            "file_type": "txt"
        })
        
    return sections
