# app/ingestion/pdf_parser.py
import fitz  # PyMuPDF


def is_heading(line: dict, avg_font_size: float) -> bool:
    """
    Heuristic to detect headings.
    """
    text = line["text"].strip()

    if not text:
        return False

    return (
        line["font_size"] > avg_font_size * 1.2 and  # bigger than normal
        len(text) < 120 and                         # not too long
        not text.endswith(".")                      # headings usually don't end with dot
    )


def extract_sections_from_pdf(file_path: str) -> list[dict]:
    """
    Extract structured sections from PDF using font size heuristics.
    Each section = {heading, content, metadata}
    """
    doc = fitz.open(file_path)
    sections = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        lines = []

        # Extract lines with font sizes
        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                text = ""
                max_font_size = 0

                for span in line["spans"]:
                    text += span["text"]
                    max_font_size = max(max_font_size, span["size"])

                text = text.strip()
                if text:
                    lines.append({
                        "text": text,
                        "font_size": max_font_size
                    })

        if not lines:
            continue

        # Compute average font size (baseline)
        avg_font_size = sum(l["font_size"] for l in lines) / len(lines)

        current_heading = "Unknown Section"
        current_content = []

        for line in lines:
            if is_heading(line, avg_font_size):
                # Save previous section
                if current_content:
                    sections.append({
                        "heading": current_heading,
                        "text": " ".join(current_content),
                        "source": file_path.split("/")[-1],
                        "page": page_num + 1,
                        "file_type": "pdf"
                    })
                    current_content = []

                current_heading = line["text"]
            else:
                current_content.append(line["text"])

        # Save last section
        if current_content:
            sections.append({
                "heading": current_heading,
                "text": " ".join(current_content),
                "source": file_path.split("/")[-1],
                "page": page_num + 1,
                "file_type": "pdf"
            })

    doc.close()
    return sections