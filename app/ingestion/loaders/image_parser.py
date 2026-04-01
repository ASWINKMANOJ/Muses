import pytesseract
from PIL import Image
from pathlib import Path

def extract_sections_from_image(file_path: str) -> list[dict]:
    source_name = Path(file_path).name
    
    try:
        image = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(image).strip()
    except Exception as e:
        extracted_text = f"Error extracting text from image: {e}"
        
    if not extracted_text:
        extracted_text = "No text found in image."
        
    return [{
        "heading": "Image Content",
        "text": extracted_text,
        "source": source_name,
        "page": 1,
        "file_type": "image"
    }]
