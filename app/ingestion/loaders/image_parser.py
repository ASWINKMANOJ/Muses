"""
image_parser.py
───────────────
Extracts text from images using Gemma 3 vision (via local Ollama).
Falls back to Tesseract OCR if Ollama is unreachable.

Gemma 3 handles:
  • Handwritten notes from phone photos
  • Skewed / unevenly-lit paper
  • Mixed printed + handwritten content
"""

import base64
import io
import requests
from pathlib import Path
from PIL import Image as PilImage

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
VISION_MODEL = "gemma3:4b"       # change to "llava" etc. if you prefer
OLLAMA_TIMEOUT = 300             # seconds — large images on slow GPU may need more
MAX_IMAGE_PX   = 1024            # downscale to this max dimension before sending

OCR_PROMPT = (
    "You are an OCR assistant. Your ONLY job is to transcribe every word "
    "visible in this image exactly as written — including headings, bullet points, "
    "numbering, and any symbols. "
    "Preserve the original line structure. "
    "Do NOT add explanations, summaries, or formatting that isn't in the image. "
    "If a word is unclear, write your best guess followed by [?]. "
    "Output the raw transcribed text only."
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _image_to_base64(file_path: str) -> str:
    """
    Load the image, downscale to MAX_IMAGE_PX on the longest side (preserving
    aspect ratio), then return as a JPEG base64 string.
    Phone photos at 4K+ resolution waste tokens and slow inference significantly;
    1024px retains all readable text for a vision model.
    """
    with PilImage.open(file_path) as img:
        img = img.convert("RGB")   # normalise mode (e.g. RGBA PNG → RGB)
        w, h = img.size
        longest = max(w, h)
        if longest > MAX_IMAGE_PX:
            scale = MAX_IMAGE_PX / longest
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, PilImage.LANCZOS)
            print(f"[image_parser] Resized {w}×{h} → {new_size[0]}×{new_size[1]} for vision inference.")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_via_gemma(file_path: str) -> str:
    """Send image to Gemma 3 vision via Ollama and return the transcribed text."""
    b64 = _image_to_base64(file_path)

    payload = {
        "model": VISION_MODEL,
        "prompt": OCR_PROMPT,
        "images": [b64],
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temp = faithful transcription
            "num_predict": 2048,  # allow long notes
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def _extract_via_tesseract(file_path: str) -> str:
    """Fallback: Tesseract OCR (works well only on clean printed text)."""
    try:
        import pytesseract
        from PIL import Image
        image = Image.open(file_path)
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"[Tesseract fallback failed: {e}]"


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_sections_from_image(file_path: str) -> list[dict]:
    """
    Extract text from an image file.

    Strategy:
      1. Try Gemma 3 vision via Ollama (best for handwriting / phone photos).
      2. If Ollama is unreachable, fall back to Tesseract.

    Returns a list with a single section dict compatible with chunk_sections().
    """
    source_name = Path(file_path).name
    method_used = "gemma3-vision"

    print(f"[image_parser] Processing '{source_name}' with {VISION_MODEL} vision…")

    try:
        extracted_text = _extract_via_gemma(file_path)
        if not extracted_text:
            extracted_text = "No text detected by vision model."
    except requests.exceptions.ConnectionError:
        print(f"[image_parser] ⚠  Ollama unreachable — falling back to Tesseract.")
        extracted_text = _extract_via_tesseract(file_path)
        method_used = "tesseract-fallback"
    except Exception as e:
        print(f"[image_parser] ⚠  Gemma vision error: {e} — falling back to Tesseract.")
        extracted_text = _extract_via_tesseract(file_path)
        method_used = "tesseract-fallback"

    print(f"[image_parser] ✅  Extracted {len(extracted_text)} chars via {method_used}.")

    return [{
        "heading": "Image Content",
        "text":    extracted_text,
        "source":  source_name,
        "page":    1,
        "file_type": "image",
        "ocr_method": method_used,   # bonus metadata for debugging
    }]
