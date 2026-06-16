from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def clean_text(text: str) -> str:
    # Normalize whitespace while preserving single newlines (important for OCR output)
    text = re.sub(r'[ \t]+', ' ', text)       # collapse horizontal whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)    # collapse excess blank lines
    return text.strip()


def chunk_sections(sections, chunk_size=600, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    global_chunk_index = 0

    for section in sections:
        cleaned_text = clean_text(section["text"])
        splits = splitter.split_text(cleaned_text)

        for i, split in enumerate(splits):
            full_text = f"{section['heading']}\n{split}"

            chunk = {
                "id": f"{section['source']}_{global_chunk_index}",
                "text": full_text,
                "heading": section["heading"],
                "source": section["source"],
                "page": section["page"],
                "chunk_index": i,
                "global_chunk_index": global_chunk_index,
                "file_type": section["file_type"],
            }

            # Forward optional OCR metadata if present (e.g. from image_parser)
            if "ocr_method" in section:
                chunk["ocr_method"] = section["ocr_method"]

            chunks.append(chunk)
            global_chunk_index += 1

    return chunks