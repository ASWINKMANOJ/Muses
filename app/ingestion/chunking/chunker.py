from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def clean_text(text: str) -> str:
    # Normalize spacing
    text = re.sub(r'\s+', ' ', text)

    # Fix common PDF splits (light touch)
    text = text.replace(" ot", "ot").replace(" at", "at")

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

            chunks.append({
                "id": f"{section['source']}_{global_chunk_index}",
                "text": full_text,
                "heading": section["heading"],
                "source": section["source"],
                "page": section["page"],
                "chunk_index": i,
                "global_chunk_index": global_chunk_index,
                "file_type": section["file_type"]
            })

            global_chunk_index += 1

    return chunks