# app/ingestion/chunking/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_sections(sections, chunk_size=400, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    global_chunk_index = 0

    for section in sections:
        splits = splitter.split_text(section["text"])

        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "heading": section["heading"],
                "source": section["source"],
                "page": section["page"],
                "chunk_index": i,
                "global_chunk_index": global_chunk_index,
                "file_type": section["file_type"]
            })
            global_chunk_index += 1

    return chunks