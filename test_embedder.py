from app.ingestion.loaders.pdf_parser import extract_sections_from_pdf
from app.ingestion.chunking.chunker import chunk_sections
from app.embedding.embedder import embed_chunks

sections = extract_sections_from_pdf("data/sample.pdf")
chunks = chunk_sections(sections)

embeddings = embed_chunks(chunks)

print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))