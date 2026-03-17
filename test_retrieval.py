from app.pipeline.ingest_pipeline import ingest_pdf_pipeline
from app.retrieval.vector_store import query_similar

# Step 1: ingest
ingest_pdf_pipeline("data/sample.pdf")

# Step 2: query
query = "What is SeatForge?"

results = query_similar(query)

print(results)