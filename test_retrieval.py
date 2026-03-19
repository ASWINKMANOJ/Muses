from app.pipeline.ingest_pipeline import ingest_pdf_pipeline
from app.retrieval.vector_store import query_similar

# Step 1
ingest_pdf_pipeline("data/sample.pdf")

# Step 2
query = "What prevents double booking?"

results = query_similar(query)

for score, doc, meta in results:
    print("\n---")
    print("Score:", score)
    print("Heading:", meta["heading"])
    print("Text:", doc[:300])