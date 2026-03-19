# app/retrieval/vector_store.py

import chromadb
from chromadb.config import Settings
from app.embedding.embedder import model

client = chromadb.PersistentClient(path="db")


collection = client.get_or_create_collection(name="documents")


def store_embeddings(chunks, embeddings):
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "heading": chunk["heading"],
            "source": chunk["source"],
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print("Stored chunks:", len(chunks))  # keep this for debug
    


# 🔥 NEW: reranking logic
def rerank_results(results, query: str):
    keywords = query.lower().split()

    reranked = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = dist

        heading = meta["heading"].lower()
        doc_lower = doc.lower()

        # 🔥 Boost important sections
        if "booking" in heading:
            score -= 0.2
        if "objective" in heading:
            score -= 0.1

        # 🔥 Keyword match boost
        if any(k in doc_lower for k in keywords):
            score -= 0.1

        reranked.append((score, doc, meta))

    reranked.sort(key=lambda x: x[0])
    return reranked


# 🔥 UPDATED query function
def query_similar(query: str, n_results: int = 5):
    # Step 1: embed query using SAME model
    query_embedding = model.encode([query])[0]

    # Step 2: raw retrieval
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Step 3: rerank
    reranked = rerank_results(results, query)

    return reranked