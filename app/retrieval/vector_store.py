# app/retrieval/vector_store.py

import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.Client(Settings(
    persist_directory="db",
    anonymized_telemetry=False
))

collection = client.get_or_create_collection(name="documents")


def store_embeddings(chunks: list[dict], embeddings: list[list[float]]):
    """
    Store chunks + embeddings in ChromaDB
    """

    ids = [f"chunk_{i}" for i in range(len(chunks))]
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


def query_similar(query: str, n_results: int = 5):
    """
    Retrieve similar chunks
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return results