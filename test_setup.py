# test_setup.py
import torch
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

print("=== MuSeS Setup Verification ===\n")

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check embedding model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
test_vec = model.encode(["test sentence"])
print(f"Embedding model OK — vector size: {len(test_vec[0])}")

# Check ChromaDB
print("\nChecking ChromaDB...")
client = chromadb.PersistentClient(path="./db")
col = client.get_or_create_collection("test")
print("ChromaDB OK")

# Check Ollama
print("\nChecking Ollama...")
response = ollama.chat(
    model="qwen3.5:4b",
    messages=[{"role": "user", "content": "Say OK"}]
)
print(f"Ollama OK — response: {response['message']['content']}")

print("\n✅ All systems go!")