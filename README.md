# Muses — Legal-Optimized Multimodal RAG System

**Muses** is an advanced Retrieval-Augmented Generation (RAG) system specifically optimized for ingesting legal documents (contracts, statutes, legal briefs, court judgments) and querying them for fast, precise, and cited responses.

It combines **Dense Semantic Search (Legal-BERT)** with **Sparse Keyword Search (BM25)** via **Reciprocal Rank Fusion (RRF)**, **Hypothetical Document Embeddings (HyDE)**, and a local streaming LLM via **Ollama**.

---

## 🌟 Key Features

- **Legal-Optimized Embeddings**: Uses `nlpaueb/legal-bert-base-uncased` fine-tuned on legal corpora for accurate semantic understanding.
- **Hybrid Retrieval (Dense + BM25)**: Fuses vector search (ChromaDB) and exact term matching (BM25) using Reciprocal Rank Fusion (RRF) to pinpoint exact legal clauses and section numbers.
- **Legal-Aware Chunking**: Preserves full clauses, sub-clauses, and structural boundaries (e.g., `Section 4.2`, `WHEREAS`, `Schedule A`) rather than cutting text mid-sentence.
- **Enhanced Document Parsing**:
  - **PDF**: Automatic detection of numbered legal headings, table extraction via `pdfplumber`, and Tesseract OCR fallback for scanned PDFs.
  - **DOCX / TXT / Images**: Multimodal support with Gemma 3 vision & OCR capabilities.
- **HyDE (Hypothetical Document Embedding)**: Generates hypothetical legal provisions to bridge the vocabulary gap between user questions and formal contract phrasing.
- **Async & Non-Blocking Ingestion**: Background document processing with task status polling (`202 Accepted`).
- **Verbatim Citations & Legal System Prompt**: Instructs the LLM to cite document sources, pages, and clause numbers, quote key passages verbatim, and append mandatory legal disclaimers.
- **SHA256 Deduplication**: Prevents duplicate uploads and re-indexing of identical legal documents.
- **Centralized Configuration**: All parameters fully configurable via environment variables (`.env`).
- **Web UI & REST API**: Dark glassmorphic user interface + FastAPI backend with OpenAPI Swagger documentation.

---

## 🏗️ Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Browser / Client UI    │
                          └────────────┬─────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │  FastAPI Server (server.py)     │
                      └────────────────┬────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
  Ingestion Pipeline                                     Query Pipeline
 ┌──────────────────────┐                               ┌──────────────────────┐
 │ File Upload & Hash   │                               │ User Question        │
 ├──────────────────────┤                               ├──────────────────────┤
 │ Parsers (PDF/DOCX)   │                               │ HyDE Query Expansion │
 ├──────────────────────┤                               ├──────────────────────┤
 │ Legal-Aware Chunker  │                               │ Hybrid Search (RRF)  │
 ├──────────────────────┤                               │ ┌──────────────────┐ │
 │ Embedder (Legal-BERT)│                               │ │ Dense (ChromaDB) │ │
 ├──────────────────────┤                               │ ├──────────────────┤ │
 │ Vector Store & BM25  │                               │ │ Sparse (BM25)    │ │
 └──────────────────────┘                               │ └──────────────────┘ │
                                                        ├──────────────────────┤
                                                        │ LLM Token Stream     │
                                                        │ (Ollama / Gemma 3)   │
                                                        └──────────────────────┘
```

---

## 📋 System Requirements

| Component | Minimum / Recommended |
|---|---|
| **OS** | Linux (Ubuntu/Arch), macOS, or Windows (WSL2 recommended) |
| **Python** | Python 3.10 to 3.12 (Python 3.11+ recommended) |
| **Ollama** | Must be installed & running locally |
| **System Dependencies** | `tesseract-ocr` (required for scanned document / image OCR) |
| **GPU Acceleration** | NVIDIA GPU with CUDA 11.8+ (Optional, highly recommended for faster embeddings & inference) |

---

## 🚀 Setup & Installation Guide

Follow these steps to set up and run Muses on any system.

### Step 1: Install System Dependencies

#### **Linux (Arch Linux)**
```bash
sudo pacman -S tesseract tesseract-data-eng
```

#### **Linux (Ubuntu / Debian)**
```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng
```

#### **macOS (via Homebrew)**
```bash
brew install tesseract
```

---

### Step 2: Install & Start Ollama

Muses relies on **Ollama** for running local LLMs (e.g., `gemma3:4b` or `llama3`).

1. **Install Ollama**:
   - **Linux / macOS**:
     ```bash
     curl -fsSL https://ollama.ai/install.sh | sh
     ```
   - **Windows**: Download installer from [ollama.ai](https://ollama.ai).

2. **Pull the Required Model**:
   ```bash
   ollama pull gemma3:4b
   ```

3. **Start the Ollama Service**:
   ```bash
   ollama serve
   ```
   *(Keep this running in a separate terminal window, or start it via systemd: `sudo systemctl enable --now ollama`)*

---

### Step 3: Clone Repository & Create Virtual Environment

```bash
# Clone the repository
git clone https://github.com/ASWINKMANOJ/Muses.git
cd Muses

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux / macOS (bash/zsh):
source venv/bin/activate

# On Linux / macOS (fish shell):
source venv/bin/activate.fish

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

---

### Step 4: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirement.txt
```

*(Optional: For PyTorch CUDA support on Linux/Windows, install PyTorch with CUDA explicitly)*:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 5: Environment Configuration

Copy the sample environment file to `.env`:

```bash
cp .env.example .env
```

You can customize options in `.env`:

```ini
# LLM (Ollama) Settings
OLLAMA_URL=http://localhost:11434/api/generate
LLM_MODEL=gemma3:4b
LLM_TEMPERATURE=0.1
LLM_NUM_CTX=4096

# Embedding Model (Default: Legal-BERT)
EMBEDDING_MODEL=nlpaueb/legal-bert-base-uncased
EMBEDDING_BATCH_SIZE=32

# Retrieval & Hybrid Search
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=5
BM25_WEIGHT=0.5
HYDE_ENABLED=true

# Storage Paths
CHROMA_DB_PATH=db
UPLOADS_DIR=uploads
```

---

## 🏃 Running the Application

### Option A: Web UI & Server Mode (Recommended)

Start the FastAPI application server:

```bash
python main.py --serve
```

With custom host and port:
```bash
python main.py --serve --host 127.0.0.1 --port 8000
```

- **Web Dashboard**: Open [http://localhost:8000](http://localhost:8000) in your web browser.
- **Interactive API Docs (Swagger UI)**: Access [http://localhost:8000/docs](http://localhost:8000/docs).

---

### Option B: Interactive CLI Mode

Run Muses directly inside your terminal:

```bash
# Interactive mode (prompts for document path)
python main.py

# Specify document at startup
python main.py /path/to/contract.pdf
```

---

## 🛠️ API Reference

Muses provides a RESTful API for document ingestion, querying, and document retrieval.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ingest` | Upload file(s) for asynchronous ingestion (`multipart/form-data`) |
| `GET` | `/api/ingest/{task_id}/status` | Check status and progress of an ingestion task |
| `POST` | `/api/chat` | Stream an answer for a query via Server-Sent Events (SSE) |
| `GET` | `/api/documents` | List all ingested documents, page count, and chunk stats |
| `GET` | `/api/documents/{filename}/download` | Download an uploaded document |
| `DELETE` | `/api/documents/{filename}` | Delete a document and purge its vector/BM25 embeddings |

---

### API Usage Examples

#### 1. Ingest a Legal Document (cURL)
```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "files=@/path/to/contract.pdf"
```
**Response:**
```json
{
  "tasks": [
    {
      "task_id": "8f3b2d10-4c11-4f90-a612-c2149b10a202",
      "filename": "contract.pdf",
      "status": "queued"
    }
  ]
}
```

#### 2. Check Ingestion Task Status
```bash
curl http://localhost:8000/api/ingest/8f3b2d10-4c11-4f90-a612-c2149b10a202/status
```

#### 3. Query via SSE Stream (cURL)
```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the liability cap under Section 4?",
    "document_filter": ["contract.pdf"]
  }'
```

---

## 🧪 Testing & Verification

Run diagnostic scripts to verify system setup and component integration:

```bash
# Run environment setup check
python test_setup.py

# Run component-specific tests
python test_parser.py
python test_embedder.py
python test_retrieval.py
python test_rag.py
```

---

## ❓ Troubleshooting

| Issue | Cause & Solution |
|---|---|
| **`ConnectionError: Cannot connect to Ollama`** | Ollama is not running. Run `ollama serve` in a terminal window. |
| **`Model 'gemma3:4b' not found`** | Pull the model first by running `ollama pull gemma3:4b`. |
| **`ModuleNotFoundError: No module named '...'`** | Ensure your virtual environment is activated (`source venv/bin/activate`). |
| **`TesseractNotFoundError`** | Install Tesseract OCR on your system (see Step 1). |
| **ChromaDB / Embedding mismatch after config change** | If you change `EMBEDDING_MODEL` in `.env`, purge the database: `rm -rf db/ uploads/manifest.json`. |

---

## 📜 License

[MIT License](LICENSE)
