# Muses

**Muses** is a local, legal-oriented multimodal RAG (Retrieval-Augmented Generation) system. Upload contracts, statutes, briefs, or scanned pages; ask questions in natural language; get streamed answers with document/page/clause citations — all running on your machine via Ollama.

---

## Features

| Area | What you get |
|------|----------------|
| **Hybrid retrieval** | Dense vectors (ChromaDB) + sparse BM25, fused with Reciprocal Rank Fusion (RRF) |
| **Embeddings** | `BAAI/bge-base-en-v1.5` bi-encoder (CPU by default so GPU stays free for the LLM) |
| **Reranking** | Optional cross-encoder (`ms-marco-MiniLM-L-6-v2`) for top-k precision |
| **HyDE** | Optional hypothetical-answer expansion to bridge user wording and legal prose |
| **CRAG guardrail** | On low rerank confidence: corrective re-retrieve, then refuse instead of guessing |
| **Semantic cache** | Near-duplicate questions reuse answers; scoped by `document_filter` |
| **Legal-aware chunking** | Prefers clause/section boundaries (`Article`, `Section`, `WHEREAS`, numbered clauses) |
| **Multimodal ingest** | PDF (text + tables + OCR), DOCX, TXT, PNG/JPG (vision OCR via Ollama / Tesseract) |
| **Dedup & upsert** | SHA256 manifest skip for identical bytes; same filename replaces old chunks |
| **Citations** | System prompt requires inline `[Doc / Page / Section]` cites, verbatim quotes, disclaimer |
| **Eval & cache APIs** | Benchmark Hit@K / MRR; inspect or clear the query cache |
| **UI + API** | Static web frontend + FastAPI (Swagger at `/docs`) + interactive CLI |

---

## Architecture

```
Browser / CLI / curl
        │
        ▼
┌───────────────────────────────────────────────┐
│  FastAPI  (python main.py --serve)            │
│  /api/ingest  /api/chat  /api/documents       │
│  /api/eval/*  /api/cache/*                    │
└───────────────────┬───────────────────────────┘
                    │
     ┌──────────────┴──────────────┐
     ▼                             ▼
 Ingest pipeline              Query pipeline
 Parse → Chunk → Embed        Cache? → HyDE? → Hybrid RRF
      → Chroma + BM25         → Cross-encoder → CRAG
                              → Stream Ollama answer → Cache
```

**Supported uploads:** `.pdf` `.docx` `.txt` `.png` `.jpg` `.jpeg` (max 100 MB per file via API)

---

## Requirements

| Component | Notes |
|-----------|--------|
| **OS** | Linux, macOS, or Windows (WSL2 recommended on Windows) |
| **Python** | 3.10–3.12 recommended (3.11 is a safe default) |
| **Ollama** | Installed and running; model pulled (default `gemma3:4b`) |
| **Tesseract OCR** | Needed for scanned PDFs / image fallback |
| **RAM** | **16 GB comfortable.** 8 GB works with the low-memory profile below |
| **GPU** | Optional. NVIDIA GPU with ~4 GB+ VRAM can run `gemma3:4b` in Ollama. Embeddings stay on CPU by default |

### Low-end laptop (example: 8 GB RAM + RTX 2050 4 GB)

It runs, but you should use the lighter profile in [Hardware tuning](#hardware-tuning-8-gb-ram--4-gb-vram). Expect swap under load if Chrome/IDE stay open.

---

## Setup on a new machine

### 1. System packages

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng python3-venv python3-pip
```

**Arch / CachyOS**
```bash
sudo pacman -S tesseract tesseract-data-eng
```

**macOS**
```bash
brew install tesseract
```

**Windows**  
Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and ensure it is on `PATH`. Prefer WSL2 for fewer path/CUDA issues.

---

### 2. Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Pull the default chat model
ollama pull gemma3:4b

# Start the server (if not already a service)
ollama serve
```

Keep `ollama serve` running (or enable the systemd/user service). Verify:

```bash
curl http://localhost:11434/api/tags
```

---

### 3. Clone and create a virtualenv

```bash
git clone https://github.com/ASWINKMANOJ/Muses.git
cd Muses

python3 -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows (cmd)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

---

### 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirement.txt
```

**Optional — PyTorch with CUDA** (NVIDIA GPU; versions vary by driver):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Muses still defaults `EMBEDDING_DEVICE=cpu` so the GPU is reserved for Ollama. Only change that if you know VRAM headroom.

First run will download Hugging Face models (`bge-base-en-v1.5`, cross-encoder). Needs network once.

---

### 5. Configure environment

```bash
cp .env.example .env
```

Edit `.env` as needed. Important defaults:

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_MODEL` | `gemma3:4b` | Ollama model name |
| `LLM_NUM_CTX` | `4096` | Context window (lower on 4 GB VRAM) |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence embedding model |
| `EMBEDDING_DEVICE` | `cpu` | Keep `cpu` when Ollama uses the GPU |
| `CHROMA_COLLECTION` | `documents_bge_base` | Change if you switch embedding models |
| `HYDE_ENABLED` | `true` | Extra LLM call before retrieval |
| `CROSS_ENCODER_ENABLED` | `true` | Rerank + CRAG confidence |
| `CRAG_MIN_CONFIDENCE` | `-5.0` | Cross-encoder logit floor |
| `SEMANTIC_CACHE_ENABLED` | `true` | Cache near-duplicate Q&A |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Cosine similarity for cache hit |

**Changing `EMBEDDING_MODEL`:** use a new `CHROMA_COLLECTION` name and re-ingest documents. Old vectors are not compatible.

```bash
# Clean slate after an embedding model change
rm -rf db/ uploads/manifest.json
# Then re-upload / re-ingest your files
```

---

## Run

### Web UI + API (recommended)

```bash
source venv/bin/activate   # if not already
python main.py --serve
```

Custom bind:

```bash
python main.py --serve --host 127.0.0.1 --port 8000
```

| URL | Purpose |
|-----|---------|
| http://localhost:8000 | Web UI |
| http://localhost:8000/docs | Swagger / OpenAPI |

### Interactive CLI

```bash
python main.py
# or ingest then chat:
python main.py /path/to/contract.pdf
```

Commands in the CLI: type a question, `ingest` to add another file, `quit` to exit.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/ingest` | Upload one or more files (`multipart/form-data`, field `files`) |
| `POST` | `/api/chat` | SSE stream answer (`query`, optional `document_filter`) |
| `GET` | `/api/documents` | List ingested sources and chunk counts |
| `GET` | `/api/documents/{filename}/download` | Download original upload |
| `DELETE` | `/api/documents/{filename}` | Remove vectors, BM25, manifest entry, file; clear cache |
| `GET` | `/api/eval/benchmark` | Run retrieval benchmark (`?top_k=5`) |
| `GET` | `/api/cache/stats` | Semantic cache stats |
| `POST` | `/api/cache/clear` | Clear semantic cache |

### Ingest

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "files=@/path/to/contract.pdf"
```

Example response:

```json
{
  "results": [
    {
      "filename": "contract.pdf",
      "status": "success",
      "chunks": 42,
      "replaced": 0,
      "message": ""
    }
  ]
}
```

`status` may be `success`, `skipped` (identical SHA256 already indexed), or `error`.

### Chat (SSE)

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the liability cap under Section 4?",
    "document_filter": ["contract.pdf"]
  }'
```

Events look like:

```text
data: {"token": "The "}
data: {"token": "liability "}
...
data: {"done": true}
```

Omit `document_filter` (or pass `[]`) to search all documents. Cache entries are scoped per filter, so scoped and global answers do not collide.

### Delete a document

```bash
curl -X DELETE http://localhost:8000/api/documents/contract.pdf
```

Removes Chroma chunks, BM25 entries, ingest-manifest hashes (so the same file can be re-uploaded), the file under `uploads/`, and clears the semantic cache.

---

## Query pipeline (what happens on each question)

1. **Semantic cache** — if a similar query exists for the same document scope (cosine ≥ threshold), return cached answer.
2. **HyDE** (optional) — LLM writes a short hypothetical provision; its embedding is averaged with the query embedding.
3. **Hybrid retrieval** — dense (BGE + Chroma) + BM25 → RRF → optional cross-encoder rerank → top-k.
4. **CRAG** — if best cross-encoder score &lt; `CRAG_MIN_CONFIDENCE`: re-retrieve without HyDE and with a wider candidate pool; if still low, **refuse** instead of answering from weak context.
5. **Generate** — stream from Ollama with legal citation system prompt; cache the full answer when confidence is acceptable.

---

## Evaluation

CLI:

```bash
python test_eval.py --top-k 5 --output eval_report.json
```

Or:

```bash
curl "http://localhost:8000/api/eval/benchmark?top_k=5"
```

Metrics: Hit Rate @ K, MRR, average retrieval score, latency.

Optional labeled set: create `tests/eval_dataset.json`:

```json
[
  {
    "query": "What is the liability cap under Section 4?",
    "expected_keywords": ["section 4", "liability", "contract.pdf"],
    "category": "contract_clause"
  }
]
```

Keywords are matched against retrieved chunk **metadata** (source / section / clause). Without a dataset file, a small built-in fallback set is used.

Other smoke scripts:

```bash
python test_setup.py
python test_parser.py
python test_embedder.py
python test_retrieval.py
python test_rag.py
```

---

## Hardware tuning (8 GB RAM / 4 GB VRAM)

Put these in `.env` (then re-ingest if you change the embedding model or collection):

```ini
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=8
CHROMA_COLLECTION=documents_bge_small

LLM_MODEL=gemma3:4b
LLM_NUM_CTX=2048
LLM_NUM_PREDICT=512
LLM_NUM_GPU=99

HYDE_ENABLED=false
CROSS_ENCODER_ENABLED=true
SEMANTIC_CACHE_ENABLED=true
```

Tips:

- Close heavy browser tabs while ingesting or querying.
- Keep embeddings on **CPU**; let Ollama use the GPU.
- After switching to `bge-small`, wipe `db/` and `uploads/manifest.json` and re-ingest.

---

## Project layout

```text
Muses/
├── main.py                 # CLI + --serve entrypoint
├── requirement.txt
├── .env.example
├── frontend/               # Static web UI
├── app/
│   ├── api/                # FastAPI app + routes
│   ├── cache/              # Semantic query cache
│   ├── core/config.py      # Settings from env
│   ├── embedding/          # SentenceTransformer wrapper
│   ├── evaluation/         # Hit@K / MRR benchmark
│   ├── generation/llm.py   # Ollama + legal system prompt + HyDE
│   ├── ingestion/          # Parsers + legal chunker
│   ├── pipeline/           # Ingest + query pipelines
│   └── retrieval/          # Chroma + BM25 + RRF + rerank
├── db/                     # Chroma + BM25 index (created at runtime)
└── uploads/                # Uploaded files + manifest.json
```

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| Cannot connect to Ollama | Run `ollama serve`; check `OLLAMA_URL` in `.env` |
| Model not found | `ollama pull gemma3:4b` (or whatever `LLM_MODEL` is) |
| CUDA / VRAM OOM during chat | Lower `LLM_NUM_CTX` to `2048`; ensure `EMBEDDING_DEVICE=cpu` |
| System RAM thrashing / swap | Use `bge-small`, `EMBEDDING_BATCH_SIZE=8`, disable HyDE; free RAM |
| Embedding / Chroma mismatch | New collection name + `rm -rf db uploads/manifest.json` + re-ingest |
| Re-upload of same file skipped wrongly | Delete via API (clears manifest); or remove that hash from `uploads/manifest.json` |
| `TesseractNotFoundError` | Install Tesseract and ensure it is on `PATH` |
| Import errors | Activate `venv` and re-run `pip install -r requirement.txt` |
| First query very slow | Models downloading / loading into memory — subsequent calls are faster |

---

## Disclaimer

Muses assists with **document retrieval and explanation**. It is **not legal advice**. Always verify citations against the source PDF and consult a qualified lawyer for legal opinions.

---

## License

MIT
