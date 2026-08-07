# app/api/server.py
"""
Muses FastAPI server.

Start with:
    uvicorn app.api.server:app --reload --host 0.0.0.0 --port 8000

Or via the CLI helper:
    python main.py --serve
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.ingest import router as ingest_router
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.eval import router as eval_router

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Muses RAG API",
    description="Multimodal RAG — upload, chat, download cited documents, and evaluate performance.",
    version="1.1.0",
)

# ── CORS (open for local dev) ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(ingest_router,    prefix="/api", tags=["Ingest"])
app.include_router(chat_router,      prefix="/api", tags=["Chat"])
app.include_router(documents_router, prefix="/api", tags=["Documents"])
app.include_router(eval_router,      prefix="/api", tags=["Evaluation & Cache"])

# ── Serve frontend static files ───────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ── Startup hook ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    uploads = Path("uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    print("✔ Muses API ready — uploads dir:", uploads.resolve())
