import os
import sys
from pathlib import Path
import argparse


# ── Colour helpers (no external deps) ────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def c(text, *codes): return "".join(codes) + str(text) + RESET
def banner():
    print(c("""
 ███╗   ███╗██╗   ██╗███████╗███████╗███████╗
 ████╗ ████║██║   ██║██╔════╝██╔════╝██╔════╝
 ██╔████╔██║██║   ██║███████╗█████╗  ███████╗
 ██║╚██╔╝██║██║   ██║╚════██║██╔══╝  ╚════██║
 ██║ ╚═╝ ██║╚██████╔╝███████║███████╗███████║
 ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝
""", CYAN, BOLD))
    print(c("  Document Q&A — Powered by local LLM + ChromaDB\n", DIM))


# ── Supported file types (extend here later) ─────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"}


def resolve_path(raw: str) -> Path:
    """Resolve user-supplied path; accept quoted strings and ~ expansion."""
    return Path(raw.strip().strip('"').strip("'")).expanduser().resolve()


# ── Ingestion ─────────────────────────────────────────────────────────────────
def ingest_document(file_path: Path) -> bool:
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        print(c(f"\n  ✗ Unsupported file type '{ext}'. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}\n", RED))
        return False

    if not file_path.exists():
        print(c(f"\n  ✗ File not found: {file_path}\n", RED))
        return False

    print(c(f"\n  ● Ingesting: {file_path.name}", CYAN))

    try:
        # Dispatch generic ingest pipeline
        from app.pipeline.ingest_pipeline import ingest_file_pipeline
        result = ingest_file_pipeline(str(file_path))

        print(c(f"  ✔ Done — {result.get('chunks', 0)} chunks stored.\n", GREEN))
        return True

    except Exception as e:
        print(c(f"\n  ✗ Ingestion failed: {e}\n", RED))
        return False


# ── Query loop ────────────────────────────────────────────────────────────────
def query_loop():
    from app.pipeline.query_pipeline import query_pipeline_stream

    print(c("  Type your question below.", DIM))
    print(c("  Commands: 'ingest' — add another doc | 'quit' / 'exit' — leave\n", DIM))

    while True:
        try:
            query = input(c("  You › ", CYAN, BOLD)).strip()
        except (KeyboardInterrupt, EOFError):
            print(c("\n\n  Goodbye!\n", YELLOW))
            sys.exit(0)

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print(c("\n  Goodbye!\n", YELLOW))
            sys.exit(0)

        if query.lower() == "ingest":
            prompt_and_ingest()
            continue

        # ── Stream the answer ────────────────────────────────────────────────
        print(c("\n  Muses › ", GREEN, BOLD), end="", flush=True)
        try:
            for token in query_pipeline_stream(query):
                print(token, end="", flush=True)
        except Exception as e:
            print(c(f"\n  ✗ Generation error: {e}", RED))

        print("\n")  # spacer after answer


# ── Ingestion prompt (reusable) ───────────────────────────────────────────────
def prompt_and_ingest() -> bool:
    """Ask the user for a file path and ingest it. Returns True on success."""
    print(c("\n  Supported types: PDF, TXT, DOCX, PNG, JPG", DIM))
    try:
        raw = input(c("  Document path › ", CYAN, BOLD)).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return False

    if not raw:
        print(c("  No path provided.\n", YELLOW))
        return False

    return ingest_document(resolve_path(raw))


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Muses Document Q&A")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI server instead of the interactive CLI",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host for the FastAPI server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for the FastAPI server (default: 8000)",
    )
    parser.add_argument(
        "file", nargs="?",
        help="Optional document path to ingest before starting the CLI",
    )
    args = parser.parse_args()

    # ── API server mode ───────────────────────────────────────────────────────
    if args.serve:
        import uvicorn
        print(c("\n  ● Starting Muses API server…\n", CYAN, BOLD))
        print(c(f"  → http://{args.host}:{args.port}\n", DIM))
        uvicorn.run(
            "app.api.server:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
        return

    # ── Interactive CLI mode ──────────────────────────────────────────────────
    banner()

    if args.file:
        file_path = resolve_path(args.file)
        success = ingest_document(file_path)
        if not success:
            sys.exit(1)
    else:
        # Interactive ingestion on first launch
        print(c("  No document loaded yet. Let's ingest one first.\n", YELLOW))
        while True:
            success = prompt_and_ingest()
            if success:
                break
            retry = input(c("  Try another file? [y/n] › ", CYAN)).strip().lower()
            if retry != "y":
                print(c("\n  Exiting — no documents loaded.\n", YELLOW))
                sys.exit(0)

    # ── Ask whether to ingest more before querying ────────────────────────────
    while True:
        more = input(c("  Ingest another document? [y/n] › ", CYAN)).strip().lower()
        if more != "y":
            break
        prompt_and_ingest()

    print(c("\n  ✔ Ready. Ask anything about your documents.", GREEN, BOLD))
    print()
    query_loop()


if __name__ == "__main__":
    main()