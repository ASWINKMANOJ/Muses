# app/generation/llm.py
"""
LLM interface for Muses RAG.

Improvements:
- Legal-specific system prompt: instructs model to cite exact provisions,
  use verbatim quotes, and add a disclaimer.
- generate_hypothetical_answer() for HyDE query expansion.
- Explicit num_gpu option passed to Ollama payload for GPU layer offloading.
- All parameters driven from centralized settings.
"""

import json
import requests

from app.core.config import settings


# ── System prompt ─────────────────────────────────────────────────────────────

LEGAL_SYSTEM_PROMPT = """\
You are a precise legal document assistant. Your role is to help users \
understand legal documents by retrieving and interpreting relevant provisions.

RULES:
1. BASE YOUR ANSWER ENTIRELY on the provided context excerpts. Do not invent \
   clauses, facts, or obligations that are not in the context.
2. CITE PRECISELY: After every factual claim, add an inline citation in the \
   format [Doc: <source>, Page: <page>, Section: <heading>]. If multiple \
   sections support the claim, cite all of them.
3. QUOTE VERBATIM when precision matters: use block quotes (>) for direct \
   excerpts, especially for defined terms, obligations, and penalties.
4. INTERPRET CAREFULLY: After quoting, briefly explain in plain English what \
   the provision means. Distinguish between "the document states" and your \
   interpretation.
5. If the context does not contain enough information to answer the question, \
   say explicitly: "The provided documents do not contain sufficient information \
   to answer this question."
6. DISCLAIMER: End every response with: \
   "*This is document retrieval assistance, not legal advice. \
   Consult a qualified lawyer for legal opinions.*"
7. Keep responses structured. Use headings for multi-part answers. \
   Use bullet lists for obligations or conditions.
"""


# ── Streaming answer ──────────────────────────────────────────────────────────

def stream_answer(query: str, context_chunks: list[str]):
    """
    Stream an LLM answer token by token (generator).

    Args:
        query:          User's question.
        context_chunks: List of formatted context strings with source metadata.

    Yields:
        str tokens from the LLM.
    """
    if not context_chunks:
        yield "No relevant provisions found in the uploaded documents."
        return

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""{LEGAL_SYSTEM_PROMPT}

---

CONTEXT EXCERPTS FROM DOCUMENTS:
{context}

---

QUESTION:
{query}

---

ANSWER (cite provisions inline, quote verbatim where relevant):
"""

    try:
        response = requests.post(
            settings.ollama_url,
            json={
                "model": settings.llm_model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": settings.llm_temperature,
                    "top_p": settings.llm_top_p,
                    "num_ctx": settings.llm_num_ctx,
                    "num_predict": settings.llm_num_predict,
                    "num_gpu": settings.llm_num_gpu,  # 🔥 Force GPU layer offloading
                },
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        yield (
            "\n⚠ Cannot connect to Ollama at "
            f"`{settings.ollama_url}`. "
            "Run `ollama serve` and try again."
        )
        return
    except requests.exceptions.RequestException as e:
        yield f"\n⚠ LLM request error: {e}"
        return

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")
                if token:
                    yield token
            except json.JSONDecodeError:
                continue


# ── HyDE: Hypothetical Document Embedding ─────────────────────────────────────

HYDE_PROMPT_TEMPLATE = """\
You are a legal document drafting assistant. Given the following question \
about a legal document, write a short paragraph (3-5 sentences) that \
represents what the ANSWER might look like if it were found in a legal \
contract or statute. Use formal legal language. Do not say you don't know — \
always produce a plausible hypothetical legal provision.

Question: {query}

Hypothetical legal provision (answer only, no preamble):
"""


def generate_hypothetical_answer(query: str) -> str | None:
    """
    Generate a hypothetical document excerpt for HyDE retrieval.

    Returns the generated text, or None if the LLM is unreachable.
    This call is non-streaming and intentionally short (128 tokens).
    """
    prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
    try:
        response = requests.post(
            settings.ollama_url,
            json={
                "model": settings.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 128,
                    "num_ctx": 1024,
                    "num_gpu": settings.llm_num_gpu,  # 🔥 Force GPU layer offloading
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip() or None
    except Exception as e:
        print(f"[llm] HyDE generation failed (non-fatal): {e}")
        return None
