import json

import requests


def stream_answer(query: str, context_chunks: list[str]):
    # 🔒 Safety check
    if not context_chunks:
        yield "No relevant context found."
        return

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant that answers questions using the provided context from embedded documents.

RULES:
- The context below is your PRIMARY source of truth — always base your answer on it.
- If the context directly answers the question, use it. Cite the source at the end.
- If you have general knowledge that helps clarify or explain something mentioned in the context, add a brief and simple explanation — but keep it grounded in what the context says.
- Do NOT invent facts or details that are not in the context.
- Do NOT show your reasoning or thinking steps.
- Keep responses concise. Use plain prose for simple questions; use a short list only when the answer has clearly multiple distinct points.
- If the context is completely unrelated to the question or empty, reply: "I couldn't find anything relevant in the provided documents."

---

Context:
{context}

---

Question:
{query}

---

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,  # 🔥 reduces hallucination
                "top_p": 0.9,
                "num_ctx": 4096,  # Limit context window for 4GB VRAM
                "num_predict": 512,  # Prevent runaway generation consuming VRAM
            },
        },
        stream=True,
    )

    # 🔥 Stream tokens safely
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")
                yield token
            except json.JSONDecodeError:
                continue
