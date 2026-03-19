import requests
import json


def stream_answer(query: str, context_chunks: list[str]):
    # 🔒 Safety check
    if not context_chunks:
        yield "No relevant context found."
        return

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI assistant answering STRICTLY from the provided context.

IMPORTANT RULES:
- Use ONLY the given context
- Do NOT use outside knowledge
- Do NOT show thinking or reasoning steps
- Be concise and structured
- If answer is not in context, say: "Not found in provided documents"

RESPONSE FORMAT (STRICT):

Answer:
- Point 1
- Point 2
- Point 3

Citations:
- [Source: <filename>, Page: <page>, Section: <heading>]
- [Source: <filename>, Page: <page>, Section: <heading>]

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
                "top_p": 0.9
            }
        },
        stream=True
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