"""Prompt construction.

Isolated in its own module because the prompt *is* a big part of RAG quality, and
Phase 6 (citations) and the experiment runner will want to version and swap it. Keep
prompt text here, not inlined in the generator.

The numbered context blocks (`[1]`, `[2]`, ...) are deliberate: they cost nothing now
and they are exactly the handles Phase 6 uses for citation grounding.
"""

from __future__ import annotations

from app.models import RetrievedChunk

INSUFFICIENT_CONTEXT_REPLY = "The provided context is insufficient to answer this question."

SYSTEM_PROMPT = f"""You are a precise question-answering assistant.

Answer the user's question using ONLY the information in the provided context blocks.

Rules:
- Do not use outside knowledge or make assumptions beyond the context.
- If the context does not contain the answer, reply with exactly this sentence and nothing else:
  "{INSUFFICIENT_CONTEXT_REPLY}"
- Be concise. Quote figures, dates, and durations exactly as they appear in the context.
"""


def format_context(retrieved: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, source-attributed blocks."""
    if not retrieved:
        return "(no context was retrieved)"
    return "\n\n".join(
        f"[{item.rank}] (source: {item.chunk.document_id})\n{item.chunk.text.strip()}"
        for item in retrieved
    )


def build_rag_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    """The user-message content sent to the model."""
    return (
        f"Context:\n{format_context(retrieved)}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
