"""Embeddings.

`EmbeddingProvider` is the seam between "we need vectors for this text" and "which
model / API produces them". The rest of the platform depends only on the abstract
base class, so Phase 7 can swap embedding models per experiment, and you could add an
OpenAI or Voyage provider later without touching retrieval code.

Phase 1 ships one real implementation: a local `sentence-transformers` model. It runs
on your machine with no API key and no per-call cost, which matters because
evaluation re-embeds the whole corpus on every experiment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Contract every embedding backend must satisfy."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single string (typically a query)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings (typically chunk texts). Order is preserved."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vectors this provider returns."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for the underlying model, recorded in traces/experiments."""

    def count_tokens(self, text: str) -> int:
        """Estimate the token count of `text` (for cost tracking).

        Default: the standard ~4-chars-per-token approximation. A provider backed by
        a real tokenizer should override this for an exact count.
        """
        return max(1, len(text) // 4)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local embeddings via the `sentence-transformers` library.

    The model is downloaded (and cached under ~/.cache) the first time it is
    constructed. `normalize_embeddings=True` returns unit vectors, so a dot product
    between two of them is exactly their cosine similarity.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        # Imported lazily so that code paths / tests that never embed anything
        # (e.g. the chunker tests) don't pay the torch import cost.
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    def count_tokens(self, text: str) -> int:
        """Exact count via the model's own tokenizer (no special tokens)."""
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))
