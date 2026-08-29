"""The RAG pipeline: the system that later phases evaluate and observe.

`RagPipeline` is a thin orchestrator. It owns no logic of its own - it wires the
stage objects together and defines the two operations the rest of the platform cares
about:

    ingest(directory)      files            -> chunks -> vectors in the store
    answer(question, k)     question         -> retrieved chunks -> generated answer

Every collaborator is injected (constructor arguments), so a test or an experiment
can substitute a fake embedder, a different retriever, a stub LLM, etc. without
editing this file. `build_default_pipeline()` is the one place that picks concrete
implementations from `settings`.
"""

from __future__ import annotations

from app.config import Settings, settings as default_settings
from app.generation.generator import AnthropicGenerator, LLMGenerator
from app.ingestion.chunker import TextChunker
from app.ingestion.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from app.ingestion.loader import load_documents
from app.models import IngestionResult, RagResult, RetrievedChunk
from app.retrieval.retriever import BaseRetriever, DenseRetriever
from app.retrieval.vector_store import InMemoryVectorStore, VectorStore


class RagPipeline:
    def __init__(
        self,
        chunker: TextChunker,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        retriever: BaseRetriever,
        generator: LLMGenerator | None = None,
        default_top_k: int = 4,
    ) -> None:
        # `generator` is optional: a retrieval-only pipeline (Phase 3 eval, Phase 9
        # tracing) can be built without paying for an LLM. `answer()` guards on it.
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator
        self.default_top_k = default_top_k

    def ingest(self, documents_dir: str) -> IngestionResult:
        """Load -> chunk -> embed -> store. Returns a small summary."""
        documents = load_documents(documents_dir)
        chunks = self.chunker.chunk_documents(documents)

        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        self.vector_store.add(chunks, vectors)

        return IngestionResult(
            document_count=len(documents),
            chunk_count=len(chunks),
            document_ids=[doc.document_id for doc in documents],
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Run only the retrieval half of the pipeline (no LLM call).

        Phase 3's retrieval evaluation and Phase 9's tracing both need context
        without paying for generation.
        """
        k = top_k if top_k is not None else self.default_top_k
        return self.retriever.retrieve(question, k)

    def answer(self, question: str, top_k: int | None = None) -> RagResult:
        """Retrieve context for `question`, then generate a grounded answer."""
        if self.generator is None:
            raise RuntimeError("pipeline has no generator; it can only .retrieve()")
        retrieved = self.retrieve(question, top_k)
        generated = self.generator.generate(question, retrieved)
        return RagResult(
            question=question,
            retrieved_chunks=retrieved,
            generated_answer=generated,
        )


def build_default_pipeline(config: Settings | None = None) -> RagPipeline:
    """Assemble a pipeline from configuration - the real, production wiring."""
    config = config or default_settings

    embeddings = SentenceTransformerEmbeddingProvider(config.embedding_model_name)
    vector_store = InMemoryVectorStore()

    return RagPipeline(
        chunker=TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        ),
        embeddings=embeddings,
        vector_store=vector_store,
        retriever=DenseRetriever(embeddings, vector_store),
        generator=AnthropicGenerator(
            model=config.generation_model,
            api_key=config.anthropic_api_key,
            max_tokens=config.generation_max_tokens,
        ),
        default_top_k=config.top_k,
    )
