# RAG Evaluation & Observability Platform

A production-style platform for **evaluating, comparing, debugging, and observing** RAG
pipelines — not just running one. Built incrementally, one phase at a time, as a
learning project.

## Status

| Phase | Scope | State |
|------:|-------|-------|
| 1 | Minimal RAG pipeline (load → chunk → embed → store → retrieve → generate) | ✅ done |
| 2 | Golden evaluation dataset | ⬜ |
| 3 | Retrieval evaluation (Hit@K, Precision@K, Recall@K, MRR, NDCG) | ⬜ |
| 4 | Generation evaluation (correctness, relevance; LLM-as-judge) | ⬜ |
| 5 | Faithfulness evaluation (claim extraction + grounding) | ⬜ |
| 6 | Citation-grounded RAG + citation metrics | ⬜ |
| 7 | Experiment configuration system | ⬜ |
| 8 | Experiment tracking (SQLite) | ⬜ |
| 9 | Observability & tracing | ⬜ |
| 10 | Latency tracking | ⬜ |
| 11 | Token & cost tracking | ⬜ |
| 12 | Experiment comparison | ⬜ |
| 13 | Failure analysis | ⬜ |
| 14 | Dataset slices | ⬜ |
| 15 | Streamlit dashboard | ⬜ |

## Architecture (Phase 1)

```
app/
  config.py                 process-level settings (env / .env)
  models.py                 shared Pydantic types (Document, Chunk, RetrievedChunk, ...)
  pipeline.py               RagPipeline orchestrator + build_default_pipeline()
  ingestion/
    loader.py               .txt / .md  -> Document
    chunker.py              Document    -> [Chunk]   (sliding window)
    embeddings.py           EmbeddingProvider ABC + SentenceTransformer impl
  retrieval/
    vector_store.py         VectorStore ABC + InMemoryVectorStore (numpy cosine)
    retriever.py            BaseRetriever ABC + DenseRetriever
  generation/
    prompt.py               system prompt + grounded prompt builder
    generator.py            LLMGenerator ABC + AnthropicGenerator
data/documents/             sample corpus (6 policy docs)
scripts/ask.py              Phase 1 CLI
tests/                      unit tests for loader, chunker, vector store, retriever, pipeline
```

Flow: `Documents → Loader → Chunker → Embeddings → VectorStore`, then
`Question → query embedding → Retriever → RetrievedChunks → Prompt → LLM → Answer`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # runtime + pytest
cp .env.example .env                        # then paste your ANTHROPIC_API_KEY
```

The embedding model (~90 MB) downloads automatically on first use.

## Run

```bash
python -m scripts.ask -q "How many days do I have to request a refund?"
python -m scripts.ask -q "How much is express shipping?" --top-k 3 --show-prompt
```

## Test

```bash
pytest
```
