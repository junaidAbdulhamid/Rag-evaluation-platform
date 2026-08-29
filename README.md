# RAG Evaluation & Observability Platform

A production-style platform for **evaluating, comparing, debugging, and observing** RAG
pipelines — not just running one. Built incrementally, one phase at a time, as a
learning project.

## Status

| Phase | Scope | State |
|------:|-------|-------|
| 1 | Minimal RAG pipeline (load → chunk → embed → store → retrieve → generate) | ✅ done |
| 2 | Golden evaluation dataset | ✅ done |
| 3 | Retrieval evaluation (Hit@K, Precision@K, Recall@K, MRR, NDCG) | ✅ done |
| 4 | Generation evaluation (correctness, relevance; LLM-as-judge) | ✅ done |
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
  llm.py                    TextLLM seam (text-in/text-out) + AnthropicTextLLM
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
  evaluation/
    dataset.py              EvalExample model + EvalDataset container + JSON loader
    retrieval_metrics.py    pure from-scratch metrics: Hit/Precision/Recall/RR/DCG/NDCG @k
    retrieval.py            chunk→doc reduction, per-question + aggregate, dataset driver
    generation_metrics.py   deterministic scorers: exact_match, token_f1, number_coverage, abstention
    judge.py                LLM-as-judge: strict GenerationJudgement model + parse/retry loop
    generation.py           deterministic + judge per-question + aggregate, dataset driver
data/documents/             sample corpus (6 policy docs)
data/eval_dataset.json      24 labelled ground-truth questions
scripts/ask.py              Phase 1 CLI
scripts/show_dataset.py     Phase 2 CLI (dataset summary / iteration demo)
scripts/eval_retrieval.py   Phase 3 CLI (retrieval metrics + worst questions)
scripts/eval_generation.py  Phase 4 CLI (deterministic + judge, --no-judge / --limit)
tests/                      unit tests per module
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

python -m scripts.show_dataset                       # dataset summary
python -m scripts.show_dataset --slice multi_document --full

python -m scripts.eval_retrieval                     # retrieval metrics @ default k
python -m scripts.eval_retrieval --top-k 1 --worst 3 # see the precision/recall tradeoff

python -m scripts.eval_generation --no-judge         # deterministic scorers only (1 API call/Q)
python -m scripts.eval_generation --limit 5          # + LLM judge on the first 5 questions
```

## Test

```bash
pytest
```
