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
| 5 | Faithfulness evaluation (claim extraction + grounding) | ✅ done |
| 6 | Citation-grounded RAG + citation metrics | ✅ done |
| 7 | Experiment configuration system | ✅ done |
| 8 | Experiment tracking (SQLite) | ✅ done |
| 9 | Observability & tracing | ✅ done |
| 10 | Latency tracking | ✅ done |
| 11 | Token & cost tracking | ✅ done |
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
    citation.py             CitedGenerator: [n] markers tied to retrieved chunks
  evaluation/
    dataset.py              EvalExample model + EvalDataset container + JSON loader
    retrieval_metrics.py    pure from-scratch metrics: Hit/Precision/Recall/RR/DCG/NDCG @k
    retrieval.py            chunk→doc reduction, per-question + aggregate, dataset driver
    generation_metrics.py   deterministic scorers: exact_match, token_f1, number_coverage, abstention
    judge.py                LLM-as-judge: strict GenerationJudgement model + parse/retry loop
    generation.py           deterministic + judge per-question + aggregate, dataset driver
    structured_output.py    shared JSON-extract + validate + retry helper for LLM calls
    faithfulness.py         claim extraction + verification vs numbered context; chunk-id linkage
    faithfulness_eval.py    per-question + macro/micro aggregate, dataset driver
    citation.py             citation precision / completeness / correctness / hallucination
    citation_eval.py        per-question + macro/micro aggregate, dataset driver
  experiment/
    config.py               ExperimentConfig - every per-run knob, in one model
    runner.py               run_experiment(config): wires Phases 1-6, times/meters/costs, traces, saves
    results.py              ExperimentResult + per-question record models
    metering.py             TokenMeter + RecordingTextLLM (evaluation token capture)
    cost.py                 pricing registry (MODEL_PRICING / EMBEDDING_PRICING) + cost fns; file overrides
    store.py                SQLite tracking: columns for querying + JSON for fidelity (+ traces table)
  observability/
    trace.py                Trace model - question / retrieval / generation / evaluation / performance
    recorder.py             TraceRecorder: context-manager timing, builds a Trace per execution
    latency.py              contextvar stage timing + mean/median/p95 distribution stats
    timing.py               record_ms explicit-dict stopwatch (TraceRecorder standalone path)
data/documents/             sample corpus (6 policy docs)
data/eval_dataset.json      24 labelled ground-truth questions
scripts/ask.py              Phase 1 CLI
scripts/show_dataset.py     Phase 2 CLI (dataset summary / iteration demo)
scripts/eval_retrieval.py   Phase 3 CLI (retrieval metrics + worst questions)
scripts/eval_generation.py  Phase 4 CLI (deterministic + judge, --no-judge / --limit)
scripts/eval_faithfulness.py Phase 5 CLI (claim-level grounding, --limit)
scripts/eval_citations.py   Phase 6 CLI (cited answers + citation metrics, --limit)
scripts/run_experiment.py   Phase 7 CLI (run one config end to end; saves to the DB)
scripts/experiments.py      Phase 8 CLI (list / show / metrics / delete tracked runs)
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

python -m scripts.eval_faithfulness --limit 5        # claim-level grounding vs retrieved context
python -m scripts.eval_citations --limit 5           # cited answers + citation precision/completeness

python -m scripts.run_experiment --name base --limit 5              # one full run -> saved to db
python -m scripts.run_experiment --name c300 --chunk-size 300 --top-k 3 --faithfulness --limit 5

python -m scripts.experiments list                    # tracked runs, newest first
python -m scripts.experiments metrics <experiment_id> # full aggregate metrics for one run
python -m scripts.experiments trace <experiment_id>   # full execution trace for the first question
python -m scripts.experiments trace <experiment_id> --question q005
python -m scripts.experiments cost <experiment_id>    # cost breakdown + quality-vs-cost
```

## Test

```bash
pytest
```
