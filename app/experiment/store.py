"""Persistent experiment tracking in SQLite (stdlib `sqlite3`, no ORM).

Storage strategy - **columns for querying, JSON for fidelity**:

* Headline aggregates, config essentials, latency, tokens and cost are stored as
  real columns, so "list experiments", filtering, Phase 12 comparison and Phase 13
  "worst questions" can be plain SQL.
* The complete ``ExperimentResult`` (and each ``QuestionExperimentResult``) is also
  stored verbatim as a JSON string, so ``get()`` reconstructs the exact Pydantic
  object with nothing lost. Adding a metric later never requires a migration.

Two tables: ``experiments`` (one row per run) and ``question_results`` (one row per
question, ``ON DELETE CASCADE``).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from app.config import settings
from app.experiment.results import ExperimentResult, QuestionExperimentResult
from app.observability.trace import Trace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),

    num_questions   INTEGER NOT NULL,
    num_errors      INTEGER NOT NULL,
    document_count  INTEGER NOT NULL,
    chunk_count     INTEGER NOT NULL,

    chunk_size       INTEGER NOT NULL,
    chunk_overlap    INTEGER NOT NULL,
    top_k            INTEGER NOT NULL,
    generation_model TEXT NOT NULL,
    embedding_model  TEXT NOT NULL,
    citations_enabled INTEGER NOT NULL,

    retrieval_hit_rate  REAL,
    retrieval_precision REAL,
    retrieval_recall    REAL,
    retrieval_mrr       REAL,
    retrieval_ndcg      REAL,

    gen_exact_match         REAL,
    gen_token_f1            REAL,
    gen_abstention_accuracy REAL,
    judge_correctness       REAL,
    judge_relevance         REAL,

    faithfulness_macro REAL,
    faithfulness_micro REAL,

    citation_completeness       REAL,
    citation_precision          REAL,
    citation_correctness        REAL,
    citation_hallucination_rate REAL,

    latency_retrieval_ms  REAL,
    latency_generation_ms REAL,
    latency_evaluation_ms REAL,
    latency_total_ms      REAL,

    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    estimated_cost_usd REAL,

    result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS question_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    question_id   TEXT NOT NULL,
    generated_answer    TEXT,
    retrieved_chunk_ids TEXT,   -- JSON array
    error TEXT,

    retrieval_recall   REAL,
    retrieval_hit_rate REAL,
    reciprocal_rank    REAL,
    judge_correctness  REAL,
    judge_relevance    REAL,
    faithfulness_score REAL,
    citation_precision REAL,
    citation_completeness REAL,

    latency_total_ms  REAL,
    total_tokens      INTEGER,
    estimated_cost_usd REAL,

    detail_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_qr_experiment ON question_results(experiment_id);

CREATE TABLE IF NOT EXISTS traces (
    trace_id      TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    question_id   TEXT,
    total_ms      REAL,
    total_tokens  INTEGER,
    estimated_cost_usd REAL,
    has_error     INTEGER NOT NULL,
    trace_json    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_traces_experiment ON traces(experiment_id);
"""


class ExperimentSummary(BaseModel):
    """The list-view projection of an experiment row."""

    experiment_id: str
    experiment_name: str
    started_at: str
    num_questions: int
    num_errors: int
    generation_model: str
    chunk_size: int
    top_k: int
    retrieval_recall: Optional[float] = None
    judge_correctness: Optional[float] = None
    faithfulness_macro: Optional[float] = None
    citation_precision: Optional[float] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: float = 0.0


def _opt(obj, *path):
    """Safe nested attribute access: _opt(result, 'retrieval', 'recall') -> value or None."""
    for name in path:
        if obj is None:
            return None
        obj = getattr(obj, name, None)
    return obj


class ExperimentStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.path = str(db_path or settings.experiments_db)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    # -- lifecycle -------------------------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes ---------------------------------------------------------------------------
    def save(self, result: ExperimentResult) -> None:
        """Insert (or replace) one experiment and all its question rows, atomically."""
        exp_row = self._experiment_row(result)
        with self._conn:  # transaction
            cols = ", ".join(exp_row)
            placeholders = ", ".join(f":{c}" for c in exp_row)
            self._conn.execute(
                f"INSERT OR REPLACE INTO experiments ({cols}) VALUES ({placeholders})", exp_row
            )
            self._conn.execute(
                "DELETE FROM question_results WHERE experiment_id = ?", (result.experiment_id,)
            )
            for question in result.per_question:
                q_row = self._question_row(result.experiment_id, question)
                q_cols = ", ".join(q_row)
                q_ph = ", ".join(f":{c}" for c in q_row)
                self._conn.execute(
                    f"INSERT INTO question_results ({q_cols}) VALUES ({q_ph})", q_row
                )

            self._conn.execute(
                "DELETE FROM traces WHERE experiment_id = ?", (result.experiment_id,)
            )
            for trace in result.traces:
                t_row = self._trace_row(result.experiment_id, trace)
                t_cols = ", ".join(t_row)
                t_ph = ", ".join(f":{c}" for c in t_row)
                self._conn.execute(f"INSERT INTO traces ({t_cols}) VALUES ({t_ph})", t_row)

    def delete(self, experiment_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,)
            )
        return cur.rowcount > 0

    # -- reads ----------------------------------------------------------------------------
    def exists(self, experiment_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return row is not None

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    def list(self, limit: int = 50) -> list[ExperimentSummary]:
        rows = self._conn.execute(
            "SELECT * FROM experiments ORDER BY started_at DESC, created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            ExperimentSummary(
                experiment_id=r["experiment_id"],
                experiment_name=r["experiment_name"],
                started_at=r["started_at"],
                num_questions=r["num_questions"],
                num_errors=r["num_errors"],
                generation_model=r["generation_model"],
                chunk_size=r["chunk_size"],
                top_k=r["top_k"],
                retrieval_recall=r["retrieval_recall"],
                judge_correctness=r["judge_correctness"],
                faithfulness_macro=r["faithfulness_macro"],
                citation_precision=r["citation_precision"],
                total_tokens=r["total_tokens"],
                estimated_cost_usd=r["estimated_cost_usd"] or 0.0,
            )
            for r in rows
        ]

    def get(self, experiment_id: str) -> Optional[ExperimentResult]:
        row = self._conn.execute(
            "SELECT result_json FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        return ExperimentResult.model_validate_json(row["result_json"])

    def get_traces(self, experiment_id: str) -> list[Trace]:
        rows = self._conn.execute(
            "SELECT trace_json FROM traces WHERE experiment_id = ? ORDER BY rowid", (experiment_id,)
        ).fetchall()
        return [Trace.model_validate_json(r["trace_json"]) for r in rows]

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        row = self._conn.execute(
            "SELECT trace_json FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        return Trace.model_validate_json(row["trace_json"]) if row else None

    # -- row builders --------------------------------------------------------------------
    @staticmethod
    def _experiment_row(result: ExperimentResult) -> dict:
        cfg = result.config
        return {
            "experiment_id": result.experiment_id,
            "experiment_name": cfg.experiment_name,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "num_questions": result.num_questions,
            "num_errors": result.num_errors,
            "document_count": result.document_count,
            "chunk_count": result.chunk_count,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "top_k": cfg.top_k,
            "generation_model": cfg.generation_model,
            "embedding_model": cfg.embedding_model,
            "citations_enabled": int(cfg.citations_enabled),
            "retrieval_hit_rate": _opt(result, "retrieval", "hit_rate"),
            "retrieval_precision": _opt(result, "retrieval", "precision"),
            "retrieval_recall": _opt(result, "retrieval", "recall"),
            "retrieval_mrr": _opt(result, "retrieval", "mrr"),
            "retrieval_ndcg": _opt(result, "retrieval", "ndcg"),
            "gen_exact_match": _opt(result, "generation", "exact_match"),
            "gen_token_f1": _opt(result, "generation", "token_f1"),
            "gen_abstention_accuracy": _opt(result, "generation", "abstention_accuracy"),
            "judge_correctness": _opt(result, "generation", "judge_correctness"),
            "judge_relevance": _opt(result, "generation", "judge_relevance"),
            "faithfulness_macro": _opt(result, "faithfulness", "faithfulness"),
            "faithfulness_micro": _opt(result, "faithfulness", "claim_support_rate"),
            "citation_completeness": _opt(result, "citation", "citation_completeness"),
            "citation_precision": _opt(result, "citation", "citation_precision"),
            "citation_correctness": _opt(result, "citation", "citation_correctness"),
            "citation_hallucination_rate": _opt(result, "citation", "citation_hallucination_rate"),
            "latency_retrieval_ms": result.latency.retrieval_ms,
            "latency_generation_ms": result.latency.generation_ms,
            "latency_evaluation_ms": result.latency.evaluation_ms,
            "latency_total_ms": result.latency.total_ms,
            "prompt_tokens": result.total_token_usage.prompt_tokens,
            "completion_tokens": result.total_token_usage.completion_tokens,
            "total_tokens": result.total_token_usage.total_tokens,
            "estimated_cost_usd": result.estimated_cost_usd,
            "result_json": result.model_dump_json(),
        }

    @staticmethod
    def _question_row(experiment_id: str, q: QuestionExperimentResult) -> dict:
        return {
            "experiment_id": experiment_id,
            "question_id": q.question_id,
            "generated_answer": q.generated_answer,
            "retrieved_chunk_ids": json.dumps(q.retrieved_chunk_ids),
            "error": q.error,
            "retrieval_recall": _opt(q, "retrieval", "metrics", "recall"),
            "retrieval_hit_rate": _opt(q, "retrieval", "metrics", "hit_rate"),
            "reciprocal_rank": _opt(q, "retrieval", "metrics", "reciprocal_rank"),
            "judge_correctness": _opt(q, "generation", "judgement", "correctness"),
            "judge_relevance": _opt(q, "generation", "judgement", "relevance"),
            "faithfulness_score": _opt(q, "faithfulness", "result", "score"),
            "citation_precision": _opt(q, "citation", "result", "citation_precision"),
            "citation_completeness": _opt(q, "citation", "result", "citation_completeness"),
            "latency_total_ms": q.latency_ms.get("total"),
            "total_tokens": q.token_usage.total_tokens,
            "estimated_cost_usd": q.estimated_cost_usd,
            "detail_json": q.model_dump_json(),
        }

    @staticmethod
    def _trace_row(experiment_id: str, trace: Trace) -> dict:
        return {
            "trace_id": trace.trace_id,
            "experiment_id": experiment_id,
            "question_id": trace.question_id,
            "total_ms": trace.performance.total_ms,
            "total_tokens": trace.performance.token_usage.total_tokens,
            "estimated_cost_usd": trace.performance.estimated_cost_usd,
            "has_error": int(bool(trace.errors)),
            "trace_json": trace.model_dump_json(),
        }
