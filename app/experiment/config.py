"""ExperimentConfig - every knob for one RAG experiment, in one place.

The spec's rule for this phase: "do not hardcode these throughout the application".
`config.py` (Settings) holds *process* defaults - API keys, file paths. This holds
the parameters that change *per experiment run* and get recorded with the result so
Phase 12 can compare runs.

`reranker_enabled` and `retrieval_strategy != "dense"` are accepted here (the config
stays a pure data object so Phase 12 can load and compare any config) but rejected by
the runner's factory with a clear NotImplementedError - they are the designated
extension points for a later phase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str

    # --- chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- retrieval ---
    top_k: int = 4
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_strategy: str = "dense"   # only "dense" is implemented
    reranker_enabled: bool = False      # not implemented; the factory raises if True

    # --- generation ---
    generation_model: str = "claude-opus-5"
    temperature: float = 0.0            # recorded; current Claude models ignore it
    citations_enabled: bool = False     # use the cited generator + citation eval

    # --- which evaluators to run (cost control) ---
    run_retrieval_eval: bool = True
    run_generation_eval: bool = True    # deterministic scorers ...
    use_judge: bool = True              # ... plus the LLM judge
    run_faithfulness: bool = False
    run_citation_eval: bool = False     # forced on when citations_enabled
    judge_model: Optional[str] = None   # defaults to generation_model
    max_retries: int = 2

    # --- data / limits ---
    documents_dir: Optional[str] = None
    dataset_path: Optional[str] = None
    limit: Optional[int] = None

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.generation_model

    @property
    def citation_eval_enabled(self) -> bool:
        return self.run_citation_eval or self.citations_enabled

    @model_validator(mode="after")
    def _numeric_checks(self) -> "ExperimentConfig":
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when set")
        return self


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.model_dump(), indent=2) + "\n", encoding="utf-8")
    return path
