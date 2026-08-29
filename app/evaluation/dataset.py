"""The golden evaluation dataset.

This is the *ground truth*. Every metric from Phase 3 onward is defined as "how far
is the pipeline's output from what this dataset says the answer should be".

Two types:

* ``EvalExample`` - one labelled question. A Pydantic model so a malformed record in
  the JSON file fails loudly (unknown fields are rejected, blank strings are
  rejected) instead of silently poisoning an experiment.
* ``EvalDataset`` - an in-memory collection of examples. A plain class (not a
  Pydantic model) so it can expose clean iteration and filtering without fighting
  ``BaseModel``'s own ``__iter__``.

Design decision: ground truth is anchored at the **document** level
(``relevant_document_ids``), not the chunk level. Chunk ids depend on ``chunk_size``
/ ``chunk_overlap``, so chunk-level labels would silently break every time Phase 7
sweeps those knobs. ``relevant_chunk_ids`` stays available for cases where you really
do want chunk-precise labels.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from app.config import settings


class EvalExample(BaseModel):
    """One ground-truth question/answer record."""

    model_config = ConfigDict(extra="forbid")  # reject typo'd keys in the JSON

    id: str
    question: str
    expected_answer: str
    relevant_document_ids: list[str]

    # --- optional ---
    relevant_chunk_ids: list[str] = []
    category: Optional[str] = None
    difficulty: Optional[str] = None  # convention: "easy" | "medium" | "hard"
    slices: list[str] = []            # slice labels, used for per-slice metrics in Phase 14
    metadata: dict = {}

    @field_validator("id", "question", "expected_answer")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @property
    def is_unanswerable(self) -> bool:
        """True when the corpus is not expected to contain the answer."""
        return not self.relevant_document_ids or "unanswerable" in self.slices


class EvalDataset:
    """An ordered collection of ``EvalExample`` with lookup and filtering helpers."""

    def __init__(self, examples: Sequence[EvalExample]) -> None:
        self._examples: list[EvalExample] = list(examples)

        self._by_id: dict[str, EvalExample] = {}
        for example in self._examples:
            if example.id in self._by_id:
                raise ValueError(f"Duplicate eval example id: {example.id!r}")
            self._by_id[example.id] = example

    # -- iteration / container protocol -------------------------------------------------
    def __iter__(self) -> Iterator[EvalExample]:
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> EvalExample:
        return self._examples[index]

    @property
    def examples(self) -> list[EvalExample]:
        return list(self._examples)  # copy, so callers can't mutate our list

    # -- lookup ----------------------------------------------------------------------------
    def get(self, example_id: str) -> Optional[EvalExample]:
        return self._by_id.get(example_id)

    # -- aggregate views -----------------------------------------------------------------
    def categories(self) -> list[str]:
        return sorted({e.category for e in self._examples if e.category})

    def slice_labels(self) -> list[str]:
        return sorted({label for e in self._examples for label in e.slices})

    # -- filtering (returns a new EvalDataset) -----------------------------------------
    def filter_by_slice(self, label: str) -> "EvalDataset":
        return EvalDataset([e for e in self._examples if label in e.slices])

    def filter_by_category(self, category: str) -> "EvalDataset":
        return EvalDataset([e for e in self._examples if e.category == category])


def load_eval_dataset(path: str | Path | None = None) -> EvalDataset:
    """Load and validate the dataset from a JSON file (a top-level array of records).

    ``path`` defaults to ``settings.eval_dataset_path``.
    """
    path = Path(path) if path is not None else Path(settings.eval_dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Eval dataset file must contain a JSON array of records.")

    examples = [EvalExample.model_validate(item) for item in raw]
    return EvalDataset(examples)
