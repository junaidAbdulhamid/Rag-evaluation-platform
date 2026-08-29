import json
from pathlib import Path

import pytest

from app.evaluation.dataset import EvalDataset, EvalExample, load_eval_dataset
from app.generation.prompt import INSUFFICIENT_CONTEXT_REPLY

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- a minimal in-memory dataset used by the unit tests -----------------------------
RECORDS = [
    {
        "id": "a1",
        "question": "  How long is the warranty?  ",
        "expected_answer": "12 months.",
        "relevant_document_ids": ["warranty_policy"],
        "category": "warranty",
        "difficulty": "easy",
        "slices": ["simple_lookup", "numerical"],
    },
    {
        "id": "a2",
        "question": "Which countries do you ship to?",
        "expected_answer": "US and Canada.",
        "relevant_document_ids": ["shipping_policy"],
        "category": "shipping",
        "slices": ["policy"],
    },
    {
        "id": "a3",
        "question": "Do you price match?",
        "expected_answer": INSUFFICIENT_CONTEXT_REPLY,
        "relevant_document_ids": [],
        "category": "unsupported",
        "slices": ["unanswerable"],
    },
]


def write_dataset(tmp_path: Path, records) -> Path:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


# --- EvalExample validation ---------------------------------------------------------
def test_blank_question_is_rejected():
    with pytest.raises(ValueError):
        EvalExample(id="x", question="   ", expected_answer="a", relevant_document_ids=[])


def test_whitespace_is_stripped_from_key_fields():
    example = EvalExample.model_validate(RECORDS[0])
    assert example.question == "How long is the warranty?"


def test_unknown_field_is_rejected():
    bad = {**RECORDS[0], "catgory": "typo"}  # misspelled key
    with pytest.raises(ValueError):
        EvalExample.model_validate(bad)


def test_is_unanswerable_property():
    assert EvalExample.model_validate(RECORDS[2]).is_unanswerable is True
    assert EvalExample.model_validate(RECORDS[0]).is_unanswerable is False


# --- EvalDataset behaviour --------------------------------------------------------------
def test_load_parses_all_records(tmp_path: Path):
    dataset = load_eval_dataset(write_dataset(tmp_path, RECORDS))

    assert len(dataset) == 3
    assert [e.id for e in dataset] == ["a1", "a2", "a3"]  # iteration preserves order
    assert dataset[0].id == "a1"                           # __getitem__


def test_duplicate_ids_raise(tmp_path: Path):
    dupes = [RECORDS[0], {**RECORDS[1], "id": "a1"}]
    with pytest.raises(ValueError, match="Duplicate"):
        load_eval_dataset(write_dataset(tmp_path, dupes))


def test_non_array_json_raises(tmp_path: Path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"examples": RECORDS}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_eval_dataset(path)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_eval_dataset(tmp_path / "nope.json")


def test_lookup_and_aggregate_views(tmp_path: Path):
    dataset = load_eval_dataset(write_dataset(tmp_path, RECORDS))

    assert dataset.get("a2").question == "Which countries do you ship to?"
    assert dataset.get("missing") is None
    assert dataset.categories() == ["shipping", "unsupported", "warranty"]
    assert dataset.slice_labels() == ["numerical", "policy", "simple_lookup", "unanswerable"]


def test_filtering_returns_subset_datasets(tmp_path: Path):
    dataset = load_eval_dataset(write_dataset(tmp_path, RECORDS))

    numerical = dataset.filter_by_slice("numerical")
    assert isinstance(numerical, EvalDataset)
    assert [e.id for e in numerical] == ["a1"]

    shipping = dataset.filter_by_category("shipping")
    assert [e.id for e in shipping] == ["a2"]


# --- the dataset that actually ships with the repo ------------------------------------
def test_shipped_dataset_is_valid_and_grounded():
    dataset = load_eval_dataset(REPO_ROOT / "data" / "eval_dataset.json")

    assert len(dataset) >= 20  # Phase 2 asks for 15-20+

    doc_ids_on_disk = {p.stem for p in (REPO_ROOT / "data" / "documents").glob("*.md")}
    for example in dataset:
        for doc_id in example.relevant_document_ids:
            assert doc_id in doc_ids_on_disk, f"{example.id} references unknown doc {doc_id!r}"

        if example.is_unanswerable:
            assert example.relevant_document_ids == []
            assert example.expected_answer == INSUFFICIENT_CONTEXT_REPLY


def test_shipped_dataset_has_slice_coverage():
    dataset = load_eval_dataset(REPO_ROOT / "data" / "eval_dataset.json")
    labels = set(dataset.slice_labels())

    # the slices Phase 14 will report on should each have at least one example
    for required in {"simple_lookup", "numerical", "policy", "multi_document", "unanswerable"}:
        assert required in labels
