import json

import pytest

from app.experiment.config import (
    ExperimentConfig,
    load_experiment_config,
    save_experiment_config,
)


def test_defaults_and_derived_properties():
    c = ExperimentConfig(experiment_name="baseline")
    assert c.chunk_size == 500 and c.top_k == 4
    assert c.retrieval_strategy == "dense"
    assert c.effective_judge_model == c.generation_model
    assert c.citation_eval_enabled is False


def test_judge_model_override_and_citations_force_citation_eval():
    c = ExperimentConfig(experiment_name="x", judge_model="claude-haiku-4-5", citations_enabled=True)
    assert c.effective_judge_model == "claude-haiku-4-5"
    assert c.citation_eval_enabled is True  # forced on by citations_enabled


def test_unknown_field_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(experiment_name="x", chunksize=500)  # typo


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_overlap": 500, "chunk_size": 500},  # overlap == size
        {"chunk_overlap": -1},
        {"top_k": 0},
        {"limit": 0},
    ],
)
def test_numeric_validation(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(experiment_name="x", **kwargs)


def test_json_round_trip(tmp_path):
    c = ExperimentConfig(experiment_name="chunk300", chunk_size=300, chunk_overlap=30, top_k=3,
                         run_faithfulness=True)
    path = save_experiment_config(c, tmp_path / "cfg.json")
    assert json.loads(path.read_text())["chunk_size"] == 300
    assert load_experiment_config(path) == c
