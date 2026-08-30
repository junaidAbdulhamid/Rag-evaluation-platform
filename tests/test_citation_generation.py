"""Tests for citation-grounded generation: marker parsing + the generator."""

from app.generation.citation import (
    AnthropicCitedGenerator,
    build_cited_answer,
    parse_citations,
)
from app.models import Chunk, RetrievedChunk
from tests.fakes import FakeTextLLM


def ctx(doc_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"{doc_id}::c{rank}", document_id=doc_id, text=f"text {rank}"),
        score=1.0,
        rank=rank,
    )


RETRIEVED = [ctx("refund_policy", 1), ctx("payment_methods", 2)]


def test_parse_citations_resolves_markers_to_chunks():
    cites = parse_citations("Refunds take 30 days. [1] Paid back to card. [2]", RETRIEVED)

    assert [c.marker for c in cites] == [1, 2]
    assert cites[0].exists is True
    assert cites[0].chunk_id == "refund_policy::c1"
    assert cites[1].document_id == "payment_methods"


def test_parse_citations_flags_out_of_range_marker_as_nonexistent():
    cites = parse_citations("Some claim. [5]", RETRIEVED)

    assert cites[0].marker == 5
    assert cites[0].exists is False
    assert cites[0].chunk_id is None


def test_parse_citations_dedupes_and_sorts():
    cites = parse_citations("a [2] b [1] c [2]", RETRIEVED)
    assert [c.marker for c in cites] == [1, 2]


def test_parse_citations_none_present():
    assert parse_citations("No markers here at all.", RETRIEVED) == []


def test_generator_returns_cited_answer_with_resolved_citations():
    llm = FakeTextLLM("Refunds take 30 days. [1]\n\nSources:\n[1] refund_policy")
    answer = AnthropicCitedGenerator(llm=llm).generate("How long for a refund?", RETRIEVED)

    assert answer.answer.startswith("Refunds take 30 days. [1]")
    assert [c.marker for c in answer.citations] == [1]
    assert answer.citations[0].chunk_id == "refund_policy::c1"
    assert "citation markers" in llm.calls[0]  # the cited prompt was used


def test_build_cited_answer_strips_and_parses():
    result = build_cited_answer("  Claim. [1]  ", "m", None, "prompt", RETRIEVED)
    assert result.answer == "Claim. [1]"
    assert result.citations[0].marker == 1
