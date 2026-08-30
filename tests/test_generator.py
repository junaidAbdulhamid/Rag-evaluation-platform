"""AnthropicGenerator now composes a TextLLM, so it can be unit-tested with a fake."""

from app.generation.generator import AnthropicGenerator
from app.generation.prompt import INSUFFICIENT_CONTEXT_REPLY, build_rag_prompt
from app.llm import _supports_effort
from app.models import Chunk, RetrievedChunk
from tests.fakes import FakeTextLLM


def test_supports_effort_only_for_capable_models():
    assert _supports_effort("claude-opus-5") is True
    assert _supports_effort("claude-sonnet-5") is True
    assert _supports_effort("claude-opus-4-6") is True
    assert _supports_effort("claude-haiku-4-5") is False
    assert _supports_effort("claude-sonnet-4-5") is False


def chunk(doc_id: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"{doc_id}::0", document_id=doc_id, text=text),
        score=1.0,
        rank=rank,
    )


def test_generator_builds_the_rag_prompt_and_wraps_the_llm_response():
    llm = FakeTextLLM("Customers have 30 days.")
    generator = AnthropicGenerator(llm=llm)

    retrieved = [chunk("refund_policy", "Refunds within 30 days of delivery.", 1)]
    answer = generator.generate("How many days for a refund?", retrieved)

    assert answer.answer == "Customers have 30 days."
    assert answer.model == "fake-text-llm"
    assert answer.token_usage.total_tokens == 2
    # the prompt actually sent to the LLM is the RAG prompt, and it is echoed back
    assert answer.prompt == build_rag_prompt("How many days for a refund?", retrieved)
    assert "Refunds within 30 days" in llm.calls[0]


def test_generator_passes_through_an_abstention():
    llm = FakeTextLLM(INSUFFICIENT_CONTEXT_REPLY)
    answer = AnthropicGenerator(llm=llm).generate("Do you price match?", [])
    assert answer.answer == INSUFFICIENT_CONTEXT_REPLY
