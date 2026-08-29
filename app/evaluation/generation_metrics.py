"""Deterministic (no-LLM) scorers for a generated answer vs. the reference answer.

These are cheap, reproducible, and free. They do **not** capture meaning - "you have
30 days" and "a month is the limit" score badly against each other despite agreeing -
so they are a complement to the LLM judge, not a replacement. Where the spec says
"a simple deterministic evaluator where possible", this is it: correctness proxies.
There is deliberately no deterministic *relevance* score - lexical overlap with the
question is a bad proxy ("How many days?" -> "30 days" shares no words) and that axis
is left to the judge.

Scorers:
* ``exact_match``      - normalized string equality (0/1)
* ``token_f1`` / ``token_recall`` / ``token_precision`` - SQuAD-style bag-of-words overlap
* ``number_coverage`` - fraction of the numbers in the reference that appear in the
  answer (great for the 16 "numerical" questions in the dataset)
* ``is_abstention``    - did the answer decline to answer ("insufficient context")?
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Optional

from app.generation.prompt import INSUFFICIENT_CONTEXT_REPLY

_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
_NON_WORD = re.compile(r"[^\w\s]")
# $39.99, 30, 7-10 (captured as 7 and 10), 5%, 1,000
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace (SQuAD style)."""
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def _overlap_count(a: Sequence[str], b: Sequence[str]) -> int:
    """Size of the multiset intersection of two token lists."""
    return sum((Counter(a) & Counter(b)).values())


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def token_precision(prediction: str, reference: str) -> float:
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred:
        return 0.0
    return _overlap_count(pred, ref) / len(pred)


def token_recall(prediction: str, reference: str) -> float:
    pred, ref = _tokens(prediction), _tokens(reference)
    if not ref:
        return 0.0
    return _overlap_count(pred, ref) / len(ref)


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall. 0.0 if either is 0."""
    precision = token_precision(prediction, reference)
    recall = token_recall(prediction, reference)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _numbers(text: str) -> set[str]:
    return {match.group().replace(",", "") for match in _NUMBER.finditer(text)}


def number_coverage(prediction: str, reference: str) -> Optional[float]:
    """Fraction of the reference's numbers that also appear in the prediction.

    Returns ``None`` when the reference contains no numbers (nothing to check).
    Matching is digit-boundary aware so "2" does not match inside "2024".
    """
    expected = _numbers(reference)
    if not expected:
        return None
    hits = sum(
        1
        for number in expected
        if re.search(rf"(?<!\d){re.escape(number)}(?!\d)", prediction)
    )
    return hits / len(expected)


def is_abstention(text: str) -> bool:
    """True if the answer declines to answer for lack of context."""
    norm = normalize_answer(text)
    canonical = normalize_answer(INSUFFICIENT_CONTEXT_REPLY)
    return canonical in norm or ("insufficient" in norm and "context" in norm)
