"""Retrieval metrics, implemented from scratch (no RAGAS / no eval libraries).

Every function here is **pure**: it takes a ranked list of document ids and a set of
relevant document ids, and returns a number. No models, no I/O. That makes the
edge-case unit tests trivial and keeps the maths honest and inspectable - which is
the whole point of building these by hand.

## Conventions used throughout

* ``retrieved_doc_ids`` is a **de-duplicated, rank-ordered** list of document ids:
  index 0 is the top hit. The glue layer (``retrieval.py``) produces it from the
  retriever's chunk output by collapsing multiple chunks of the same document to
  that document's best rank.
* Relevance is **binary**: a document is relevant or it is not. We have no graded
  relevance labels, so every gain is 0 or 1.
* ``k`` is the cut-off. Metrics look at ``retrieved_doc_ids[:k]`` only.
* When there are **no relevant documents** for a question (an "unanswerable" item),
  ``recall_at_k`` and ``ndcg_at_k`` return ``None`` - they are genuinely undefined
  (0/0, and there is no ideal ranking). ``precision``, ``hit_rate`` and
  ``reciprocal_rank`` stay defined and return ``0.0``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import Optional


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be a positive integer")


def hit_rate_at_k(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Collection[str], k: int
) -> float:
    """1.0 if at least one relevant document is in the top ``k``, else 0.0.

    The coarsest possible retrieval signal: "did we get *anything* useful in front
    of the reader?" Averaged over a dataset it becomes "hit rate".
    """
    _check_k(k)
    relevant = set(relevant_doc_ids)
    return 1.0 if any(doc_id in relevant for doc_id in retrieved_doc_ids[:k]) else 0.0


def precision_at_k(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Collection[str], k: int
) -> float:
    """(relevant documents in the top k) / (documents actually retrieved in the top k).

        precision@k = |retrieved_topk  ∩  relevant|  /  |retrieved_topk|

    The denominator is the number of documents *actually* returned (``min(k, n)``),
    not ``k`` itself - so retrieving 2 relevant docs when only 2 exist scores 1.0,
    not 0.4. Returns 0.0 when nothing was retrieved.
    """
    _check_k(k)
    relevant = set(relevant_doc_ids)
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(top_k)


def recall_at_k(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Collection[str], k: int
) -> Optional[float]:
    """(distinct relevant documents in the top k) / (total relevant documents).

        recall@k = |retrieved_topk  ∩  relevant|  /  |relevant|

    "Of everything we were supposed to find, how much did we surface?" Returns
    ``None`` when the question has no relevant documents (undefined).
    """
    _check_k(k)
    relevant = set(relevant_doc_ids)
    if not relevant:
        return None
    found = set(retrieved_doc_ids[:k]) & relevant
    return len(found) / len(relevant)


def reciprocal_rank(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Collection[str]
) -> float:
    """1 / (rank of the first relevant document), or 0.0 if none is retrieved.

    Rank is 1-based, so a relevant doc at the very top scores 1.0, second place
    0.5, third 0.333, ... The dataset-level mean of this value is **MRR** (Mean
    Reciprocal Rank). It answers "how far does the reader have to scan before the
    first useful hit?" and rewards getting *one* good result to the top.
    """
    relevant = set(relevant_doc_ids)
    for index, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant:
            return 1.0 / (index + 1)
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain over the first ``k`` positions.

        DCG@k = Σ_{i=1..k}  gain_i / log2(i + 1)

    (``i`` is the 1-based rank.) The ``log2(i + 1)`` term is the *discount*: a hit
    at rank 1 is divided by log2(2) = 1 (no discount), rank 2 by log2(3) ≈ 1.585,
    rank 3 by log2(4) = 2, and so on - so the same relevant document is worth less
    the further down it sits. With binary gains, ``gain_i`` is 1 for a relevant
    document and 0 otherwise.

    Implementation note: enumerating from 0, position ``i`` (0-based) has 1-based
    rank ``i + 1``, so the divisor is ``log2((i + 1) + 1) == log2(i + 2)``.
    """
    _check_k(k)
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains[:k]))


def ndcg_at_k(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Collection[str], k: int
) -> Optional[float]:
    """Normalized DCG@k - DCG divided by the best DCG achievable for this question.

        NDCG@k = DCG@k(actual ranking) / DCG@k(ideal ranking)

    The **ideal ranking** puts every relevant document first, so with ``r`` relevant
    documents and cut-off ``k`` the ideal gain vector is ``[1] * min(k, r)``. That
    normalisation makes NDCG comparable across questions that have different numbers
    of relevant documents, and bounds it to ``[0.0, 1.0]``:

    * 1.0  -> every relevant doc is packed at the top, in the first ``k`` slots
    * 0.0  -> no relevant doc appears in the top ``k``
    * between -> relevant docs are present but pushed down the ranking

    Unlike hit_rate / precision / recall, NDCG is **rank-sensitive**: moving a
    relevant document from rank 3 to rank 1 raises the score even though the set of
    retrieved documents is unchanged.

    Returns ``None`` when the question has no relevant documents (no ideal ranking
    exists).
    """
    _check_k(k)
    relevant = set(relevant_doc_ids)
    if not relevant:
        return None

    actual_gains = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved_doc_ids[:k]]
    ideal_gains = [1.0] * min(k, len(relevant))

    dcg = dcg_at_k(actual_gains, k)
    idcg = dcg_at_k(ideal_gains, k)
    return dcg / idcg if idcg > 0 else 0.0
