"""
Retrieval metrics (binary relevance, since our gold labels are simply
"relevant" / "not relevant" - resolved via anchor-snippet containment, see
resolve_gold.py) and answer-string metrics (EM/F1, SQuAD-style).

All retrieval metrics take:
    retrieved_ids: list[str]   - ranked chunk ids returned by the retriever (len <= k)
    gold_ids:      set[str]    - chunk ids considered relevant for this question
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter


def hit_rate(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    return 1.0 if any(rid in gold_ids for rid in retrieved_ids) else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    hits = len(set(retrieved_ids) & gold_ids)
    return hits / len(gold_ids)


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """Binary-relevance nDCG@k. IDCG assumes the ideal ranking places
    min(|gold|, k) relevant items at the very top."""
    k = len(retrieved_ids)
    dcg = sum(
        (1.0 if rid in gold_ids else 0.0) / math.log2(i + 1)
        for i, rid in enumerate(retrieved_ids, start=1)
    )
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """Context precision, implemented as order-aware Average Precision@k
    (this is the same idea RAGAS calls "context precision": precision is
    rewarded more for relevant chunks appearing earlier in the ranking)."""
    k = len(retrieved_ids)
    if not gold_ids or k == 0:
        return 0.0
    hits_so_far = 0
    precision_sum = 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        is_rel = rid in gold_ids
        if is_rel:
            hits_so_far += 1
            precision_sum += hits_so_far / i
    denom = min(len(gold_ids), k)
    return precision_sum / denom if denom > 0 else 0.0


# --- answer string metrics (SQuAD-style) ---

_ARTICLES = {"a", "an", "the"}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
