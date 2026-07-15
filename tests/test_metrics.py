import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.metrics import (
    average_precision_at_k,
    exact_match,
    f1_score,
    hit_rate,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

# Hand-computed toy case, worked out on paper:
#   gold = {A, B}, retrieved = [X, A, B]
#   DCG   = 0/log2(2) + 1/log2(3) + 1/log2(4) = 0 + 0.63093 + 0.5      = 1.13093
#   IDCG  = 1/log2(2) + 1/log2(3)             = 1 + 0.63093            = 1.63093
#   nDCG  = 1.13093 / 1.63093                                         ≈ 0.69340
#   MRR   = 1 / 2 (first hit at rank 2)                               = 0.5
#   AP    = (0.5 + 0.66667) / min(2,3)                                ≈ 0.58333
#   Recall@3 = 2/2 = 1.0 ; HitRate = 1.0


def test_toy_case_matches_hand_computation():
    retrieved = ["X", "A", "B"]
    gold = {"A", "B"}

    assert hit_rate(retrieved, gold) == 1.0
    assert recall_at_k(retrieved, gold) == 1.0
    assert abs(mrr(retrieved, gold) - 0.5) < 1e-9
    assert abs(ndcg_at_k(retrieved, gold) - 0.69340) < 1e-4
    assert abs(average_precision_at_k(retrieved, gold) - 0.58333) < 1e-4


def test_perfect_ranking_gives_perfect_scores():
    retrieved = ["A", "B", "X"]
    gold = {"A", "B"}
    assert hit_rate(retrieved, gold) == 1.0
    assert recall_at_k(retrieved, gold) == 1.0
    assert mrr(retrieved, gold) == 1.0
    assert abs(ndcg_at_k(retrieved, gold) - 1.0) < 1e-9
    assert abs(average_precision_at_k(retrieved, gold) - 1.0) < 1e-9


def test_no_hits_gives_zero_everywhere():
    retrieved = ["X", "Y", "Z"]
    gold = {"A", "B"}
    assert hit_rate(retrieved, gold) == 0.0
    assert recall_at_k(retrieved, gold) == 0.0
    assert mrr(retrieved, gold) == 0.0
    assert ndcg_at_k(retrieved, gold) == 0.0
    assert average_precision_at_k(retrieved, gold) == 0.0


def test_recall_with_partial_gold_coverage():
    retrieved = ["A", "X", "Y"]
    gold = {"A", "B"}  
    assert recall_at_k(retrieved, gold) == 0.5
    assert hit_rate(retrieved, gold) == 1.0


def test_em_and_f1_basic():
    assert exact_match("The answer is 400 days.", "400 days") == 0.0  
    assert exact_match("400 days", "400 Days") == 1.0  
    assert exact_match("a 12 hours", "12 hours") == 1.0  
    assert f1_score("400 days", "400 days") == 1.0
    assert f1_score("completely unrelated text", "400 days") == 0.0
    f1 = f1_score("the retention is 400 days total", "400 days")
    assert 0.0 < f1 < 1.0  


def test_f1_handles_empty_prediction():
    assert f1_score("", "400 days") == 0.0
    assert f1_score("", "") == 1.0
