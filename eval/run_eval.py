"""
Runs the full three-layer evaluation (retrieval / answer / latency) against
the fixed eval/eval_dataset.json question set and writes results to
eval/results/.

Embedder note: this run uses EMBEDDER_BACKEND=spacy, not the recommended
production default (sentence-transformers/bge-small-en-v1.5). The build
sandbox this project was created in does not have network access to Hugging
Face Hub (verified: a direct request to huggingface.co returns 403 from the
sandbox's egress proxy) but does have access to GitHub, which is where
spaCy's pretrained models are hosted - see ragapp/embeddings.py. Re-run this
script with EMBEDDER_BACKEND=sentence-transformers on a machine with normal
internet access to reproduce these numbers with the stronger, recommended
embedder; the pipeline code is identical either way.

LLM note: GROQ_API_KEY is not set in the build sandbox (api.groq.com is not
reachable from it either - see ragapp/llm.py), so generation and judging use
StubLLMClient (see ragapp/llm.py) - clearly labelled in the output. Retrieval
metrics are unaffected (they don't depend on the LLM at all). Answer-quality
numbers (faithfulness/relevance/EM/F1) below reflect the stub's heuristics,
not a real Groq model - re-run with a real GROQ_API_KEY for those.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("EMBEDDER_BACKEND", "spacy")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "en_core_web_md")
os.environ.setdefault("MIN_RELEVANCE_SCORE", "0.892")
os.environ.setdefault("QDRANT_PATH", "./eval/qdrant_eval_data")
os.environ.setdefault("LOG_PATH", "./eval/results/eval_queries.jsonl")

from src.ragapp.bootstrap import build_service  # noqa: E402
from src.ragapp.config import get_settings  # noqa: E402
from src.ragapp.ingest import ingest_corpus  # noqa: E402
from eval.metrics import (  # noqa: E402
    average_precision_at_k,
    exact_match,
    f1_score,
    hit_rate,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from eval.resolve_gold import resolve_gold_chunk_ids  # noqa: E402

RESULTS_DIR = Path("eval/results")
DATASET_PATH = Path("eval/eval_dataset.json")
K_VALUES = [3, 5, 10]
PRIMARY_K = 5


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(pct / 100 * (len(values) - 1))))
    return values[idx]


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    service = build_service(settings)

    print(f"embedder: {service.embedder.model_name} (dim={service.embedder.dimension})")
    print(f"llm_is_live: {service.llm_is_live}")
    print(f"min_relevance_score: {service.min_relevance_score}")

    report = ingest_corpus(
        settings.corpus_dir, service.store, service.embedder, settings.chunk_size, settings.chunk_overlap
    )
    print(f"ingested: {report.total_chunks} chunks total, {report.total_embedded} embedded this run")

    dataset = json.loads(DATASET_PATH.read_text())
    gold_map = resolve_gold_chunk_ids(service.store, dataset)

    retrieval_per_question = []
    max_k = max(K_VALUES)
    for item in dataset:
        qid = item["id"]
        gold_ids = set(gold_map[qid])
        q_vec = service.embedder.embed_one(item["question"])
        results = service.store.search(q_vec, top_k=max_k)
        ranked_ids = [r.id for r in results]

        row = {"id": qid, "question": item["question"], "is_answerable": item["is_answerable"],
               "gold_ids": list(gold_ids), "ranked_ids": ranked_ids,
               "scores": [r.score for r in results]}

        if item["is_answerable"]:
            for k in K_VALUES:
                top_k_ids = ranked_ids[:k]
                row[f"hit_rate@{k}"] = hit_rate(top_k_ids, gold_ids)
                row[f"recall@{k}"] = recall_at_k(top_k_ids, gold_ids)
                row[f"mrr@{k}"] = mrr(top_k_ids, gold_ids)
                row[f"ndcg@{k}"] = ndcg_at_k(top_k_ids, gold_ids)
                row[f"context_precision(ap)@{k}"] = average_precision_at_k(top_k_ids, gold_ids)

        retrieval_per_question.append(row)

    def _agg(metric_key: str) -> float:
        vals = [r[metric_key] for r in retrieval_per_question if r["is_answerable"]]
        return sum(vals) / len(vals) if vals else 0.0

    retrieval_summary = {}
    for k in K_VALUES:
        retrieval_summary[f"k={k}"] = {
            "hit_rate": _agg(f"hit_rate@{k}"),
            "recall": _agg(f"recall@{k}"),
            "mrr": _agg(f"mrr@{k}"),
            "ndcg": _agg(f"ndcg@{k}"),
            "context_precision_ap": _agg(f"context_precision(ap)@{k}"),
        }

    (RESULTS_DIR / "retrieval_metrics.json").write_text(
        json.dumps({"summary": retrieval_summary, "per_question": retrieval_per_question}, indent=2)
    )
    print("wrote eval/results/retrieval_metrics.json")

    #full pipeline: answer quality, refusal behavior, latency
    answer_rows = []
    for item in dataset:
        result = service.query(item["question"], top_k=PRIMARY_K)
        row = {
            "id": item["id"],
            "question": item["question"],
            "is_answerable": item["is_answerable"],
            "answer": result.answer,
            "has_sufficient_context": result.has_sufficient_context,
            "cited_chunk_ids": result.cited_chunk_ids,
            "embed_latency_ms": result.embed_latency_ms,
            "retrieval_latency_ms": result.retrieval_latency_ms,
            "generation_latency_ms": result.generation_latency_ms,
            "total_latency_ms": result.total_latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

        if item["is_answerable"]:
            row["gold_answer"] = item["gold_answer"]
            row["exact_match"] = exact_match(result.answer, item["gold_answer"])
            row["f1"] = f1_score(result.answer, item["gold_answer"])

            if result.has_sufficient_context:
                from src.ragapp.llm import RetrievedChunk
                chunks = [
                    RetrievedChunk(chunk_id=r.chunk_id, text=r.text, source_file=r.source_file, score=r.score)
                    for r in result.retrieved
                ]
                judge = service.llm_client.judge_answer(item["question"], chunks, result.answer)
                row["faithfulness_score"] = judge.faithfulness_score
                row["faithfulness_rationale"] = judge.faithfulness_rationale
                row["relevance_score"] = judge.relevance_score
                row["relevance_rationale"] = judge.relevance_rationale

        answer_rows.append(row)

    answerable_rows = [r for r in answer_rows if r["is_answerable"]]
    unanswerable_rows = [r for r in answer_rows if not r["is_answerable"]]

    correctly_answered = sum(1 for r in answerable_rows if r["has_sufficient_context"])
    false_refusals = len(answerable_rows) - correctly_answered
    correctly_refused = sum(1 for r in unanswerable_rows if not r["has_sufficient_context"])
    false_answers_on_unanswerable = len(unanswerable_rows) - correctly_refused

    def _mean(key, rows):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    answer_summary = {
        "n_answerable": len(answerable_rows),
        "n_unanswerable": len(unanswerable_rows),
        "answerable_answered_rate": correctly_answered / len(answerable_rows) if answerable_rows else 0.0,
        "answerable_false_refusal_count": false_refusals,
        "unanswerable_correctly_refused_rate": correctly_refused / len(unanswerable_rows) if unanswerable_rows else 0.0,
        "unanswerable_hallucinated_answer_count": false_answers_on_unanswerable,
        "mean_exact_match": _mean("exact_match", answerable_rows),
        "mean_f1": _mean("f1", answerable_rows),
        "mean_faithfulness": _mean("faithfulness_score", answerable_rows),
        "mean_relevance": _mean("relevance_score", answerable_rows),
        "llm_is_live": service.llm_is_live,
    }

    all_latencies = {
        "embed_ms": [r["embed_latency_ms"] for r in answer_rows],
        "retrieval_ms": [r["retrieval_latency_ms"] for r in answer_rows],
        "generation_ms": [r["generation_latency_ms"] for r in answer_rows],
        "total_ms": [r["total_latency_ms"] for r in answer_rows],
    }
    latency_summary = {
        stage: {
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
            "mean": statistics.mean(vals) if vals else 0.0,
        }
        for stage, vals in all_latencies.items()
    }

    (RESULTS_DIR / "answer_metrics.json").write_text(
        json.dumps(
            {"summary": answer_summary, "latency": latency_summary, "per_question": answer_rows}, indent=2
        )
    )
    print("wrote eval/results/answer_metrics.json")

    print(json.dumps({"retrieval_summary": retrieval_summary["k=5"], "answer_summary": answer_summary,
                       "latency_summary": latency_summary}, indent=2))


if __name__ == "__main__":
    run()
