import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_temp_project():
    tmp = tempfile.mkdtemp()
    corpus_dir = Path(tmp) / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "widgets.md").write_text(
        "# Widgets\n\n"
        "NimbusWidget devices are manufactured in three factories and ship within two business days.\n\n"
        "# Gadgets\n\n"
        "NimbusGadget accessories are sold only as bundles with a NimbusWidget device."
    )
    return tmp, corpus_dir


def test_ingest_then_query_with_context_and_without(monkeypatch):
    tmp, corpus_dir = _make_temp_project()
    try:
        monkeypatch.setenv("EMBEDDER_BACKEND", "hashing")
        monkeypatch.setenv("CORPUS_DIR", str(corpus_dir))
        monkeypatch.setenv("QDRANT_PATH", str(Path(tmp) / "qdrant"))
        monkeypatch.setenv("LOG_PATH", str(Path(tmp) / "logs" / "queries.jsonl"))
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        # Low threshold: the hashing embedder is a crude bag-of-ngrams model,
        # not semantic, so its absolute similarity scores run lower than a
        # real embedding model's - this only needs to demonstrate the gate
        # logic works, not match production calibration (see eval/results
        # for the real, semantically-calibrated threshold).
        monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.05")

        # import (only once - the module builds its service at import time using
        # a QdrantClient file lock on QDRANT_PATH, so re-importing/reloading in
        # the same process would try to open that lock twice and fail)
        from src.ragapp import api as api_module

        from fastapi.testclient import TestClient
        client = TestClient(api_module.app)

        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["llm_is_live"] is False

        ingest_resp = client.post("/ingest").json()
        assert ingest_resp["total_chunks"] >= 2
        assert ingest_resp["total_embedded"] >= 2

        # re-ingest should be a no-op
        ingest_resp2 = client.post("/ingest").json()
        assert ingest_resp2["total_embedded"] == 0

        stats = client.get("/stats").json()
        assert stats["vector_count"] == ingest_resp["total_chunks"]

        q1 = client.post("/query", json={"question": "How many factories manufacture NimbusWidget devices?"}).json()
        assert q1["has_sufficient_context"] is True
        assert len(q1["retrieved"]) > 0
        assert q1["cited_chunk_ids"], "a grounded answer should cite at least one chunk"
        assert all(r["text"].strip() for r in q1["retrieved"]), "retrieved chunks must carry their actual text"

        # metadata filter: restrict to a doc_category that doesn't exist -> no results -> no context
        q2 = client.post("/query", json={
            "question": "How many factories manufacture NimbusWidget devices?",
            "metadata_filter": {"doc_category": "nonexistent-category"},
        }).json()
        assert q2["has_sufficient_context"] is False
        assert q2["retrieved"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
