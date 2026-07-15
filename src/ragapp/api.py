"""
FastAPI application.

    POST /ingest   - (re)ingest the configured corpus directory, idempotently
    POST /query    - ask a question, get a grounded + cited answer
    GET  /health   - liveness + which LLM/embedder mode is active
    GET  /stats    - collection stats (vector count, model, dimensionality)

Run with: uvicorn ragapp.api:app --reload  (see README)
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .bootstrap import build_service
from .config import get_settings
from .ingest import ingest_corpus

app = FastAPI(title="Cost-efficient RAG service", version="0.1.0")

_settings = get_settings()
_service = build_service(_settings)


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, description="Defaults to TOP_K_DEFAULT if omitted.")
    metadata_filter: dict[str, Any] | None = Field(
        default=None, description='e.g. {"doc_category": "pricing"}'
    )


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    source_file: str
    section_path: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    cited_chunk_ids: list[str]
    has_sufficient_context: bool
    retrieved: list[RetrievedChunkOut]
    embed_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    llm_is_live: bool


class IngestResponse(BaseModel):
    total_chunks: int
    total_embedded: int
    total_skipped_unchanged: int
    total_deleted_stale: int
    embedding_model: str
    embedding_dimension: int
    per_source: list[dict]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_is_live": _service.llm_is_live,
        "embedder_backend": _settings.embedder_backend,
        "embedding_model": _service.embedder.model_name,
        "embedding_dimension": _service.embedder.dimension,
    }


@app.get("/stats")
def stats():
    return {
        "vector_count": _service.store.count(),
        "collection": _settings.qdrant_collection,
        "embedding_model": _service.embedder.model_name,
        "embedding_dimension": _service.embedder.dimension,
        "top_k_default": _service.top_k_default,
        "min_relevance_score": _service.min_relevance_score,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    report = ingest_corpus(
        corpus_dir=_settings.corpus_dir,
        store=_service.store,
        embedder=_service.embedder,
        chunk_size=_settings.chunk_size,
        chunk_overlap=_settings.chunk_overlap,
    )
    return IngestResponse(
        total_chunks=report.total_chunks,
        total_embedded=report.total_embedded,
        total_skipped_unchanged=report.total_skipped,
        total_deleted_stale=report.total_deleted,
        embedding_model=report.embedding_model,
        embedding_dimension=report.embedding_dimension,
        per_source=[vars(s) for s in report.per_source],
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = _service.query(req.question, top_k=req.top_k, metadata_filter=req.metadata_filter)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        cited_chunk_ids=result.cited_chunk_ids,
        has_sufficient_context=result.has_sufficient_context,
        retrieved=[
            RetrievedChunkOut(
                chunk_id=r.chunk_id, source_file=r.source_file, section_path=r.section_path,
                score=r.score, text=r.text,
            )
            for r in result.retrieved
        ],
        embed_latency_ms=result.embed_latency_ms,
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        total_latency_ms=result.total_latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        llm_is_live=result.llm_is_live,
    )
