"""
CLI entry points, mainly so ingestion/querying can be scripted without going
over HTTP - the eval harness uses these same underlying functions directly
for speed rather than round-tripping through FastAPI.

Usage:
    python -m ragapp.cli ingest
    python -m ragapp.cli query "How many times is data replicated?"
    python -m ragapp.cli serve
"""
from __future__ import annotations

import argparse
import json
import sys

from .bootstrap import build_service
from .config import get_settings
from .ingest import ingest_corpus


def cmd_ingest(args):
    settings = get_settings()
    service = build_service(settings)
    report = ingest_corpus(
        corpus_dir=settings.corpus_dir,
        store=service.store,
        embedder=service.embedder,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(f"embedding model: {report.embedding_model} (dim={report.embedding_dimension})")
    print(f"chunks total={report.total_chunks} embedded={report.total_embedded} "
          f"skipped_unchanged={report.total_skipped} deleted_stale={report.total_deleted}")
    for s in report.per_source:
        print(f"  - {s.source_file}: total={s.chunks_total} embedded={s.chunks_embedded} "
              f"skipped={s.chunks_skipped_unchanged} deleted={s.chunks_deleted_stale}")


def cmd_query(args):
    settings = get_settings()
    service = build_service(settings)
    if not service.llm_is_live:
        print("[warning] GROQ_API_KEY not set - using StubLLMClient (not a real answer)", file=sys.stderr)
    result = service.query(args.question, top_k=args.top_k)
    print(json.dumps({
        "question": result.question,
        "answer": result.answer,
        "cited_chunk_ids": result.cited_chunk_ids,
        "has_sufficient_context": result.has_sufficient_context,
        "retrieved": [{"chunk_id": r.chunk_id, "source_file": r.source_file, "score": r.score} for r in result.retrieved],
        "total_latency_ms": round(result.total_latency_ms, 1),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }, indent=2))


def cmd_serve(args):
    import uvicorn
    uvicorn.run("ragapp.api:app", host="0.0.0.0", port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(prog="ragapp")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="(re)ingest the configured corpus directory")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="ask a question")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", type=int, default=None)
    p_query.set_defaults(func=cmd_query)

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
