# Cost Efficient RAG System

**A retrieval-augmented QA service that costs pennies to run — self-hosted embedded vector search, free-tier LLM inference, and a real evaluation harness to prove it.**

Most RAG stacks default to a managed vector DB billed by always-on capacity, whether or not anyone's querying it. This project asks a simple question — *does a large-but-lightly-queried knowledge base actually need that?* — and answers it with a working service, real benchmarks, and a cost model instead of a hand-wavy blog post.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Vector Store](https://img.shields.io/badge/vector%20store-Qdrant%20(embedded)-orange)
![LLM](https://img.shields.io/badge/inference-Groq-red)

---

## Why this exists

- **No always-on infra.** Qdrant runs embedded (`QdrantClient(path=...)`) — no server, no Docker, no idle bill.
- **No paid LLM key required.** Generation and judging run on [Groq](https://console.groq.com)'s free developer tier via its OpenAI-compatible tool-calling API.
- **Idempotent by design.** Chunk IDs are content-derived hashes — re-ingesting an unchanged corpus is a true no-op (zero re-embeds), edits only touch what changed, and deleted files clean up their own vectors.
- **Grounded, not creative.** Answers cite the exact chunks they used and refuse to guess when retrieval comes up empty — verified with a fixed set of deliberately unanswerable questions, not just asserted.
- **Evaluated like it matters.** Real IR metrics (Hit Rate, Recall@k, MRR, nDCG@k, Average Precision) unit-tested against hand-computed cases, plus a from-scratch, auditable cost model — not a table typed by hand.

## Features

- Ingests PDF, HTML, and Markdown with a deterministic, header-aware recursive chunker
- Configurable chunk size/overlap, top-k, and metadata filtering (e.g. restrict retrieval to a doc category)
- FastAPI HTTP service (`/query`, `/ingest`, `/health`, `/stats`) + a CLI
- Per-query structured logging: latency breakdown, chunk count, token usage
- Three swappable embedding backends (production-grade `sentence-transformers`, offline-friendly `spaCy`, and a deterministic zero-dependency fallback for tests/CI)
- Full evaluation harness: retrieval metrics, answer faithfulness/relevance (LLM-as-judge), EM/F1, latency percentiles, and a scaled cost projection (100K → 10M vectors)

## Quick start

```bash
git clone <your-repo-url> && cd frugal-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# set GROQ_API_KEY in .env — free at https://console.groq.com

python -m ragapp.cli ingest
python -m ragapp.cli query "How many times is data replicated?"
python -m ragapp.cli serve   # → http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "What is the per-GB storage price on the Pro tier?"}'
```

## Configuration

All config is environment-driven (`.env`, see `.env.example`) — nothing hardcoded.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(none)* | Enables real generation/judging; falls back to a labelled stub without it |
| `GENERATOR_MODEL` | `openai/gpt-oss-20b` | Fast, cheap generation model |
| `JUDGE_MODEL` | `openai/gpt-oss-120b` | Larger model for answer-quality scoring |
| `EMBEDDER_BACKEND` | `sentence-transformers` | `sentence-transformers` \| `spacy` \| `hashing` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Characters, recursive header/paragraph/sentence splitter |
| `TOP_K_DEFAULT` | `5` | Retrieval depth |
| `MIN_RELEVANCE_SCORE` | `0.45` | Similarity floor below which the service refuses rather than guesses |

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/query` | POST | `{question, top_k?, metadata_filter?}` → grounded, cited answer |
| `/ingest` | POST | (Re)ingest the corpus directory, idempotently |
| `/health` | GET | Liveness + active embedder/LLM mode |
| `/stats` | GET | Vector count, embedding model, dimensionality |

## Benchmarks

Measured against a fixed 22-question set (18 answerable, 4 deliberately not, to test refusal honestly) — full breakdown and methodology in [`eval/results/`](./eval/results):

| Retrieval @k=5 | Score |
|---|---|
| Hit Rate | 0.83 |
| Recall | 0.79 |
| MRR | 0.64 |
| nDCG | 0.66 |
| Context precision (AP) | 0.60 |

| Refusal behavior | Result |
|---|---|
| Answerable questions correctly answered | 18/18 |
| Unanswerable questions correctly refused | 3/4 |

| Latency (embed + retrieve, no LLM call) | p50 | p95 |
|---|---|---|
| Total | 7.2 ms | 8.3 ms |

**Cost** — self-hosted embedded Qdrant vs. a managed alternative, 384-dim vectors, light query volume:

| Vectors | Self-hosted (this project) | Managed (serverless, realistic bill) |
|---|---|---|
| 100K | **$18/mo** | $0 (free tier) |
| 1M | **$18/mo** | $50/mo |
| 10M | **$21/mo** | $50/mo |

Self-hosted cost stays essentially flat across three orders of magnitude because a small fixed VM dominates the bill and disk is cheap — see [`eval/results/cost_latency.md`](./eval/results/cost_latency.md) for cited sources and every assumption spelled out.

## Design notes

- **Qdrant, embedded** over pgvector/ChromaDB/LanceDB/FAISS/sqlite-vec: a real engine with a native filter DSL and upsert-by-ID (not a bare index you'd have to hand-roll metadata handling around), and the same engine Qdrant Cloud runs — so a managed-vs-self-hosted comparison isolates the actual hosting premium instead of conflating it with a software choice.
- **Groq** for inference: OpenAI-compatible tool-calling, so structured output (an answer with citations, or a judge's rubric score) is forced via schema rather than hoped for from prompted JSON.
- **Anchor-snippet gold resolution**: eval ground truth isn't a hardcoded list of chunk IDs (those are content hashes and shift if chunking parameters change) — each eval question carries a short, unique text snippet, resolved to whichever chunk(s) currently contain it. Robust to re-chunking, and multi-document facts naturally produce multi-gold test cases instead of being special-cased away.
