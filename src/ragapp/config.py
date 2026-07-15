"""
Central configuration for the RAG service.

Everything that could plausibly change between environments (models, paths,
thresholds, chunk sizing) lives here and is sourced from the environment /
a local .env file - never hardcoded in application code, and never a secret
committed to git.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Groq - OpenAI-compatible API, free developer tier) ---
    groq_api_key: str | None = None
    # openai/gpt-oss-20b: fast + cheap generator. openai/gpt-oss-120b: larger
    # judge (same family, bigger - not a cross-family separation, see llm.py).
    # llama-3.3-70b-versatile/llama-3.1-8b-instant were avoided as defaults:
    # Groq announced their deprecation 2026-06-17 in favor of GPT-OSS models.
    generator_model: str = "openai/gpt-oss-20b"
    judge_model: str = "openai/gpt-oss-120b"

    # --- Vector store ---
    qdrant_path: str = "./qdrant_data"
    qdrant_collection: str = "rag_chunks"

    # --- Embeddings ---
    # "sentence-transformers": best quality, recommended for real production use.
    #     Needs internet access to Hugging Face Hub on first run.
    # "spacy": real pretrained (GloVe-style) vectors, weaker than sentence-transformers
    #     but genuinely semantic. Used to produce this project's checked-in eval
    #     numbers, since it could be installed fully offline-from-HF in the build
    #     sandbox (see README) - spaCy models are GitHub release assets, not HF Hub.
    #     Not the default - see eval/run_eval.py / README for why this run used it.
    # "hashing": deterministic, offline, zero pretrained knowledge. Test/CI fallback only.
    embedder_backend: str = "sentence-transformers"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    hashing_embedding_dim: int = 384  # used only when embedder_backend == "hashing"

    # --- Chunking ---
    chunk_size: int = 1000       # characters
    chunk_overlap: int = 200     # characters

    # --- Retrieval / generation ---
    top_k_default: int = 5
    min_relevance_score: float = 0.45  # calibrated empirically, see eval/results

    # --- Paths ---
    corpus_dir: str = "./corpus"
    log_path: str = "./logs/queries.jsonl"


def get_settings() -> Settings:
    """Fresh Settings instance (re-reads env each call; cheap, and avoids
    stale config in long-running test sessions that mutate os.environ)."""
    return Settings()
