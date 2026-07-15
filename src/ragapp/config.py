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

    groq_api_key: str | None = None
 
    generator_model: str = "openai/gpt-oss-20b"
    judge_model: str = "openai/gpt-oss-120b"


    qdrant_path: str = "./qdrant_data"
    qdrant_collection: str = "rag_chunks"


    embedder_backend: str = "sentence-transformers"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    hashing_embedding_dim: int = 384  

    chunk_size: int = 1000       
    chunk_overlap: int = 200     

    top_k_default: int = 5
    min_relevance_score: float = 0.45  

    corpus_dir: str = "./corpus"
    log_path: str = "./logs/queries.jsonl"


def get_settings() -> Settings:
    return Settings()
