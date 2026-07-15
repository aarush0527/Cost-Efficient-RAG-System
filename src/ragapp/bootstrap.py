from __future__ import annotations

from .config import Settings, get_settings
from .embeddings import BaseEmbedder, build_embedder
from .llm import build_llm_client
from .logging_utils import QueryLogger
from .rag_service import RagService
from .vectorstore import VectorStore


def build_embedder_from_settings(settings: Settings) -> BaseEmbedder:
    return build_embedder(
        backend=settings.embedder_backend,
        model_name=settings.embedding_model_name,
        hashing_dim=settings.hashing_embedding_dim,
    )


def build_service(settings: Settings | None = None) -> RagService:
    settings = settings or get_settings()
    embedder = build_embedder_from_settings(settings)
    store = VectorStore(
        path=settings.qdrant_path, collection=settings.qdrant_collection, dimension=embedder.dimension
    )
    llm_client, is_live = build_llm_client(
        api_key=settings.groq_api_key,
        generator_model=settings.generator_model,
        judge_model=settings.judge_model,
    )
    logger = QueryLogger(settings.log_path)
    return RagService(
        store=store,
        embedder=embedder,
        llm_client=llm_client,
        llm_is_live=is_live,
        logger=logger,
        top_k_default=settings.top_k_default,
        min_relevance_score=settings.min_relevance_score,
    )
