"""
The core query pipeline: embed -> retrieve (top-k + optional metadata filter)
-> relevance gate -> generate (grounded, cited) -> log.

Two independent lines of defense against hallucinated "no context" answers:
  1. A pre-generation score threshold (`min_relevance_score`): if the best
     retrieved chunk scores below this, we skip the LLM call entirely -
     cheaper, faster, and structurally can't hallucinate since no generation
     happens.
  2. The model's own `has_sufficient_context` self-assessment: even when
     retrieval clears the threshold, the *specific* chunks retrieved might
     not actually answer *this* question, and the model is instructed to
     say so rather than guess.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .embeddings import BaseEmbedder
from .llm import AnswerResult, BaseLLMClient, RetrievedChunk
from .logging_utils import QueryLogger
from .vectorstore import VectorStore

NO_CONTEXT_MESSAGE = "I don't have enough information in the provided documents to answer this question."


@dataclass
class RetrievedInfo:
    chunk_id: str
    source_file: str
    section_path: str
    score: float
    text: str = ""


@dataclass
class QueryResult:
    question: str
    answer: str
    cited_chunk_ids: list[str]
    has_sufficient_context: bool
    retrieved: list[RetrievedInfo] = field(default_factory=list)
    embed_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_is_live: bool = False


class RagService:
    def __init__(
        self,
        store: VectorStore,
        embedder: BaseEmbedder,
        llm_client: BaseLLMClient,
        llm_is_live: bool,
        logger: QueryLogger,
        top_k_default: int = 5,
        min_relevance_score: float = 0.45,
    ):
        self.store = store
        self.embedder = embedder
        self.llm_client = llm_client
        self.llm_is_live = llm_is_live
        self.logger = logger
        self.top_k_default = top_k_default
        self.min_relevance_score = min_relevance_score

    def query(
        self,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> QueryResult:
        top_k = top_k or self.top_k_default
        t_start = time.perf_counter()

        q_vec = self.embedder.embed_one(question)
        t_embed = time.perf_counter()

        results = self.store.search(q_vec, top_k=top_k, metadata_filter=metadata_filter)
        t_retrieve = time.perf_counter()

        retrieved_info = [
            RetrievedInfo(
                chunk_id=r.id,
                source_file=r.payload.get("source_file", ""),
                section_path=r.payload.get("section_path", ""),
                score=r.score,
                text=r.payload.get("text", ""),
            )
            for r in results
        ]

        best_score = results[0].score if results else 0.0

        if not results or best_score < self.min_relevance_score:
            answer_result = AnswerResult(
                answer=NO_CONTEXT_MESSAGE, cited_chunk_ids=[], has_sufficient_context=False
            )
        else:
            chunks = [
                RetrievedChunk(
                    chunk_id=r.id,
                    text=r.payload.get("text", ""),
                    source_file=r.payload.get("source_file", ""),
                    score=r.score,
                )
                for r in results
            ]
            answer_result = self.llm_client.generate_answer(question, chunks)
            if not answer_result.has_sufficient_context and not answer_result.answer.strip():
                answer_result.answer = NO_CONTEXT_MESSAGE

        t_generate = time.perf_counter()

        result = QueryResult(
            question=question,
            answer=answer_result.answer,
            cited_chunk_ids=answer_result.cited_chunk_ids,
            has_sufficient_context=answer_result.has_sufficient_context,
            retrieved=retrieved_info,
            embed_latency_ms=(t_embed - t_start) * 1000,
            retrieval_latency_ms=(t_retrieve - t_embed) * 1000,
            generation_latency_ms=(t_generate - t_retrieve) * 1000,
            total_latency_ms=(t_generate - t_start) * 1000,
            input_tokens=answer_result.input_tokens,
            output_tokens=answer_result.output_tokens,
            llm_is_live=self.llm_is_live,
        )

        self.logger.log(
            {
                "question": question,
                "top_k": top_k,
                "metadata_filter": metadata_filter,
                "chunk_count": len(retrieved_info),
                "retrieved_chunk_ids": [r.chunk_id for r in retrieved_info],
                "retrieved_scores": [r.score for r in retrieved_info],
                "has_sufficient_context": result.has_sufficient_context,
                "embed_latency_ms": result.embed_latency_ms,
                "retrieval_latency_ms": result.retrieval_latency_ms,
                "generation_latency_ms": result.generation_latency_ms,
                "total_latency_ms": result.total_latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "llm_is_live": result.llm_is_live,
            }
        )

        return result
