"""
LLM client for grounded generation and answer judging - backed by Groq.

Why Groq: Groq hosts open-weight models (Llama, GPT-OSS, Qwen, ...) on its
own fast inference hardware and offers a genuinely free developer tier, so
this project can run end-to-end without an Anthropic/OpenAI-style paid key -
that was an explicit requirement (see README).

Structured output via forced tool-calling
------------------------------------------
Groq's API is OpenAI-compatible: `tools` is a list of {"type": "function",
"function": {name, description, parameters}}, and `tool_choice` can force a
specific function via {"type": "function", "function": {"name": ...}}. As
with the original Anthropic-based design, we define one tool per task
(submit_answer / submit_judgment) and force the model to call it, rather
than asking for "JSON in your reply" and hoping nothing wraps it in prose or
markdown fences. Unlike Anthropic's Python SDK, Groq's OpenAI-compatible SDK
does NOT pre-parse the tool call arguments into a dict - `tool_call.function.
arguments` is a raw JSON *string* that must be parsed ourselves, so the
malformed-JSON safety net here is doing real work, not just defensive
boilerplate.

Model choice: `openai/gpt-oss-20b` (generator - fast, cheap, 131k context)
and `openai/gpt-oss-120b` (judge - larger, same family, on the same "don't
grade your own homework with the same weights" logic as before, though this
is a *same-family* size difference, not a cross-family separation - Groq
also hosts Qwen/Llama/Kimi families if a genuinely different judge model
family is wanted; see .env.example). `llama-3.3-70b-versatile` and
`llama-3.1-8b-instant` were deliberately avoided as defaults despite being
extremely well-known Groq models: Groq announced their deprecation on
2026-06-17 in favor of the GPT-OSS models, so the GPT-OSS models were picked
as the more future-proof default. Verify current model status at
https://console.groq.com/docs/models before relying on any specific ID.

Two clients, one interface
---------------------------
GroqLLMClient talks to the real Groq API (needs GROQ_API_KEY).
StubLLMClient needs no network or key: it produces clearly-labelled,
simplistic answers/scores so the rest of the pipeline (retrieval, citation
plumbing, metric aggregation) can be exercised end-to-end without a key -
such as the sandbox this project was built in, which cannot reach
api.groq.com at all (verified: a direct request returns 403 from the
sandbox's egress proxy, same as huggingface.co - see README). Because of
that, the Groq request shape below could *not* be live-verified against the
real API the way the equivalent Anthropic request was in an earlier version
of this project (that check relied on api.anthropic.com being reachable,
which api.groq.com is not, in this sandbox) - it is built carefully from
Groq's current, OpenAI-compatible documentation instead. Real
generation/faithfulness numbers require a real GROQ_API_KEY on a machine
that can reach api.groq.com; build_llm_client() below picks the right
client automatically and says which one is active.
"""
from __future__ import annotations

import json
import re as _re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_file: str
    score: float


@dataclass
class AnswerResult:
    answer: str
    cited_chunk_ids: list[str]
    has_sufficient_context: bool
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class JudgeResult:
    faithfulness_score: float  # 0.0-1.0, normalized from a 1-5 rubric
    faithfulness_rationale: str
    relevance_score: float  # 0.0-1.0, normalized from a 1-5 rubric
    relevance_rationale: str
    input_tokens: int = 0
    output_tokens: int = 0


_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the final answer to the user's question, grounded only in the provided context chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the question, using only information present in the provided chunks. "
                    "If the chunks don't contain enough information, say so plainly instead of guessing.",
                },
                "cited_chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "chunk_id values (from the provided chunks) that the answer actually draws on. Empty if has_sufficient_context is false.",
                },
                "has_sufficient_context": {
                    "type": "boolean",
                    "description": "true only if the provided chunks actually contain enough information to answer the question.",
                },
            },
            "required": ["answer", "cited_chunk_ids", "has_sufficient_context"],
        },
    },
}

_JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_judgment",
        "description": "Submit faithfulness and relevance scores for a generated answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "faithfulness_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "1 = answer contains claims not supported by the context (hallucinated). "
                    "5 = every claim in the answer is directly supported by the context.",
                },
                "faithfulness_rationale": {"type": "string", "description": "One or two sentences justifying the score, citing which claims are/aren't supported."},
                "relevance_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "1 = answer does not address the question asked. 5 = answer directly and completely addresses it.",
                },
                "relevance_rationale": {"type": "string", "description": "One or two sentences justifying the score."},
            },
            "required": ["faithfulness_score", "faithfulness_rationale", "relevance_score", "relevance_rationale"],
        },
    },
}


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[chunk_id: {c.chunk_id}] (source: {c.source_file})\n{c.text}")
    return "\n\n---\n\n".join(blocks)


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> AnswerResult: ...

    @abstractmethod
    def judge_answer(self, question: str, chunks: list[RetrievedChunk], answer: str) -> JudgeResult: ...


class GroqLLMClient(BaseLLMClient):
    def __init__(self, api_key: str, generator_model: str, judge_model: str):
        from groq import Groq  # local import: optional dep, only needed on this path

        self._client = Groq(api_key=api_key)
        self.generator_model = generator_model
        self.judge_model = judge_model

    def _call_tool(self, model: str, system: str, user: str, tool: dict, tool_name: str, max_tokens: int):
        """Shared call + parse logic. Returns (parsed_args_dict_or_None, usage, error_or_None)."""
        try:
            resp = self._client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": tool_name}},
            )
            message = resp.choices[0].message
            tool_calls = message.tool_calls or []
            if not tool_calls:
                return None, resp.usage, "model returned no tool call"
            # Groq's SDK does NOT pre-parse arguments into a dict - it's a raw
            # JSON string, so this json.loads is doing real, necessary work.
            args = json.loads(tool_calls[0].function.arguments)
            return args, resp.usage, None
        except Exception as e:
            return None, None, str(e)

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> AnswerResult:
        system = (
            "You are a QA assistant for a company knowledge base. Answer ONLY using the "
            "provided context chunks. Never use outside knowledge. If the chunks don't "
            "contain enough information to answer, set has_sufficient_context to false "
            "and say so in the answer rather than guessing. Always call submit_answer."
        )
        user = f"Question: {question}\n\nContext chunks:\n\n{_format_context(chunks)}"

        data, usage, error = self._call_tool(self.generator_model, system, user, _ANSWER_TOOL, "submit_answer", 1024)
        if error:
            return AnswerResult(
                answer=f"[generation failed, treating as no-context: {error}]",
                cited_chunk_ids=[],
                has_sufficient_context=False,
            )
        return AnswerResult(
            answer=data["answer"],
            cited_chunk_ids=list(data.get("cited_chunk_ids", [])),
            has_sufficient_context=bool(data.get("has_sufficient_context", False)),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    def judge_answer(self, question: str, chunks: list[RetrievedChunk], answer: str) -> JudgeResult:
        system = (
            "You are a strict evaluator of a QA system's answers. Score faithfulness "
            "(is every claim in the answer supported by the given context?) and relevance "
            "(does the answer address the question?) independently. Always call submit_judgment."
        )
        user = f"Question: {question}\n\nContext chunks:\n\n{_format_context(chunks)}\n\nGenerated answer:\n{answer}"

        data, usage, error = self._call_tool(self.judge_model, system, user, _JUDGE_TOOL, "submit_judgment", 512)
        if error:
            return JudgeResult(
                faithfulness_score=0.0,
                faithfulness_rationale=f"[judge failed: {error}]",
                relevance_score=0.0,
                relevance_rationale=f"[judge failed: {error}]",
            )
        return JudgeResult(
            faithfulness_score=(int(data["faithfulness_score"]) - 1) / 4.0,
            faithfulness_rationale=data["faithfulness_rationale"],
            relevance_score=(int(data["relevance_score"]) - 1) / 4.0,
            relevance_rationale=data["relevance_rationale"],
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on",
    "for", "and", "or", "it", "this", "that", "with", "as", "by", "at", "from", "does",
    "do", "did", "can", "what", "how", "when", "where", "which", "who", "your", "you",
}


def _content_words(text: str) -> set[str]:
    words = _re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class StubLLMClient(BaseLLMClient):
    """No network, no key. Used automatically when GROQ_API_KEY is unset,
    so the ingestion/retrieval/API/eval plumbing can still be exercised.

    judge_answer here is a real, computed lexical-overlap heuristic, not a
    hardcoded placeholder score: faithfulness is approximated as the fraction
    of the answer's content words that also appear somewhere in the provided
    context (a crude, non-semantic proxy for groundedness - it can't catch
    subtle misstatements, but it does respond to the actual answer/context
    given, unlike a constant), and relevance is approximated as content-word
    overlap between the question and the answer. Both are clearly weaker than
    a real LLM judge and are labelled as such in every output row - but they
    are *measured* per-question, not *asserted* as one fixed number regardless
    of input, which is the property this stand-in is trying to preserve while
    no live LLM is available."""

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> AnswerResult:
        if not chunks:
            return AnswerResult(answer="[stub] No relevant context was retrieved.", cited_chunk_ids=[], has_sufficient_context=False)
        best = chunks[0]
        return AnswerResult(
            answer=f"[stub answer, no live LLM call] Based on {best.source_file}: {best.text[:200]}",
            cited_chunk_ids=[best.chunk_id],
            has_sufficient_context=True,
            input_tokens=0,
            output_tokens=0,
        )

    def judge_answer(self, question: str, chunks: list[RetrievedChunk], answer: str) -> JudgeResult:
        context_words = set()
        for c in chunks:
            context_words |= _content_words(c.text)
        answer_words = _content_words(answer)

        if answer_words:
            grounded = len(answer_words & context_words) / len(answer_words)
        else:
            grounded = 0.0

        question_words = _content_words(question)
        if question_words and answer_words:
            addressed = len(answer_words & question_words) / len(question_words)
        else:
            addressed = 0.0

        return JudgeResult(
            faithfulness_score=round(grounded, 3),
            faithfulness_rationale=(
                f"[stub heuristic judge, not a real LLM] {len(answer_words & context_words)}/"
                f"{len(answer_words) or 1} of the answer's content words also appear in the "
                f"provided context - a crude lexical proxy for groundedness, not semantic entailment."
            ),
            relevance_score=round(addressed, 3),
            relevance_rationale=(
                f"[stub heuristic judge, not a real LLM] {len(answer_words & question_words)}/"
                f"{len(question_words) or 1} of the question's content words are echoed in the "
                f"answer - a crude lexical proxy for topical relevance, not real judgment of whether "
                f"the question was actually answered."
            ),
        )


def build_llm_client(api_key: str | None, generator_model: str, judge_model: str) -> tuple[BaseLLMClient, bool]:
    """Returns (client, is_live). is_live=False means StubLLMClient is active
    (no GROQ_API_KEY was configured) - callers should surface this
    prominently rather than silently presenting stub output as real."""
    if api_key:
        return GroqLLMClient(api_key=api_key, generator_model=generator_model, judge_model=judge_model), True
    return StubLLMClient(), False
