"""
Embedding backends, behind one interface.

Why two backends: the documented, recommended default for real use is a
local open-source sentence embedding model (BAAI/bge-small-en-v1.5, 384-dim)
run via `sentence-transformers` - it costs $0 per call (a real chunk of this
project's "low cost" story: no embedding API bill at any scale), and runs
fine on CPU.

That model's weights are fetched from the Hugging Face Hub on first use.
This is a completely normal one-time step on a machine with internet access
- but the sandbox this project was *built* in has network access restricted
to a small allowlist that does not include huggingface.co, so the real
backend could not be exercised end-to-end during development here (verified:
a request to huggingface.co returns 403 from the sandbox's egress proxy).

Rather than mock this away silently, there's a second, deterministic,
offline backend (`HashingEmbedder`) used for local tests / CI / this dev
sandbox: a seeded random-projection of hashed character n-grams into a fixed
dimensionality. It needs no network and no model weights, and - because it's
deterministic - is genuinely useful for fast unit tests independent of
whether HF is reachable. It is *not* a semantic embedding and is clearly
labelled as a fallback, never presented as the "real" result.

Which one is active is one env var: EMBEDDER_BACKEND=sentence-transformers|hashing.
Both record `model_name` and `dimension`, per the ingestion requirement to
track embedding model + dimensionality alongside the vectors.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    model_name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dimension) float32 array, L2-normalized (so cosine
        similarity == dot product, which is what the vector store computes)."""
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Production default. Requires `pip install sentence-transformers` and
    internet access to Hugging Face Hub the first time a given model_name is
    used (weights are then cached locally under ~/.cache/huggingface)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)


class SpacyEmbedder(BaseEmbedder):

    def __init__(self, model_name: str = "en_core_web_md"):
        import spacy  # local import: optional dep

        self.model_name = model_name
        self._nlp = spacy.load(model_name)
        self.dimension = self._nlp.vocab.vectors_length

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, doc in enumerate(self._nlp.pipe(texts)):
            v = doc.vector
            norm = np.linalg.norm(v)
            out[i] = v / norm if norm > 0 else v
        return out


_WORD_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder(BaseEmbedder):
    """Deterministic, offline, dependency-free embedder used for tests / CI /
    environments without Hugging Face access. NOT a semantic embedding model
    - do not use it to produce the project's real evaluation numbers, only
    to validate that the pipeline's plumbing (ingest -> store -> retrieve)
    is correct end to end without a network dependency.

    Technique: hash each word (and character 3-gram, to give partial credit
    for morphological/typo similarity) into a bucket of a fixed-size vector,
    seeded per-bucket sign via the hash itself so the projection is stable
    across runs, then L2-normalize. This is the classic "hashing trick"
    (Weinberger et al., used in Vowpal Wabbit and similar systems) - a real,
    if crude, technique, not a random placeholder.
    """

    def __init__(self, dimension: int = 384):
        self.model_name = "hashing-embedder-v1 (offline dev fallback, not semantic)"
        self.dimension = dimension

    def _hash_to_bucket_and_sign(self, token: str) -> tuple[int, float]:
        h = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(h[:4], "big") % self.dimension
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        return bucket, sign

    def _features(self, text: str) -> list[str]:
        text = text.lower()
        words = _WORD_RE.findall(text)
        feats = list(words)
        for w in words:
            padded = f"^{w}$"
            feats.extend(padded[i:i + 3] for i in range(len(padded) - 2))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                bucket, sign = self._hash_to_bucket_and_sign(feat)
                out[row, bucket] += sign
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


def build_embedder(backend: str, model_name: str, hashing_dim: int) -> BaseEmbedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model_name)
    if backend == "spacy":
        return SpacyEmbedder(model_name=model_name)
    if backend == "hashing":
        return HashingEmbedder(dimension=hashing_dim)
    raise ValueError(f"Unknown embedder backend: {backend!r}")
