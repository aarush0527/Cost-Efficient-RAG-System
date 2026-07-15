"""
Resolves each eval question's `anchor_snippets` (short, unique substrings of
the source text) into the actual chunk_id(s) currently stored in the vector
store that contain them.

Why not just hardcode chunk ids in eval_dataset.json: chunk ids are content
hashes (see ingest.compute_chunk_id) and depend on exactly how chunking
sliced the document - which can shift if chunk_size/overlap are retuned.
Anchor-snippet resolution stays correct across re-chunking as long as the
underlying fact is still in the corpus somewhere, which is what we actually
care about for "is this the relevant chunk", not a specific hash value.

Whitespace is normalized on both sides before the containment check, since
the corpus markdown files hard-wrap prose at ~80 columns but the chunker
otherwise preserves a section's original text verbatim when the section
already fits within chunk_size (i.e. newlines survive into chunk.text).
"""
from __future__ import annotations

import re

from src.ragapp.vectorstore import VectorStore


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_all_chunks(store: VectorStore) -> list[dict]:
    """All (id, text, source_file) currently in the collection."""
    chunks = []
    offset = None
    while True:
        points, offset = store.client.scroll(
            store.collection, limit=256, offset=offset, with_payload=True
        )
        for p in points:
            payload = p.payload or {}
            chunks.append({"id": str(p.id), "text": payload.get("text", ""), "source_file": payload.get("source_file", "")})
        if offset is None:
            break
    return chunks


def resolve_gold_chunk_ids(store: VectorStore, eval_dataset: list[dict]) -> dict[str, list[str]]:
    all_chunks = get_all_chunks(store)
    normed = [(c["id"], _norm(c["text"])) for c in all_chunks]

    gold_map: dict[str, list[str]] = {}
    for item in eval_dataset:
        if not item.get("is_answerable", True):
            gold_map[item["id"]] = []
            continue

        matched_ids: list[str] = []
        for anchor in item["anchor_snippets"]:
            anchor_n = _norm(anchor)
            for cid, text_n in normed:
                if anchor_n in text_n and cid not in matched_ids:
                    matched_ids.append(cid)

        if not matched_ids:
            raise ValueError(
                f"Could not resolve any gold chunk for question {item['id']!r} "
                f"(anchors={item['anchor_snippets']!r}). The corpus or chunking "
                f"params likely changed - update the anchor snippet."
            )
        gold_map[item["id"]] = matched_ids

    return gold_map
