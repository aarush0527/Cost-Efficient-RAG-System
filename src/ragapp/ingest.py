"""
Idempotent ingestion pipeline.

How idempotency works
----------------------
Each chunk's id is derived entirely from its content:

    chunk_id = uuid5(NAMESPACE, f"{source_file}::{chunk_index}::{sha256(text)[:16]}")

Re-running ingestion on an unchanged corpus recomputes the exact same ids,
so nothing new is written and nothing is re-embedded (we skip embedding any
chunk whose id already exists in the store - the id itself IS the
change-detection key, since it encodes a content hash).

If a document is edited, the chunk(s) at the changed index get a *new* id
(different hash), so they're embedded and upserted as new points, while
whatever *old* id used to live at that (source_file, chunk_index) - or any
index that no longer exists in the new chunking - is deleted. This is done
by diffing "ids currently stored for this source file" against "ids the
current chunking produces", per source file, via VectorStore.list_ids_by_source.

Net effect: re-ingesting an unchanged corpus is a fast no-op (no embed calls,
no writes beyond a few reads); editing one document only touches that
document's stale/changed chunks, not the whole corpus.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_text
from .embeddings import BaseEmbedder
from .loaders import iter_corpus_files, load_document
from .vectorstore import VectorStore

_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

_CATEGORY_BY_STEM = {
    "architecture_overview": "architecture",
    "pricing": "pricing",
    "security_compliance": "security",
    "api_reference": "api",
    "onboarding_guide": "onboarding",
    "troubleshooting_faq": "troubleshooting",
    "sla_support_policy": "sla",
    "data_retention_policy": "retention",
}


def compute_chunk_id(source_file: str, chunk_index: int, text: str) -> str:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    name = f"{source_file}::{chunk_index}::{content_hash}"
    return str(uuid.uuid5(_NAMESPACE, name))


def infer_doc_category(filename: str) -> str:
    stem = Path(filename).stem
    return _CATEGORY_BY_STEM.get(stem, "general")


@dataclass
class SourceIngestStats:
    source_file: str
    chunks_total: int = 0
    chunks_embedded: int = 0
    chunks_skipped_unchanged: int = 0
    chunks_deleted_stale: int = 0


@dataclass
class IngestReport:
    per_source: list[SourceIngestStats] = field(default_factory=list)
    embedding_model: str = ""
    embedding_dimension: int = 0

    @property
    def total_chunks(self) -> int:
        return sum(s.chunks_total for s in self.per_source)

    @property
    def total_embedded(self) -> int:
        return sum(s.chunks_embedded for s in self.per_source)

    @property
    def total_skipped(self) -> int:
        return sum(s.chunks_skipped_unchanged for s in self.per_source)

    @property
    def total_deleted(self) -> int:
        return sum(s.chunks_deleted_stale for s in self.per_source)


def ingest_corpus(
    corpus_dir: str | Path,
    store: VectorStore,
    embedder: BaseEmbedder,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> IngestReport:
    corpus_dir = Path(corpus_dir)
    report = IngestReport(embedding_model=embedder.model_name, embedding_dimension=embedder.dimension)

    seen_source_files: set[str] = set()

    for path in iter_corpus_files(corpus_dir):
        source_file = path.name
        seen_source_files.add(source_file)
        source_type = path.suffix.lstrip(".").lower()
        raw_text = load_document(path)
        chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        stats = SourceIngestStats(source_file=source_file, chunks_total=len(chunks))

        new_ids = []
        new_texts_to_embed = []
        new_payloads_to_embed = []
        all_new_ids_set = set()

        old_ids = store.list_ids_by_source(source_file)

        for c in chunks:
            cid = compute_chunk_id(source_file, c.chunk_index, c.text)
            new_ids.append(cid)
            all_new_ids_set.add(cid)

            if cid in old_ids:
                stats.chunks_skipped_unchanged += 1
                continue

            new_texts_to_embed.append(c.text)
            new_payloads_to_embed.append(
                {
                    "source_file": source_file,
                    "source_type": source_type,
                    "doc_category": infer_doc_category(source_file),
                    "section_path": c.section_path,
                    "chunk_index": c.chunk_index,
                    "content_hash": hashlib.sha256(c.text.encode("utf-8")).hexdigest()[:16],
                    "char_count": len(c.text),
                    "text": c.text,
                    "embedding_model": embedder.model_name,
                }
            )
            stats.chunks_embedded += 1

        if new_texts_to_embed:
            vectors = embedder.embed(new_texts_to_embed)
            new_ids_to_upsert = [
                compute_chunk_id(source_file, p["chunk_index"], p["text"]) for p in new_payloads_to_embed
            ]
            store.upsert(new_ids_to_upsert, vectors, new_payloads_to_embed)

        stale_ids = old_ids - all_new_ids_set
        if stale_ids:
            store.delete(list(stale_ids))
            stats.chunks_deleted_stale = len(stale_ids)

        report.per_source.append(stats)

    # Clean up documents that were fully removed from the corpus directory
    # (not just edited) - iter_corpus_files only visits files present on
    # disk, so this second pass is needed to catch full deletions.
    all_stored_sources = store.list_all_source_files()
    removed_sources = all_stored_sources - seen_source_files
    for removed_source in removed_sources:
        stale_ids = store.list_ids_by_source(removed_source)
        store.delete(list(stale_ids))
        report.per_source.append(
            SourceIngestStats(source_file=removed_source, chunks_deleted_stale=len(stale_ids))
        )

    return report
