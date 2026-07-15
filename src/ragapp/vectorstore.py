"""
Thin wrapper around Qdrant's embedded/local client.

Kept behind this narrow interface (upsert / search / delete / count / list_by_source)
on purpose: if embedded Qdrant ever became a problem in some environment, the
rest of the codebase (ingest.py, rag_service.py, eval/*) only talks to this
module, not to qdrant_client directly - swapping the backing store (e.g. to
ChromaDB) would touch only this file.
"""
from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict


class VectorStore:
    def __init__(self, path: str, collection: str, dimension: int):
        self.client = QdrantClient(path=path)
        self.collection = collection
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )

    def upsert(self, ids: list[str], vectors, payloads: list[dict]) -> None:
        points = [
            PointStruct(id=i, vector=v.tolist() if hasattr(v, "tolist") else list(v), payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        if points:
            self.client.upsert(self.collection, points=points)

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.client.delete(self.collection, points_selector=ids)

    def count(self) -> int:
        return self.client.count(self.collection).count

    def get_payload(self, point_id: str) -> dict | None:
        res = self.client.retrieve(self.collection, ids=[point_id], with_payload=True)
        return res[0].payload if res else None

    def list_all_source_files(self) -> set[str]:
        """Distinct source_file values across the whole collection - used by
        ingestion to detect documents that were deleted from disk entirely
        (not just edited) so their vectors can be cleaned up too."""
        sources: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True
            )
            for p in points:
                sf = (p.payload or {}).get("source_file")
                if sf:
                    sources.add(sf)
            if offset is None:
                break
        return sources

    def list_ids_by_source(self, source_file: str) -> set[str]:
        """All point ids currently stored for a given source file - used by
        ingestion to diff old vs. new chunk ids and delete stale ones after
        an edit."""
        ids: set[str] = set()
        offset = None
        flt = Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))])
        while True:
            points, offset = self.client.scroll(
                self.collection, scroll_filter=flt, limit=256, offset=offset, with_payload=False
            )
            ids.update(str(p.id) for p in points)
            if offset is None:
                break
        return ids

    def search(
        self,
        query_vector,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[SearchResult]:
        flt = None
        if metadata_filter:
            flt = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in metadata_filter.items()]
            )
        res = self.client.query_points(
            self.collection,
            query=query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector),
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        )
        return [SearchResult(id=str(p.id), score=p.score, payload=p.payload or {}) for p in res.points]
