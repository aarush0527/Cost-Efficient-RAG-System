import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ragapp.embeddings import HashingEmbedder
from src.ragapp.ingest import ingest_corpus
from src.ragapp.vectorstore import VectorStore


def _fresh_env():
    tmp = tempfile.mkdtemp()
    corpus_dir = Path(tmp) / "corpus"
    corpus_dir.mkdir()
    qdrant_dir = Path(tmp) / "qdrant"
    return tmp, corpus_dir, qdrant_dir


def test_reingest_unchanged_is_noop():
    tmp, corpus_dir, qdrant_dir = _fresh_env()
    try:
        (corpus_dir / "doc1.md").write_text("# Title\n\nSome content here about widgets and gadgets.")
        embedder = HashingEmbedder(dimension=32)
        store = VectorStore(path=str(qdrant_dir), collection="c", dimension=32)

        report1 = ingest_corpus(corpus_dir, store, embedder)
        count_after_first = store.count()
        assert report1.total_embedded > 0
        assert report1.total_skipped == 0

        report2 = ingest_corpus(corpus_dir, store, embedder)
        count_after_second = store.count()

        assert count_after_first == count_after_second, "re-ingest must not create duplicate vectors"
        assert report2.total_embedded == 0, "unchanged content should not be re-embedded"
        assert report2.total_skipped == report1.total_embedded
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edit_triggers_targeted_reembed_and_stale_cleanup():
    tmp, corpus_dir, qdrant_dir = _fresh_env()
    try:
        doc_path = corpus_dir / "doc1.md"
        doc_path.write_text("# Title\n\nOriginal content about widgets.\n\n# Other\n\nUnrelated stable section.")
        embedder = HashingEmbedder(dimension=32)
        store = VectorStore(path=str(qdrant_dir), collection="c", dimension=32)

        ingest_corpus(corpus_dir, store, embedder)
        count_before_edit = store.count()

        doc_path.write_text("# Title\n\nEDITED content about gadgets now.\n\n# Other\n\nUnrelated stable section.")
        report = ingest_corpus(corpus_dir, store, embedder)

        assert report.total_embedded >= 1, "edited chunk should be re-embedded"
        assert report.total_deleted >= 1, "stale vector for the old chunk content should be deleted"
        # unrelated section's chunk should have been recognized as unchanged
        assert report.total_skipped >= 1

        count_after_edit = store.count()
        assert count_after_edit == count_before_edit, "count should stay stable: old stale chunk replaced by new one"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_deleted_document_removes_its_vectors():
    tmp, corpus_dir, qdrant_dir = _fresh_env()
    try:
        (corpus_dir / "keep.md").write_text("# Keep\n\nThis document stays.")
        (corpus_dir / "remove.md").write_text("# Remove\n\nThis document will be deleted from disk.")
        embedder = HashingEmbedder(dimension=32)
        store = VectorStore(path=str(qdrant_dir), collection="c", dimension=32)

        ingest_corpus(corpus_dir, store, embedder)
        count_with_both = store.count()
        assert count_with_both >= 2

        (corpus_dir / "remove.md").unlink()
        report = ingest_corpus(corpus_dir, store, embedder)
        count_after_removal = store.count()

        assert report.total_deleted > 0
        assert count_after_removal < count_with_both, "vectors for a fully deleted source file must be cleaned up"
        assert store.list_ids_by_source("remove.md") == set()
        assert store.list_ids_by_source("keep.md") != set()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
