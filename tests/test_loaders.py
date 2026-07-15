import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ragapp.loaders import load_document, load_html, load_markdown, load_pdf, iter_corpus_files


def test_load_markdown_preserves_headers_and_text(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nSome body text about widgets.\n\n## Subsection\n\nMore detail here.")
    text = load_markdown(p)
    assert "# Title" in text
    assert "## Subsection" in text
    assert "widgets" in text


def test_load_html_strips_nav_and_footer_converts_headings(tmp_path):
    p = tmp_path / "doc.html"
    p.write_text(
        "<html><body>"
        "<nav>should not appear</nav>"
        "<h1>Getting started</h1>"
        "<p>Sign up with your email address.</p>"
        "<h2>Step one</h2>"
        "<p>Create a bucket in one of three regions.</p>"
        "<footer>should not appear either</footer>"
        "</body></html>"
    )
    text = load_html(p)
    assert "should not appear" not in text
    assert "should not appear either" not in text
    assert "# Getting started" in text
    assert "## Step one" in text
    assert "Create a bucket in one of three regions" in text


def test_load_pdf_extracts_real_text():
    # Uses the actual generated corpus PDF rather than building a throwaway
    # one here, so this test also catches drift if the corpus PDF changes.
    pdf_path = Path(__file__).parent.parent / "corpus" / "sla_support_policy.pdf"
    assert pdf_path.exists(), "run scripts/generate_sla_pdf.py first"
    text = load_pdf(pdf_path)
    assert "99.9%" in text
    assert "Enterprise" in text
    assert len(text) > 200


def test_load_document_dispatches_by_extension(tmp_path):
    md = tmp_path / "a.md"
    md.write_text("# X\n\nhello")
    html = tmp_path / "b.html"
    html.write_text("<html><body><h1>X</h1><p>hello</p></body></html>")
    assert "hello" in load_document(md)
    assert "hello" in load_document(html)

    bad = tmp_path / "c.txt"
    bad.write_text("hello")
    try:
        load_document(bad)
        assert False, "expected ValueError for unsupported extension"
    except ValueError:
        pass


def test_iter_corpus_files_finds_all_three_formats_in_real_corpus():
    corpus_dir = Path(__file__).parent.parent / "corpus"
    exts = {p.suffix.lower() for p in iter_corpus_files(corpus_dir)}
    assert ".md" in exts
    assert ".html" in exts
    assert ".pdf" in exts


def test_mixed_format_corpus_ingests_and_is_individually_idempotent():
    """A corpus containing all three formats together: ingest, assert all
    three contributed chunks, then confirm editing only the HTML file
    re-embeds just that file's chunks and leaves the PDF/MD chunks alone."""
    from src.ragapp.embeddings import HashingEmbedder
    from src.ragapp.ingest import ingest_corpus
    from src.ragapp.vectorstore import VectorStore

    tmp = tempfile.mkdtemp()
    try:
        corpus_dir = Path(tmp) / "corpus"
        corpus_dir.mkdir()

        (corpus_dir / "doc.md").write_text("# Notes\n\nMarkdown content about widgets and gadgets.")
        (corpus_dir / "doc.html").write_text(
            "<html><body><h1>Guide</h1><p>HTML content about onboarding steps.</p></body></html>"
        )
        real_pdf = Path(__file__).parent.parent / "corpus" / "sla_support_policy.pdf"
        shutil.copy(real_pdf, corpus_dir / "doc.pdf")

        embedder = HashingEmbedder(dimension=32)
        store = VectorStore(path=str(Path(tmp) / "qdrant"), collection="c", dimension=32)

        report = ingest_corpus(corpus_dir, store, embedder)
        sources_with_chunks = {s.source_file for s in report.per_source if s.chunks_total > 0}
        assert sources_with_chunks == {"doc.md", "doc.html", "doc.pdf"}
        assert report.total_embedded == report.total_chunks  # first ingest: everything is new

        pdf_ids_before = store.list_ids_by_source("doc.pdf")
        md_ids_before = store.list_ids_by_source("doc.md")

        (corpus_dir / "doc.html").write_text(
            "<html><body><h1>Guide</h1><p>EDITED html content about onboarding steps now.</p></body></html>"
        )
        report2 = ingest_corpus(corpus_dir, store, embedder)

        touched_sources = {s.source_file for s in report2.per_source if s.chunks_embedded > 0}
        assert touched_sources == {"doc.html"}, "only the edited HTML file's chunks should be re-embedded"
        assert store.list_ids_by_source("doc.pdf") == pdf_ids_before, "PDF chunks must be untouched"
        assert store.list_ids_by_source("doc.md") == md_ids_before, "MD chunks must be untouched"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
