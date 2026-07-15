import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ragapp.chunking import chunk_text, split_into_sections


def test_deterministic():
    text = "# Title\n\n" + ("This is a sentence. " * 200)
    c1 = chunk_text(text, chunk_size=500, chunk_overlap=100)
    c2 = chunk_text(text, chunk_size=500, chunk_overlap=100)
    assert [c.text for c in c1] == [c.text for c in c2]


def test_respects_chunk_size_roughly():
    text = "This is a sentence. " * 500
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    # allow slack: packing is greedy on sentence boundaries, not exact
    for c in chunks:
        assert len(c.text) <= 600


def test_overlap_creates_continuity():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. " * 30
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=60)
    assert len(chunks) > 1
    # some trailing words of chunk[i] should reappear at the start of chunk[i+1]
    tail = chunks[0].text[-20:]
    assert any(tail[-5:] in chunks[i].text for i in range(1, len(chunks)))


def test_no_overlap_bigger_than_size_raises():
    try:
        chunk_text("hello world", chunk_size=100, chunk_overlap=100)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_header_breadcrumb_tracked():
    text = "# Pricing\n\nIntro text.\n\n## Enterprise tier\n\nEnterprise costs a lot of money and has many features."
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=100)
    paths = [c.section_path for c in chunks]
    assert "Pricing" in paths
    assert "Pricing > Enterprise tier" in paths


def test_section_split_nesting_pops_correctly():
    text = "# A\n\ntext a\n\n## B\n\ntext b\n\n# C\n\ntext c"
    sections = split_into_sections(text)
    breadcrumbs = [s.breadcrumb for s in sections]
    assert ("A",) in breadcrumbs
    assert ("A", "B") in breadcrumbs
    assert ("C",) in breadcrumbs  # back to depth-1, "B" should have popped


def test_atomic_unit_bigger_than_chunk_hard_cuts():
    huge_word_block = "x" * 3000  # one giant "paragraph", no sentence breaks
    chunks = chunk_text(huge_word_block, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 6
    for c in chunks:
        assert len(c.text) <= 500


def test_empty_input_returns_no_chunks():
    assert chunk_text("   \n\n  ") == []
