"""
Loaders that turn PDF / HTML / MD files into plain text ready for chunking.

Each loader tries to preserve markdown-style '#' headers where it can, since
chunking.split_into_sections uses those headers to build section breadcrumbs
for citation quality. HTML headings (h1-h6) are converted to '#'..'######'.
PDFs don't have structural heading info once flattened to text, so they come
through as a single unheadered section - acceptable for our corpus since PDF
documents are short (see corpus/README notes).
"""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".html", ".htm", ".pdf"}


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")

    # drop non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    lines: list[str] = []
    body = soup.body or soup

    for el in body.descendants:
        name = getattr(el, "name", None)
        if name and re.fullmatch(r"h[1-6]", name):
            level = int(name[1])
            text = el.get_text(strip=True)
            if text:
                lines.append(f"{'#' * level} {text}")
        elif name == "li":
            text = el.get_text(strip=True)
            if text:
                lines.append(f"- {text}")
        elif name in ("p", "td", "th"):
            text = el.get_text(strip=True)
            if text:
                lines.append(text)

    return "\n\n".join(lines)


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_document(path: Path) -> str:
    """Dispatch on file extension. Raises ValueError on unsupported types so
    ingestion fails loudly rather than silently skipping a file."""
    ext = path.suffix.lower()
    if ext in (".md", ".markdown"):
        return load_markdown(path)
    if ext in (".html", ".htm"):
        return load_html(path)
    if ext == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported document type: {path} (extension {ext!r})")


def iter_corpus_files(corpus_dir: Path):
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
