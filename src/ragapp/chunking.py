"""
Deterministic, recursive text chunker.

Design goals (see README for the full rationale):
  - Deterministic: same input + same params => same chunks, every time. This
    is what makes idempotent re-ingest possible (chunk ids are derived from
    chunk content).
  - Structure-aware: prefers to split on headers, then paragraphs, then
    sentences, and only falls back to a hard character cut if a single
    "atomic" unit is still bigger than chunk_size.
  - Character-based sizing (not token-based) to avoid a tokenizer dependency.
    Swapping in a token-based length function is a one-line change - see
    `length_fn` below.

Defaults: chunk_size=1000 chars (~150-220 tokens of English text),
overlap=200 chars (~20%). Overlap preserves continuity for facts/sentences
that straddle a split point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _default_length(text: str) -> int:
    return len(text)


@dataclass
class Section:
    breadcrumb: tuple[str, ...]
    text: str


def split_into_sections(raw_text: str) -> list[Section]:
    sections: list[Section] = []
    breadcrumb_stack: list[tuple[int, str]] = []  # (level, title)
    buffer: list[str] = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            crumbs = tuple(title for _, title in breadcrumb_stack)
            sections.append(Section(breadcrumb=crumbs, text=text))
        buffer.clear()

    for line in raw_text.splitlines():
        m = _HEADER_RE.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while breadcrumb_stack and breadcrumb_stack[-1][0] >= level:
                breadcrumb_stack.pop()
            breadcrumb_stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return sections


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def _pack(units: list[str], chunk_size: int, overlap: int, length_fn: Callable[[str], int]) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def join(units_: list[str]) -> str:
        return " ".join(units_).strip()

    for unit in units:
        u_len = length_fn(unit)

        if u_len > chunk_size:
    
            if current:
                chunks.append(join(current))
                current, current_len = [], 0
            for i in range(0, len(unit), chunk_size - overlap if chunk_size > overlap else chunk_size):
                chunks.append(unit[i:i + chunk_size])
            continue

        if current_len + u_len + 1 > chunk_size and current:
            chunk_text = join(current)
            chunks.append(chunk_text)
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + length_fn(prev) > overlap:
                    break
                carry.insert(0, prev)
                carry_len += length_fn(prev) + 1
            current, current_len = carry, carry_len

        current.append(unit)
        current_len += u_len + 1

    if current:
        chunks.append(join(current))

    return chunks


@dataclass
class Chunk:
    text: str
    chunk_index: int
    section_path: str

def chunk_text(
    raw_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    length_fn: Callable[[str], int] = _default_length,
) -> list[Chunk]:

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sections = split_into_sections(raw_text)
    chunks: list[Chunk] = []
    idx = 0

    for section in sections:
        section_len = length_fn(section.text)
        breadcrumb = " > ".join(section.breadcrumb) if section.breadcrumb else ""

        if section_len <= chunk_size:
            pieces = [section.text]
        else:
            paragraphs = _split_paragraphs(section.text)

            units: list[str] = []
            for p in paragraphs:
                if length_fn(p) <= chunk_size:
                    units.append(p)
                else:
                    units.extend(_split_sentences(p))
            pieces = _pack(units, chunk_size, chunk_overlap, length_fn)

        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append(Chunk(text=piece.strip(), chunk_index=idx, section_path=breadcrumb))
            idx += 1

    return chunks
