"""Whitespace/punctuation-based text segmentation.

There is no subtitle timing here — segments are just chunks of source text
sized for one translation call each. Splits on blank-line paragraph breaks
first, then on Tibetan sentence-ending shad marks (། ༎) for any paragraph
still longer than max_chars, then falls back to plain whitespace chunking
for any single "sentence" that is still too long on its own.
"""

import re

_SHAD_SPLIT_RE = re.compile(r"(?<=[།༎])\s*")


def _chunk_by_words(text: str, max_chars: int) -> list[str]:
    words = text.split(" ")
    chunks, current = [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def _chunk_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    pieces = [p.strip() for p in _SHAD_SPLIT_RE.split(paragraph) if p.strip()]
    chunks, current = [], ""
    for piece in pieces:
        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_chunk_by_words(piece, max_chars))
        elif current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def split_into_segments(text: str, max_chars: int = 400) -> list[str]:
    """Split raw text into a list of segment strings."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    segments = []
    for para in paragraphs:
        para = " ".join(para.split())
        if not para:
            continue
        segments.extend(_chunk_paragraph(para, max_chars))
    return segments
