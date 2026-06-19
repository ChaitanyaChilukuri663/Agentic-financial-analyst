"""Turn a parsed filing into retrievable chunks.

The key idea for financial RAG: a bare table cell is meaningless out of context, so
each **row-level chunk carries its header/caption/unit context** and each value is
paired with its column header. Narrative sections become overlapping text windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ledgerlens.ingest.filing import ParsedFiling

_WINDOW = 600
_OVERLAP = 80


@dataclass
class Chunk:
    """One retrievable unit of evidence."""

    chunk_id: str
    text: str
    kind: str  # "table_row" | "text"
    metadata: dict[str, str] = field(default_factory=dict)


def serialize_table_row(header: list[str], row: list[str]) -> str:
    """Render a table row as self-describing text (FinQA's ``table_row_to_text`` form)."""
    parts: list[str] = []
    if header and header[0]:
        parts.append(header[0])
    label = row[0] if row else ""
    for head, cell in zip(header[1:], row[1:], strict=False):
        parts.append(f"the {label} of {head} is {cell} ;")
    return " ".join(parts).strip()


def chunk_filing(parsed: ParsedFiling, *, filing_id: str = "") -> list[Chunk]:
    """Produce row-level table chunks (with header context) and section text chunks."""
    chunks: list[Chunk] = []
    for ti, extracted in enumerate(parsed.tables):
        rows = extracted.table.rows
        header = rows[0] if rows else []
        prefix = f"{extracted.caption} ({extracted.unit_hint})".strip()
        for ri, row in enumerate(rows[1:], start=1):
            body = serialize_table_row(header, row)
            text = f"{prefix}. {body}".strip(". ").strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{filing_id}:t{ti}:r{ri}",
                    text=text,
                    kind="table_row",
                    metadata={
                        "filing": filing_id,
                        "caption": extracted.caption,
                        "unit": extracted.unit_hint,
                    },
                )
            )
    for section in parsed.sections:
        for ci, window in enumerate(_windows(section.text)):
            chunks.append(
                Chunk(
                    chunk_id=f"{filing_id}:{section.item}:{ci}",
                    text=window,
                    kind="text",
                    metadata={"filing": filing_id, "item": section.item},
                )
            )
    return chunks


def _windows(text: str, size: int = _WINDOW, overlap: int = _OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    windows: list[str] = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + size])
        start += size - overlap
    return windows
