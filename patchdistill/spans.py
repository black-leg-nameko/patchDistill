"""Token/character span utilities."""

from __future__ import annotations


def char_span(text: str, substring: str) -> tuple[int, int] | None:
    if not substring:
        return None
    start = text.find(substring)
    if start < 0:
        return None
    return start, start + len(substring)


def token_positions_from_offsets(offsets: list[tuple[int, int]], span: tuple[int, int] | None) -> list[int]:
    if span is None:
        return []
    start, end = span
    positions: list[int] = []
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        overlaps = tok_start < end and tok_end > start
        if overlaps:
            positions.append(i)
    return positions

