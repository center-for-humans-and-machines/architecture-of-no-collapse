"""Small, deterministic text utilities shared by metrics and the fake embedder."""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """Return lowercased word unigrams (alphanumeric runs).

    Mirrors the lexical-diversity tokenizer: split on whitespace, strip leading
    and trailing non-alphanumeric characters, and drop tokens that become empty.
    No stemming or stop-word removal.
    """
    tokens: list[str] = []
    for raw in text.lower().split():
        start = 0
        end = len(raw)
        while start < end and not raw[start].isalnum():
            start += 1
        while end > start and not raw[end - 1].isalnum():
            end -= 1
        if start < end:
            tokens.append(raw[start:end])
    return tokens
