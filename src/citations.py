"""Shared helpers for turning retrieved chunks into numbered, deduplicated sources.

Citation numbers are assigned in first-seen order over (section, page) and are
used both for the inline [n] markers the LLM is prompted to emit and for the
`sources` array returned to the frontend, so the two stay in sync.
"""

SNIPPET_LENGTH = 300


def _snippet(content: str) -> str:
    text = content.strip()
    # Strip the "[Section Title]\n\n" prefix added by chunker.py, if present.
    if text.startswith("[") and "]\n\n" in text:
        text = text.split("]\n\n", 1)[1].strip()
    if len(text) > SNIPPET_LENGTH:
        text = text[:SNIPPET_LENGTH].strip() + "…"
    return text


def dedupe_sources(chunks: list[dict]) -> tuple[list[dict], list[int]]:
    """Deduplicate chunks by (section, page) and assign 1-based citation numbers.

    Returns:
        sources: deduped list of {section, page, snippet, file}, in first-seen order.
        citation_numbers: same length as `chunks`; citation_numbers[i] is the
            1-based index into `sources` that chunks[i] maps to.
    """
    seen: dict[tuple, int] = {}
    sources: list[dict] = []
    citation_numbers: list[int] = []

    for chunk in chunks:
        key = (chunk["section"], int(float(chunk["page"])))
        if key not in seen:
            seen[key] = len(sources) + 1
            sources.append({
                "section": chunk["section"],
                "page": str(key[1]),
                "snippet": _snippet(chunk["content"]),
                "file": chunk.get("file", ""),
            })
        citation_numbers.append(seen[key])

    return sources, citation_numbers
