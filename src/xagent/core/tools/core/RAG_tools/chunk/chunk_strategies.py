from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", "。", "！", "？", ". ", ", ", " "]


def _join_paragraphs(paragraphs: List[Dict[str, Any]]) -> str:
    texts = [p.get("text", "") for p in paragraphs if p.get("text")]
    return "\n\n".join(texts)


def _create_chunk_record(
    text: str, source_paragraph: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a chunk record with position information (no ID or timestamp).

    Args:
        text: Chunk text content
        source_paragraph: Source paragraph metadata for position info

    Returns:
        Chunk record with text, position fields, and full metadata dictionary
    """
    # Extract metadata from source paragraph if available
    metadata = (source_paragraph or {}).get("metadata", {})

    return {
        "text": text,
        "page_number": metadata.get("page_number"),
        "section": metadata.get("section"),
        "anchor": metadata.get("anchor"),
        "json_path": metadata.get("json_path"),
        "metadata": metadata,  # Preserve full metadata dictionary
    }


def _split_by_separators_core(text: str, separators: Optional[List[str]]) -> List[str]:
    """Core splitter that preserves delimiters by attaching them to the previous unit.

    This function contains the shared logic for regex construction, splitting with
    capturing groups, handling None parts, and attaching delimiters to the
    previous chunk. It returns a list of pure text chunks and is reused by
    higher-level wrappers with/without metadata.

    Args:
        text: Text to split
        separators: List of separator strings to use for splitting (defaults applied inside)

    Returns:
        List of text chunks with delimiters attached to the previous chunk
    """
    if not separators:
        separators = DEFAULT_SEPARATORS

    escaped_separators = [re.escape(sep) for sep in separators]
    pattern = "|".join(f"({escaped_sep})" for escaped_sep in escaped_separators)

    parts = re.split(pattern, text)

    chunks: List[str] = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            text_part = parts[i] if parts[i] is not None else ""
            delimiter_part = parts[i + 1] if parts[i + 1] is not None else ""
            chunk = text_part + delimiter_part
            if chunk.strip():
                chunks.append(chunk)
        else:
            text_part = parts[i] if parts[i] is not None else ""
            if text_part.strip():
                chunks.append(text_part)

    return chunks


def _split_by_separators(text: str, separators: Optional[List[str]]) -> List[str]:
    """Wrapper: returns pure text chunks using the core splitter."""
    return _split_by_separators_core(text, separators)


def _split_by_separators_with_metadata(
    text: str,
    separators: Optional[List[str]],
    source_paragraph: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Wrapper: returns structured chunks with metadata using the core splitter."""
    units = _split_by_separators_core(text, separators)
    return [
        {"text": unit, "source_paragraph": source_paragraph}
        for unit in units
        if unit.strip()
    ]


def _window_with_overlap(
    tokens: List[str], chunk_size: int, chunk_overlap: int
) -> List[str]:
    if chunk_size <= 0:
        return []
    if chunk_overlap < 0:
        chunk_overlap = 0
    chunks: List[str] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = "".join(tokens[start:end])
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(start + chunk_size - chunk_overlap, start + 1)
    return chunks


def _window_with_overlap_and_metadata(
    chunk_records: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int
) -> List[Dict[str, Any]]:
    """Apply sliding window with overlap to chunk records, preserving metadata.

    Args:
        chunk_records: List of chunk records with text and source_paragraph
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of windowed chunk records with preserved metadata
    """
    if chunk_size <= 0:
        return []
    if chunk_overlap < 0:
        chunk_overlap = 0

    # Convert chunk records to character tokens while preserving metadata
    tokens: list[str] = []
    metadata_map: dict[
        int, dict[str, Any]
    ] = {}  # Maps character position to source paragraph

    for chunk_record in chunk_records:
        text = chunk_record["text"]
        source_paragraph = chunk_record.get("source_paragraph")

        start_pos = len(tokens)
        tokens.extend(list(text))
        end_pos = len(tokens)

        # Map character positions to source paragraph
        for i in range(start_pos, end_pos):
            if source_paragraph is not None:
                metadata_map[i] = source_paragraph

    # Apply sliding window
    windows = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(n, start + chunk_size)
        window_text = "".join(tokens[start:end])

        if window_text:
            # Find the first contributing paragraph for this window
            first_paragraph = None
            for i in range(start, end):
                if i in metadata_map and metadata_map[i] is not None:
                    first_paragraph = metadata_map[i]
                    break

            windows.append({"text": window_text, "source_paragraph": first_paragraph})

        if end == n:
            break
        start = max(start + chunk_size - chunk_overlap, start + 1)

    return windows


def apply_recursive_strategy(
    paragraphs: List[Dict[str, Any]], params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Apply recursive chunking strategy with metadata preservation.

    This strategy splits text by separators first, then applies sliding window
    with overlap. Metadata from source paragraphs is preserved and propagated
    to the final chunks.
    """
    if not paragraphs:
        return []

    separators: Optional[List[str]] = params.get("separators")
    chunk_size_param = params.get("chunk_size")
    chunk_overlap: int = int(params.get("chunk_overlap", 200))

    # Process each paragraph individually to preserve metadata
    all_chunk_records = []

    for paragraph in paragraphs:
        text = paragraph.get("text", "")
        if not text.strip():
            continue

        # Split this paragraph by separators, preserving its metadata
        paragraph_chunks = _split_by_separators_with_metadata(
            text, separators, paragraph
        )
        all_chunk_records.extend(paragraph_chunks)

    if not all_chunk_records:
        return []

    # Apply sliding window with overlap, preserving metadata
    if chunk_size_param is None:
        # User didn't set chunk_size, trust semantic splitting completely
        windows = all_chunk_records
    else:
        chunk_size = int(chunk_size_param)
        windows = _window_with_overlap_and_metadata(
            all_chunk_records, chunk_size, chunk_overlap
        )

    # Create final chunk records with preserved metadata
    return [
        _create_chunk_record(w["text"].strip(), w["source_paragraph"])
        for w in windows
        if w["text"].strip()
    ]


def apply_markdown_strategy(
    paragraphs: List[Dict[str, Any]], params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    text = _join_paragraphs(paragraphs)
    if not text:
        return []
    chunk_size_param = params.get("chunk_size")
    chunk_overlap: int = int(params.get("chunk_overlap", 200))
    separators: Optional[List[str]] = params.get("separators")

    # Split by Markdown headers, keep header with its section
    lines = text.splitlines()
    sections: List[str] = []
    current: List[str] = []
    header_pattern = re.compile(r"^\s{0,3}#{1,6}\s+")
    for line in lines:
        if header_pattern.match(line):
            if current:
                sections.append("\n".join(current))
                current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))

    # If no headers found, fallback to recursive
    if len(sections) <= 1:
        return apply_recursive_strategy(paragraphs, params)

    # For each section, further split using splitter if available, otherwise character-level
    chunks: List[Dict[str, Any]] = []
    for sec in sections:
        if separators and separators != DEFAULT_SEPARATORS:
            # Use splitter for semantic splitting
            section_chunks = _split_by_separators(sec, separators)
            if section_chunks:
                if chunk_size_param is None:
                    # User didn't set chunk_size, trust semantic splitting completely
                    windows = section_chunks
                else:
                    # User set chunk_size, check if semantic chunks exceed limit
                    chunk_size = int(chunk_size_param)
                    total_chars = sum(len(chunk) for chunk in section_chunks)
                    if total_chars <= chunk_size:
                        # Semantic chunks fit within size limit, use them directly
                        windows = section_chunks
                    else:
                        # Semantic chunks too large, apply semantic-level sliding window
                        windowed_chunks = _window_with_overlap_and_metadata(
                            [{"text": chunk} for chunk in section_chunks],
                            chunk_size,
                            chunk_overlap,
                        )
                        windows = [w["text"] for w in windowed_chunks]
            else:
                windows = [sec]  # Use whole section if splitter produces no chunks
        else:
            # Fallback to character-level splitting
            if chunk_size_param is None:
                # No chunk_size set, use whole section
                windows = [sec]
            else:
                chunk_size = int(chunk_size_param)
                tokens = list(sec)
                windows = _window_with_overlap(tokens, chunk_size, chunk_overlap)

        if not windows:
            continue
        for w in windows:
            if w.strip():
                chunks.append(_create_chunk_record(w.strip()))
    return chunks


def apply_fixed_size_strategy(
    paragraphs: List[Dict[str, Any]], params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    text = _join_paragraphs(paragraphs)
    if not text:
        return []
    chunk_size_param = params.get("chunk_size")
    chunk_overlap: int = int(params.get("chunk_overlap", 0))

    if chunk_size_param is None:
        # User didn't set chunk_size, return whole text as one chunk
        return [_create_chunk_record(text.strip())]
    else:
        chunk_size = int(chunk_size_param)
        tokens = list(text)
        windows = _window_with_overlap(tokens, chunk_size, chunk_overlap)
        return [_create_chunk_record(w.strip()) for w in windows if w.strip()]
