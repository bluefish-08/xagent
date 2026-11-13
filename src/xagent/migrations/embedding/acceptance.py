"""Per-document acceptance before an embedding-model switch.

The check is set equality on what search can reach: every chunk the source
model's table holds for a document must have a row under the target model with
the same owner, and none that the source does not have. Retrieval-quality
metrics are a separate phase and replace this module's checks through the same
callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.migrations.embedding.scope import (
    SKIP_NO_OWNER,
    connection,
    embeddings_table,
    iter_rows,
)


@dataclass
class AcceptanceResult:
    ok: bool
    checks: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)


def _rows(
    collection: str, model_id: str, conn: Any
) -> Optional[Dict[str, Dict[str, Set[Optional[int]]]]]:
    """doc_id -> chunk_id -> owners, or ``None`` when the model has no table."""
    table = embeddings_table(model_id, conn)
    if table is None:
        return None
    rows: Dict[str, Dict[str, Set[Optional[int]]]] = {}
    where = f"collection = '{collection}'"
    for row in iter_rows(conn, table, where, ["doc_id", "chunk_id", "user_id"]):
        doc_id = str(row.get("doc_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        if not doc_id or not chunk_id:
            continue
        owner = row.get("user_id")
        rows.setdefault(doc_id, {}).setdefault(chunk_id, set()).add(owner)
    return rows


def _ownerless(docs: Dict[str, Dict[str, Set[Optional[int]]]]) -> Set[str]:
    """Documents whose every searchable chunk has no owner."""
    return {
        doc_id
        for doc_id, chunks in docs.items()
        if all(owners == {None} for owners in chunks.values())
    }


def check_collection(
    collection: str,
    *,
    source_model_id: str,
    target_model_id: str,
    skipped: Optional[Set[str]] = None,
    conn: Optional[Any] = None,
) -> AcceptanceResult:
    """Compare the source and target models' searchable chunks per document.

    ``skipped`` names the documents the migration deliberately left out
    (ownerless ones); they are excluded from both sides.
    """
    conn = conn if conn is not None else connection()
    source = _rows(collection, source_model_id, conn)
    target = _rows(collection, target_model_id, conn)
    result = AcceptanceResult(ok=False)
    if source is None:
        result.failures.append(
            f"the source model's table for '{source_model_id}' does not exist"
        )
        return result
    # No target table means nothing was written for it: an empty row set, not a
    # read failure, so an empty or all-ownerless collection can still switch.
    target = target or {}
    excluded = _ownerless(source) | set(skipped or ())
    source = {doc: chunks for doc, chunks in source.items() if doc not in excluded}
    target = {doc: chunks for doc, chunks in target.items() if doc not in excluded}

    result.checks["ownerless_excluded"] = len(excluded)
    result.checks["documents"] = len(source)
    result.checks["chunks"] = sum(len(chunks) for chunks in source.values())
    mismatches = sorted(
        doc
        for doc in source.keys() | target.keys()
        if source.get(doc) != target.get(doc)
    )
    result.checks["document_mismatches"] = mismatches[:20]
    result.checks["document_mismatch_count"] = len(mismatches)
    if mismatches:
        result.failures.append(
            f"{len(mismatches)} documents differ in their searchable chunks"
        )
    nulls = _null_vectors(collection, target_model_id, conn, excluded)
    result.checks["null_vectors"] = nulls
    if nulls:
        result.failures.append(f"{nulls} target rows have a null vector")
    result.ok = not result.failures
    return result


def _null_vectors(collection: str, model_id: str, conn: Any, excluded: Set[str]) -> int:
    table = embeddings_table(model_id, conn)
    if table is None:
        return 0
    nulls = 0
    where = f"collection = '{collection}'"
    opened = conn.open_table(table)
    try:
        for batch in opened.to_batches(filter=where, batch_size=1000):
            for row in batch.select(["doc_id", "vector"]).to_pylist():
                if (
                    row.get("vector") is None
                    and str(row.get("doc_id") or "") not in excluded
                ):
                    nulls += 1
    finally:
        _safe_close_table(opened)
    return nulls


__all__ = ["AcceptanceResult", "SKIP_NO_OWNER", "check_collection"]
