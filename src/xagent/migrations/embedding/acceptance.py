"""Minimal per-collection acceptance before an embedding-model switch.

Per-document row counts, owners and empty vectors only: enough to stop a switch
that would hide or duplicate data. Retrieval-quality metrics are a separate
phase and replace this module's checks through the same callable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_candidates,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_metadata_store,
    get_vector_index_store,
)
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import list_table_names
from xagent.migrations.embedding.scope import SKIP_NO_OWNER

logger = logging.getLogger(__name__)

DocRows = Dict[str, Tuple[int, FrozenSet[Any]]]


@dataclass
class AcceptanceResult:
    ok: bool
    checks: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    # Skipped documents the old model still answers for: switching would hide them.
    uncovered: List[str] = field(default_factory=list)


def embeddings_table(model_id: str) -> Optional[str]:
    """The existing embeddings table for ``model_id``, if any."""
    conn = get_metadata_store().get_raw_connection()
    names = set(list_table_names(conn))
    for candidate in embeddings_table_candidates(model_id):
        if candidate in names:
            return candidate
    return None


def _doc_rows(collection: str, table: str) -> DocRows:
    counts: Dict[str, int] = {}
    owners: Dict[str, set] = {}
    for batch in get_vector_index_store().iter_batches(
        table_name=table,
        columns=["doc_id", "user_id"],
        filters={"collection": collection},
        user_id=None,
        is_admin=True,
    ):
        for row in batch.to_pylist():
            doc_id = row.get("doc_id")
            if doc_id is None:
                continue
            key = str(doc_id)
            counts[key] = counts.get(key, 0) + 1
            owners.setdefault(key, set()).add(row.get("user_id"))
    return {key: (counts[key], frozenset(owners[key])) for key in counts}


def _null_vector_rows(collection: str, table: str) -> int:
    nulls = 0
    for batch in get_vector_index_store().iter_batches(
        table_name=table,
        columns=["vector"],
        filters={"collection": collection},
        user_id=None,
        is_admin=True,
    ):
        nulls += int(batch.column("vector").null_count)
    return nulls


def check_collection(
    collection: str,
    *,
    source_model_id: str,
    target_model_id: str,
    skipped: Optional[Mapping[str, str]] = None,
) -> AcceptanceResult:
    """Compare the old and new model's rows, per document, for one collection.

    ``skipped`` maps doc_id to the scope's skip reason; those documents are left
    out of both sides. A skipped document the old model still has rows for is
    ``uncovered`` unless it is ownerless (admin-only legacy rows).
    """
    source_table = embeddings_table(source_model_id)
    target_table = embeddings_table(target_model_id)
    result = AcceptanceResult(ok=False)
    if source_table is None or target_table is None:
        result.failures.append(
            f"missing embeddings table: source={source_table} target={target_table}"
        )
        return result

    source = _doc_rows(collection, source_table)
    target = _doc_rows(collection, target_table)
    skipped = skipped or {}
    ownerless = {doc for doc, reason in skipped.items() if reason == SKIP_NO_OWNER}
    result.uncovered = sorted((set(skipped) - ownerless) & source.keys())
    result.checks["ownerless_excluded"] = len(ownerless)
    result.checks["excluded"] = len(skipped)
    if result.uncovered:
        result.failures.append(
            f"{len(result.uncovered)} skipped documents still have old-model rows"
        )

    source = {doc: rows for doc, rows in source.items() if doc not in skipped}
    target = {doc: rows for doc, rows in target.items() if doc not in skipped}
    result.checks["source_count"] = sum(count for count, _ in source.values())
    result.checks["target_count"] = sum(count for count, _ in target.values())
    mismatches = sorted(
        doc
        for doc in source.keys() | target.keys()
        if source.get(doc) != target.get(doc)
    )
    result.checks["document_mismatches"] = mismatches[:20]
    result.checks["document_mismatch_count"] = len(mismatches)
    if mismatches:
        result.failures.append(
            f"{len(mismatches)} documents differ in row count, owner or presence"
        )

    null_vectors = _null_vector_rows(collection, target_table)
    result.checks["null_vectors"] = null_vectors
    if null_vectors:
        result.failures.append(f"{null_vectors} new-model rows have a null vector")

    result.ok = not result.failures
    return result
