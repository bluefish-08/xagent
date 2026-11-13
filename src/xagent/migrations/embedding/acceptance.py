"""Minimal per-collection acceptance before an embedding-model switch.

Counts, owners and empty vectors only: enough to stop a switch that would hide
or duplicate data. Retrieval-quality metrics are a separate phase and replace
this module's checks through the same callable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_candidates,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_metadata_store,
    get_vector_index_store,
)
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import list_table_names

logger = logging.getLogger(__name__)


@dataclass
class AcceptanceResult:
    ok: bool
    checks: Dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def embeddings_table(model_id: str) -> Optional[str]:
    """The existing embeddings table for ``model_id``, if any."""
    conn = get_metadata_store().get_raw_connection()
    names = set(list_table_names(conn))
    for candidate in embeddings_table_candidates(model_id):
        if candidate in names:
            return candidate
    return None


def _count_rows(collection: str, table: str) -> int:
    return get_vector_index_store().count_rows_or_zero(
        table_name=table,
        filters={"collection": collection},
        user_id=None,
        is_admin=True,
    )


def _owners(collection: str, table: str) -> Dict[str, frozenset]:
    store = get_vector_index_store()
    owners: Dict[str, set] = {}
    for batch in store.iter_batches(
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
            raw = row.get("user_id")
            owners.setdefault(str(doc_id), set()).add(raw)
    return {doc_id: frozenset(values) for doc_id, values in owners.items()}


def _null_vector_rows(collection: str, table: str) -> int:
    store = get_vector_index_store()
    nulls = 0
    for batch in store.iter_batches(
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
) -> AcceptanceResult:
    """Compare the old and new model's rows for one collection."""
    source_table = embeddings_table(source_model_id)
    target_table = embeddings_table(target_model_id)
    result = AcceptanceResult(ok=False)
    if source_table is None or target_table is None:
        result.failures.append(
            f"missing embeddings table: source={source_table} target={target_table}"
        )
        return result

    source_count = _count_rows(collection, source_table)
    target_count = _count_rows(collection, target_table)
    result.checks["source_count"] = source_count
    result.checks["target_count"] = target_count
    if source_count != target_count:
        result.failures.append(
            f"row count differs: old model {source_count}, new model {target_count}"
        )

    source_owners = _owners(collection, source_table)
    target_owners = _owners(collection, target_table)
    owner_mismatches = [
        doc_id
        for doc_id in sorted(source_owners.keys() | target_owners.keys())
        if source_owners.get(doc_id) != target_owners.get(doc_id)
    ]
    result.checks["owner_mismatches"] = owner_mismatches[:20]
    result.checks["owner_mismatch_count"] = len(owner_mismatches)
    if owner_mismatches:
        result.failures.append(
            f"{len(owner_mismatches)} documents have different owners or presence"
        )

    null_vectors = _null_vector_rows(collection, target_table)
    result.checks["null_vectors"] = null_vectors
    if null_vectors:
        result.failures.append(f"{null_vectors} new-model rows have a null vector")

    result.ok = not result.failures
    return result
