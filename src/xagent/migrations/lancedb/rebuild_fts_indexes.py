"""LanceDB migration: rebuild FTS indexes on ``embeddings_*`` tables.

The FTS tokenizer is baked into the index at build time, so tables indexed
before the jieba switch keep the old tokenizer until the index is rebuilt.
``ensure_indexes`` only creates a missing index and the automatic rebuild in
``compact_tables`` needs fresh ingestion plus a fragment/version threshold, so a
quiescent knowledge base never picks the new tokenizer up on its own.

Rebuilds go through ``LanceDBVectorIndexStore.trigger_reindex``, which replaces
the FTS index in place and is safe to re-run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from xagent.core.tools.core.RAG_tools.core.config import DEFAULT_INDEX_POLICY
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EMBEDDINGS_PREFIX = "embeddings_"


def list_embeddings_tables(conn: Any) -> List[str]:
    """Embeddings tables only: they are the sole carriers of an FTS index."""
    return sorted(n for n in conn.table_names() if n.startswith(EMBEDDINGS_PREFIX))


def describe_indexes(conn: Any, table_name: str) -> Dict[str, Any]:
    """Snapshot of the index state, for before/after comparison.

    LanceDB does not read the tokenizer back out of a built index, so the
    operator-visible proof of a rebuild is the table version plus the indexed
    row counts; the tokenizer being written is reported from the policy.
    """
    table = None
    try:
        table = conn.open_table(table_name)
        indexes = {}
        for idx in table.list_indices():
            entry: Dict[str, Any] = {
                "type": idx.index_type,
                "columns": list(idx.columns),
            }
            try:
                stats = table.index_stats(idx.name)
                entry["indexed_rows"] = stats.num_indexed_rows
                entry["unindexed_rows"] = stats.num_unindexed_rows
            except Exception as e:  # noqa: BLE001
                entry["stats_error"] = str(e)
            indexes[idx.name] = entry
        return {"version": table.version, "indexes": indexes}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    finally:
        _safe_close_table(table)


def _has_fts(snapshot: Dict[str, Any]) -> bool:
    indexes = snapshot.get("indexes") or {}
    return any(entry["type"] == "FTS" for entry in indexes.values())


def _fts_rebuilt(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    """Whether the FTS index was actually rewritten.

    ``trigger_reindex`` reports whether ``optimize`` succeeded; it logs a failed
    FTS rebuild and still returns True, and returns False when only the
    unrelated compaction step failed. The table version is the signal for the
    one step this script exists for.
    """
    if "error" in before or "error" in after:
        return False
    return _has_fts(after) and after["version"] > before["version"]


def rebuild_fts_indexes(
    table: Optional[str] = None,
    dry_run: bool = False,
    conn: Any = None,
) -> Dict[str, Any]:
    """Rebuild the FTS index of every ``embeddings_*`` table, or just one."""
    conn = conn or get_connection_from_env()
    tables = list_embeddings_tables(conn)

    if table is not None:
        if table not in tables:
            raise ValueError(
                f"{table!r} is not an embeddings table; found: {tables or 'none'}"
            )
        tables = [table]

    target_params = {"with_position": True, **(DEFAULT_INDEX_POLICY.fts_params or {})}
    logger.info("Tables to rebuild (%d): %s", len(tables), tables or "none")
    logger.info("Target FTS params: %s", target_params)

    if dry_run:
        for name in tables:
            logger.info("[dry-run] %s before: %s", name, describe_indexes(conn, name))
        return {
            "tables": tables,
            "succeeded": [],
            "skipped": [],
            "failed": [],
            "dry_run": True,
        }

    store = LanceDBVectorIndexStore()
    succeeded: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []

    for name in tables:
        before = describe_indexes(conn, name)
        # trigger_reindex would still run the full compaction on a table that
        # carries no FTS index, and rebuild nothing.
        if "error" not in before and not _has_fts(before):
            logger.info("%s carries no FTS index; skipping", name)
            skipped.append(name)
            continue

        started = time.monotonic()
        try:
            ok = store.trigger_reindex(name)
        except Exception as e:  # noqa: BLE001
            logger.error("%s: rebuild raised %s: %s", name, type(e).__name__, e)
            ok = False
        elapsed = time.monotonic() - started
        after = describe_indexes(conn, name)

        rebuilt = _fts_rebuilt(before, after)
        logger.info("%s before: %s", name, before)
        logger.info("%s after:  %s", name, after)
        logger.info(
            "%s took %.2fs, fts_rebuilt=%s, optimize_ok=%s",
            name,
            elapsed,
            rebuilt,
            ok,
        )
        if not ok:
            logger.warning(
                "%s: compaction did not finish; the FTS index is unaffected", name
            )
        (succeeded if rebuilt else failed).append(name)

    logger.info("Succeeded (%d): %s", len(succeeded), succeeded or "none")
    logger.info("Skipped (%d): %s", len(skipped), skipped or "none")
    logger.info("Failed (%d): %s", len(failed), failed or "none")
    return {
        "tables": tables,
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
        "dry_run": False,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild FTS indexes on LanceDB embeddings tables.\n\n"
        "The tokenizer is stored inside the index, so tables built before the "
        "jieba switch need an explicit rebuild. Safe to re-run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--table", help="Rebuild only this table (default: all)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the tables that would be rebuilt without changing anything",
    )
    args = parser.parse_args(argv)

    try:
        result = rebuild_fts_indexes(table=args.table, dry_run=args.dry_run)
    except Exception as e:  # noqa: BLE001
        logger.error("Rebuild failed: %s", e, exc_info=True)
        return 2

    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
