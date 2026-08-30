"""LanceDB index maintenance for the knowledge base (#1557).

Maintenance used to hang off the ingestion hot path, so its cost scaled with
document count and it stopped entirely whenever ingestion was idle -- exactly
when a stale index goes unnoticed. On a timer it is the reverse: the sweep has
no ingestion context to scope it, so it sweeps every table, but only once per
interval. The per-table gating (``should_compact``) and the per-table advisory
lock inside ``compact_tables`` are what keep a full sweep cheap.

Driven by Celery Beat where Celery is enabled, and by the in-process loop
below where it is not (``get_celery_enabled`` defaults to False, so
Gunicorn-only and local deployments have no worker) -- the same reasoning as
``run_orphan_upload_gc_loop``. Only one of the two runs in a deployment.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...core.tools.core.RAG_tools.storage.factory import StorageFactory

logger = logging.getLogger(__name__)


def _failing_tables() -> dict[str, int]:
    """Tables whose maintenance keeps failing, for the caller's result.

    The escalated ERROR line the store logs at the same threshold is the alert
    itself: ops_signals/health is per-process and served by the backend, so it
    cannot see a sweep running in the Celery worker.
    """
    from ...core.tools.core.RAG_tools.storage.lancedb_stores import (
        failing_maintenance_tables,
    )

    return failing_maintenance_tables()


def sweep_kb_storage() -> dict[str, Any]:
    """Compact every degraded KB table and refresh its FTS index."""
    store = StorageFactory.get_factory().get_vector_index_store()
    tables = list(store.list_table_names())
    compacted = store.compact_tables(tables) if tables else []
    if compacted:
        logger.info("Compacted LanceDB tables: %s", ", ".join(compacted))
    return {
        "status": "ok",
        "scanned": len(tables),
        "compacted": compacted,
        "failing": _failing_tables(),
    }


def retrain_kb_vector_indexes() -> dict[str, Any]:
    """Rebuild every KB vector index from scratch.

    Gated apart from :func:`sweep_kb_storage`, deliberately: a retrain on every
    maintenance pass is what the fix for the never-matching existence check
    removed.
    """
    store = StorageFactory.get_factory().get_vector_index_store()
    retrained = [
        name
        for name in store.list_table_names()
        if name.startswith("embeddings_") and store.retrain_vector_index(name)
    ]
    if retrained:
        logger.info("Retrained vector indices: %s", ", ".join(retrained))
    return {
        "status": "ok",
        "retrained": retrained,
        "failing": _failing_tables(),
    }


async def run_kb_maintenance_loop(*, poll_interval_seconds: int) -> None:
    """Run the compaction sweep on a timer inside the FastAPI process.

    Compaction only. The retrain's weekly cadence rides Celery Beat's own
    schedule; reproducing it here would need a restart-surviving last-run
    marker, and this loop keeps no state.
    """
    while True:
        await asyncio.sleep(poll_interval_seconds)
        try:
            await asyncio.to_thread(sweep_kb_storage)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("KB index maintenance sweep failed")
