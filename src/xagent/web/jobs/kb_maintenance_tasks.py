"""Scheduled LanceDB index maintenance for the knowledge base (#1557).

Maintenance used to hang off the ingestion hot path, so its cost scaled with
document count and it stopped entirely whenever ingestion was idle -- exactly
when a stale index goes unnoticed. On a timer it is the reverse: the sweep has
no ingestion context to scope it, so it sweeps every table, but only once per
interval. The per-table gating (``should_compact``) and the per-table advisory
lock inside ``compact_tables`` are what keep a full sweep cheap.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.tools.core.RAG_tools.storage.factory import StorageFactory
from .celery_app import celery_app

logger = logging.getLogger(__name__)


def _failing_tables() -> dict[str, int]:
    """Tables whose maintenance keeps failing, for the task result.

    The escalated ERROR line the store logs at the same threshold is the alert
    itself: the sweep runs in the Celery worker, and ops_signals/health is
    per-process and served by the backend, so it cannot see this.
    """
    from ...core.tools.core.RAG_tools.storage.lancedb_stores import (
        failing_maintenance_tables,
    )

    return failing_maintenance_tables()


@celery_app.task(name="xagent.web.jobs.kb_maintenance_tasks.compact_kb_storage")
def compact_kb_storage() -> dict[str, Any]:
    """Celery Beat entrypoint for KB compaction and FTS/vector index refresh."""
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


@celery_app.task(name="xagent.web.jobs.kb_maintenance_tasks.retrain_kb_vector_indexes")
def retrain_kb_vector_indexes() -> dict[str, Any]:
    """Celery Beat entrypoint for the coarse vector-index retrain.

    Its own schedule, deliberately: a retrain on every compaction pass is what
    the fix for the never-matching index check removed.
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
