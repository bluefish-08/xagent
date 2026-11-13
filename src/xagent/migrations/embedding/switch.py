"""Switch one collection to a new embedding model inside a freeze window.

The window is: freeze, wait for the writers the freeze let in, embed what the
target model still lacks (including documents ingested since the precompute),
drop target rows for documents deleted since then, run the minimal acceptance
check, and flip the binding. Any failure before the flip unfreezes the
collection and leaves it on the old model; a crashed run leaves it frozen and a
rerun resumes from the same place.

    python -m xagent.migrations.embedding.switch --model <model-id> \
        --collection <name> [--wait-seconds N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    get_collection_sync,
)
from xagent.core.tools.core.RAG_tools.management.embedding_switch import (
    switch_collection_embedding_model,
)
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    IDLE,
    CollectionMigrationState,
    read_migration_state,
    write_migration_state,
)
from xagent.core.tools.core.RAG_tools.management.write_leases import (
    live_leases,
    wait_for_no_leases,
)
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from xagent.core.tools.core.RAG_tools.utils.model_resolver import (
    resolve_embedding_adapter,
)
from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context
from xagent.migrations.embedding.acceptance import (
    AcceptanceResult,
    check_collection,
    embeddings_table,
)
from xagent.migrations.embedding.common import (
    MigrationCliError,
    PacedEmbedding,
    RateLimiter,
    collection_backend,
    resolve_collection_names,
)
from xagent.migrations.embedding.embed import SKIP_OTHER_ENGINE, embed_collection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_WAIT_SECONDS = 15 * 60
DEFAULT_POLL_SECONDS = 2.0
# Admissions can race the freeze by one adjacent call; re-check right before
# the flip so a writer that slipped in still delays the switch.
FINAL_LEASE_WAIT_SECONDS = 30.0
STALE_FREEZE_SECONDS = 60 * 60

RESULT_SWITCHED = "switched"
RESULT_ALREADY = "already-switched"
RESULT_REFUSED = "refused"
RESULT_FAILED = "failed"

MODE_SWITCH = "switch"
MODE_ROLLBACK = "rollback"


class WindowRefused(Exception):
    """The collection stays on the old model; ``reason`` goes in the report."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class SwitchReport:
    name: str
    target_model_id: str
    mode: str = MODE_SWITCH
    result: str = RESULT_REFUSED
    reason: str = ""
    source_model_id: Optional[str] = None
    window_seconds: float = 0.0
    documents_embedded: int = 0
    chunks_embedded: int = 0
    pruned_rows: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    acceptance: Dict[str, Any] = field(default_factory=dict)


class InflightJobWatcher:
    """Running KB ingest jobs for a collection, read from the web database."""

    def __init__(self) -> None:
        self._session_factory: Any = None
        self._init_lock = threading.Lock()

    def _session(self) -> Any:
        if self._session_factory is None:
            with self._init_lock:
                if self._session_factory is None:
                    from xagent.web.models.database import (
                        configure_db,
                        get_session_local,
                    )

                    configure_db(read_only=True)
                    self._session_factory = get_session_local()
        return self._session_factory()

    def running(self, collection: str) -> List[Dict[str, Any]]:
        from xagent.web.models.background_job import (
            BackgroundJob,
            BackgroundJobStatus,
            BackgroundJobType,
        )

        db = self._session()
        try:
            rows = (
                db.query(BackgroundJob)
                .filter(
                    BackgroundJob.status == BackgroundJobStatus.RUNNING.value,
                    BackgroundJob.job_type.in_(
                        [
                            BackgroundJobType.KB_INGEST_DOCUMENT.value,
                            BackgroundJobType.KB_INGEST_WEB.value,
                        ]
                    ),
                )
                .all()
            )
            return [
                {"job_id": row.id, "job_type": row.job_type, "status": row.status}
                for row in rows
                if str(dict(row.payload or {}).get("collection") or "") == collection
            ]
        finally:
            db.close()


def switch_collection(
    name: str,
    target_model_id: str,
    *,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    job_watcher: Optional[InflightJobWatcher] = None,
    acceptance: Callable[..., AcceptanceResult] = check_collection,
    mode: str = MODE_SWITCH,
) -> SwitchReport:
    """Run the switch window for one collection."""
    started = time.monotonic()
    report = SwitchReport(name=name, target_model_id=target_model_id, mode=mode)

    def finish(result: str, reason: str = "") -> SwitchReport:
        report.result = result
        report.reason = reason
        report.window_seconds = round(time.monotonic() - started, 3)
        return report

    if collection_backend(name) != "lancedb":
        return finish(RESULT_REFUSED, SKIP_OTHER_ENGINE)
    try:
        info = get_collection_sync(name)
    except ValueError:
        return finish(RESULT_REFUSED, "unknown-collection")
    source_model_id = (info.embedding_model_id or "").strip() or None
    if source_model_id is None:
        # Nothing was precomputed for an unbound collection; there is no old
        # model to compare against either.
        return finish(RESULT_REFUSED, "unbound")
    report.source_model_id = source_model_id

    try:
        canonical, embedding_config, adapter = _resolve_target(target_model_id)
    except MigrationCliError as exc:
        return finish(RESULT_REFUSED, str(exc))
    except Exception as exc:  # noqa: BLE001 - provider/hub failure, not a window
        return finish(RESULT_REFUSED, f"target-not-configured: {exc}")

    state = read_migration_state(name)
    resumed = state.state == FROZEN
    if resumed:
        frozen_target = (state.target_model_id or "").strip()
        if frozen_target and frozen_target != canonical:
            return finish(RESULT_REFUSED, f"frozen-for-another-target: {frozen_target}")
        if state.updated_at is not None and _age_seconds(state.updated_at) > (
            STALE_FREEZE_SECONDS
        ):
            logger.warning(
                "%s has been frozen since %s; check the previous run before resuming",
                name,
                state.updated_at,
            )
        logger.warning(
            "%s was already frozen; resuming the window (started %s)",
            name,
            state.updated_at,
        )

    if source_model_id == canonical:
        if resumed:
            _unfreeze(name, state.source_model_id or source_model_id, canonical)
        return finish(RESULT_ALREADY)

    if not resumed:
        _freeze(name, source_model_id, canonical)

    cfg = IngestionConfig(embedding_model_id=canonical)
    try:
        _run_window(
            report,
            name,
            source_model_id=source_model_id,
            canonical=canonical,
            embedding_config=embedding_config,
            adapter=PacedEmbedding(adapter, RateLimiter(0.0)),
            cfg=cfg,
            wait_seconds=wait_seconds,
            poll_seconds=poll_seconds,
            job_watcher=job_watcher,
            acceptance=acceptance,
        )
    except WindowRefused as exc:
        _unfreeze(name, source_model_id, canonical)
        return finish(RESULT_REFUSED, exc.reason)
    except Exception as exc:  # noqa: BLE001 - keep the collection usable
        _unfreeze(name, source_model_id, canonical)
        logger.exception("%s: switch failed", name)
        return finish(RESULT_FAILED, f"{type(exc).__name__}: {exc}")

    _unfreeze(name, source_model_id, canonical)
    return finish(RESULT_SWITCHED)


def _age_seconds(stamp: Any) -> float:
    if not isinstance(stamp, datetime):
        return 0.0
    current = datetime.now(timezone.utc).replace(tzinfo=None)
    value = stamp.replace(tzinfo=None) if stamp.tzinfo else stamp
    return max(0.0, (current - value).total_seconds())


def _run_window(
    report: SwitchReport,
    name: str,
    *,
    source_model_id: str,
    canonical: str,
    embedding_config: EmbeddingModelConfig,
    adapter: BaseEmbedding,
    cfg: IngestionConfig,
    wait_seconds: float,
    poll_seconds: float,
    job_watcher: Optional[InflightJobWatcher],
    acceptance: Callable[..., AcceptanceResult],
) -> None:
    watcher = job_watcher if job_watcher is not None else InflightJobWatcher()

    stragglers = _wait_until_idle(
        name, timeout=wait_seconds, poll=poll_seconds, watcher=watcher
    )
    if stragglers:
        raise WindowRefused("in-flight-writers: " + ", ".join(stragglers[:10]))

    # A rename admitted before the freeze leaves the row on the old name.
    try:
        get_collection_sync(name)
    except ValueError:
        raise WindowRefused("renamed") from None

    item = embed_collection(
        name,
        model_id=canonical,
        embedding_config=embedding_config,
        embedding_adapter=adapter,
        cfg=cfg,
        dry_run=False,
    )
    report.documents_embedded = item.documents_embedded
    report.chunks_embedded = item.chunks_embedded
    report.skipped = dict(item.skipped)
    if item.collection_skip or item.aborted or item.failures:
        raise WindowRefused(
            "embed-failed: "
            + (item.collection_skip or f"{len(item.failures)} documents failed")
        )
    if item.skipped:
        # The old model cannot cover these either; the collection would switch
        # with documents that no longer answer. Refuse and report the reasons.
        raise WindowRefused(
            "uncovered-documents: "
            + ", ".join(f"{k}={v}" for k, v in sorted(item.skipped.items()))
        )

    report.pruned_rows = prune_orphan_documents(name, target_model_id=canonical)

    verdict = acceptance(
        name, source_model_id=source_model_id, target_model_id=canonical
    )
    report.acceptance = dict(verdict.checks)
    if not verdict.ok:
        raise WindowRefused("acceptance-failed: " + "; ".join(verdict.failures))

    leftovers = wait_for_no_leases(
        name,
        timeout_seconds=FINAL_LEASE_WAIT_SECONDS,
        poll_seconds=poll_seconds,
    )
    if leftovers:
        raise WindowRefused(
            "writers-restarted: "
            + ", ".join(f"{item.operation}@{item.owner}" for item in leftovers[:10])
        )

    switch_collection_embedding_model(name, canonical)
    return None


def prune_orphan_documents(name: str, *, target_model_id: str) -> int:
    """Delete target rows for documents that no longer have a ``documents`` row."""
    table = embeddings_table(target_model_id)
    if table is None:
        return 0
    store = get_vector_index_store()
    present = _doc_ids(name, "documents", store)
    target_docs = _doc_ids(name, table, store)
    deleted = 0
    for doc_id in sorted(target_docs - present):
        deleted += store.delete_embedding_records(
            name,
            doc_id,
            model_tag=target_model_id,
            user_id=None,
            is_admin=True,
        )
    return deleted


def _doc_ids(name: str, table: str, store: Any) -> set:
    doc_ids: set = set()
    for batch in store.iter_batches(
        table_name=table,
        columns=["doc_id"],
        filters={"collection": name},
        user_id=None,
        is_admin=True,
    ):
        for row in batch.to_pylist():
            doc_id = row.get("doc_id")
            if doc_id is not None:
                doc_ids.add(str(doc_id))
    return doc_ids


def _wait_until_idle(
    name: str,
    *,
    timeout: float,
    poll: float,
    watcher: InflightJobWatcher,
) -> List[str]:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        jobs = watcher.running(name)
        leases = live_leases(name)
        if not jobs and not leases:
            return []
        if time.monotonic() >= deadline:
            return [f"job:{job['job_id']}" for job in jobs] + [
                f"lease:{lease.operation}@{lease.owner}" for lease in leases
            ]
        time.sleep(poll)


def _resolve_target(
    target_model_id: str,
) -> tuple[str, EmbeddingModelConfig, BaseEmbedding]:
    with user_scope_context(None, True):
        config, adapter = resolve_embedding_adapter(target_model_id)
    canonical = (config.id or "").strip()
    if not canonical:
        raise MigrationCliError(
            f"'{target_model_id}' resolves to a model without a hub id"
        )
    if canonical != target_model_id.strip():
        raise MigrationCliError(
            f"'{target_model_id}' resolves to '{canonical}'; pass the configured id"
        )
    if not isinstance(config.dimension, int):
        raise MigrationCliError(f"'{canonical}' has no dimension; cannot bind it")
    return canonical, config, adapter


def _freeze(name: str, source_model_id: str, target_model_id: str) -> None:
    write_migration_state(
        name,
        CollectionMigrationState(
            state=FROZEN,
            source_model_id=source_model_id,
            target_model_id=target_model_id,
        ),
    )
    logger.info(
        "%s: frozen (source %s, target %s)", name, source_model_id, target_model_id
    )


def _unfreeze(name: str, source_model_id: str, target_model_id: str) -> None:
    # The row stays for the rollback window: it records what this switch did.
    write_migration_state(
        name,
        CollectionMigrationState(
            state=IDLE,
            source_model_id=source_model_id,
            target_model_id=target_model_id,
        ),
    )


def switch_collections(
    target_model_id: str,
    *,
    collections: Optional[Sequence[str]] = None,
    team: Optional[int] = None,
    concurrency: int = 1,
    mode: str = MODE_SWITCH,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
) -> List[SwitchReport]:
    """Run the window over every collection in scope, ``concurrency`` at a time."""
    names = resolve_collection_names(collections=collections, team=team)
    if not names:
        logger.warning("No collections in scope; check --collection/--team")
        return []
    watcher = InflightJobWatcher()

    def _one(name: str) -> SwitchReport:
        try:
            return switch_collection(
                name,
                target_model_id,
                wait_seconds=wait_seconds,
                job_watcher=watcher,
                mode=mode,
            )
        except Exception as exc:  # noqa: BLE001 - one collection must not stop the rest
            logger.exception("%s: window crashed", name)
            return SwitchReport(
                name=name,
                target_model_id=target_model_id,
                mode=mode,
                result=RESULT_FAILED,
                reason=f"{type(exc).__name__}: {exc}",
            )

    workers = max(1, min(concurrency, len(names)))
    if workers == 1:
        return [_one(name) for name in names]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, names))


def _add_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Target embedding model id")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--collection",
        action="append",
        metavar="NAME",
        help="Restrict to this collection; repeatable",
    )
    scope.add_argument(
        "--team",
        type=int,
        metavar="ID",
        help="Restrict to the knowledge bases of this team",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        metavar="N",
        help="Collections switched at once (default 1)",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        metavar="N",
        help=(
            "Maximum time to wait for in-flight writers before refusing "
            f"(default {DEFAULT_WAIT_SECONDS})"
        ),
    )


def run_cli(mode: str, argv: Optional[List[str]], *, description: str) -> int:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_scope_args(parser)
    args = parser.parse_args(argv)
    if args.concurrency <= 0:
        logger.error("--concurrency must be positive")
        return 2

    try:
        reports = switch_collections(
            args.model,
            collections=args.collection,
            team=args.team,
            concurrency=args.concurrency,
            mode=mode,
            wait_seconds=args.wait_seconds,
        )
    except MigrationCliError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        logger.error("Migration command failed: %s", exc, exc_info=True)
        return 2

    if not reports:
        return 1
    for report in reports:
        logger.info(
            "%s: %s %s%s in %.2fs (source %s, target %s, embedded %d documents/"
            "%d chunks, pruned %d rows)",
            report.name,
            report.mode,
            report.result,
            f" ({report.reason})" if report.reason else "",
            report.window_seconds,
            report.source_model_id,
            report.target_model_id,
            report.documents_embedded,
            report.chunks_embedded,
            report.pruned_rows,
        )
        for key, value in sorted(report.acceptance.items()):
            logger.info("%s: acceptance %s=%s", report.name, key, value)
    return (
        0 if all(r.result in (RESULT_SWITCHED, RESULT_ALREADY) for r in reports) else 1
    )


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(
        MODE_SWITCH,
        argv,
        description=(
            "Switch collections to a new embedding model.\n\n"
            "Precompute their vectors first (migrations.embedding.embed). A\n"
            "collection is frozen for the duration of its window; a re-run\n"
            "resumes a window that was interrupted."
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
