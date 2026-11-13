"""Switch one collection to a new embedding model inside a freeze window.

The window is: freeze the collection, wait for the writers the freeze let in
(running ingest jobs and live write leases), copy what the target model still
lacks (chunks ingested since the precompute), run the per-document acceptance
check, wait once more for writers, and flip the binding. A refusal restores the
row that was there before; a crash leaves the collection frozen, and a rerun
resumes the window only when its owner is this run or a stale one.

    python -m xagent.migrations.embedding.switch --model <model-id> \
        --collection <name> [--wait-seconds N]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    IngestionConfig,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.core.tools.core.RAG_tools.management.embedding_switch import (
    switch_collection_embedding_model,
)
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    IDLE,
    CollectionMigrationState,
    claim_freeze,
    heartbeat_freeze,
    read_migration_state,
    read_raw_state,
    restore_state,
    write_migration_state,
)
from xagent.core.tools.core.RAG_tools.management.write_leases import (
    live_leases,
)
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from xagent.core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string
from xagent.migrations.embedding.acceptance import AcceptanceResult, check_collection
from xagent.migrations.embedding.embed import (
    PacedEmbedding,
    RateLimiter,
    embed_collection,
    resolve_target_model,
)
from xagent.migrations.embedding.scope import (
    METADATA_TABLE,
    SKIP_OTHER_ENGINE,
    MigrationCliError,
    collection_engine,
    collection_filter,
    collection_info,
    connection,
    embeddings_table,
    iter_rows,
    source_embeddings_table,
    source_model_id,
    table_names,
)

logger = logging.getLogger(__name__)

DEFAULT_WAIT_SECONDS = 15 * 60
DEFAULT_POLL_SECONDS = 2.0
# Writers admitted in the race before the freeze can still land; re-check right
# before the flip so they delay the switch instead of racing it.
FINAL_LEASE_WAIT_SECONDS = 30.0
# A window renews its row every few seconds; a row that has not moved for this
# long has no live owner and may be resumed. The hour figure is only an alarm.
HEARTBEAT_SECONDS = 15.0
STALE_FREEZE_SECONDS = 5 * HEARTBEAT_SECONDS
STALE_ALARM_SECONDS = 60 * 60

RESULT_SWITCHED = "switched"
RESULT_ALREADY = "already-switched"
RESULT_REFUSED = "refused"
RESULT_FAILED = "failed"
RESULT_INTERRUPTED = "interrupted-restored"

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
    documents_copied: int = 0
    rows_copied: int = 0
    rows_pruned: int = 0
    excluded_rows: Dict[str, int] = field(default_factory=dict)
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

    def running(self, collection: str) -> List[str]:
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
                str(row.id)
                for row in rows
                if str(dict(row.payload or {}).get("collection") or "") == collection
            ]
        finally:
            db.close()


def _run_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def switch_collection(
    name: str,
    target_model_id: str,
    *,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    job_watcher: Optional[InflightJobWatcher] = None,
    acceptance: Optional[Callable[..., AcceptanceResult]] = None,
    mode: str = MODE_SWITCH,
    cfg: Optional[IngestionConfig] = None,
    rate_limit: float = 0.0,
    rate_limiter: Optional[RateLimiter] = None,
    owner: Optional[str] = None,
) -> SwitchReport:
    """Run the switch window for one collection.

    A ``KeyboardInterrupt`` inside the window first restores the row (before the
    flip) or records the switch (after it), then re-raises as
    :class:`WindowInterrupted` carrying the report.
    """
    started = time.monotonic()
    owner = owner or _run_owner()
    acceptance = acceptance or check_collection
    report = SwitchReport(name=name, target_model_id=target_model_id, mode=mode)

    def finish(result: str, reason: str = "") -> SwitchReport:
        report.result = result
        report.reason = reason
        report.window_seconds = round(time.monotonic() - started, 3)
        return report

    info = _info(name)
    if info is None:
        return finish(RESULT_REFUSED, "unknown-collection")
    if collection_engine(info) != "lancedb":
        return finish(RESULT_REFUSED, SKIP_OTHER_ENGINE)
    source = source_model_id(name, info)
    if source is None:
        return finish(RESULT_REFUSED, "unbound")
    report.source_model_id = source

    try:
        embedding_config, adapter = resolve_target_model(target_model_id)
    except MigrationCliError as exc:
        return finish(RESULT_REFUSED, str(exc))
    except Exception as exc:  # noqa: BLE001 - provider/hub failure, not a window
        return finish(RESULT_REFUSED, f"target-not-configured: {exc}")
    target = embedding_config.id.strip()

    previous_row = read_raw_state(name)
    previous = read_migration_state(name)
    if previous.state == FROZEN:
        refusal = _frozen_refusal(previous, target, mode)
        if refusal:
            return finish(RESULT_REFUSED, refusal)
    if mode == MODE_ROLLBACK:
        refusal = _rollback_refusal(previous, source, target)
        if refusal:
            return finish(RESULT_REFUSED, refusal)

    if source == target:
        if previous.state != FROZEN:
            return finish(RESULT_ALREADY)
        # A window that flipped and then died before recording it.
        if not _may_resume(previous, owner):
            return finish(RESULT_REFUSED, f"window-in-progress: {previous.owner}")
        _record_switch(name, previous.source_model_id, target, previous.mode or mode)
        return finish(RESULT_SWITCHED, "recovered-record")

    claim = claim_freeze(
        name,
        source_model_id=source,
        target_model_id=target,
        mode=mode,
        owner=owner,
        stale_after_seconds=STALE_FREEZE_SECONDS,
    )
    if claim == "busy":
        return finish(RESULT_REFUSED, f"window-in-progress: {previous.owner}")
    if claim == "resumed":
        logger.warning(
            "%s: resuming a frozen window (owner %s, last heartbeat %s)",
            name,
            previous.owner or "unknown",
            previous.updated_at,
        )
    resumed = claim == "resumed"

    heartbeat = _Heartbeat(name, owner)
    try:
        with heartbeat:
            _run_window(
                report,
                name,
                source=source,
                target=target,
                embedding_config=embedding_config,
                adapter=PacedEmbedding(
                    adapter, rate_limiter or RateLimiter(rate_limit)
                ),
                cfg=cfg or IngestionConfig(embedding_model_id=target),
                wait_seconds=wait_seconds,
                poll_seconds=poll_seconds,
                watcher=job_watcher or InflightJobWatcher(),
                acceptance=acceptance,
                heartbeat=heartbeat,
            )
    except WindowRefused as exc:
        _restore(name, owner, previous_row, source, resumed=resumed)
        return finish(RESULT_REFUSED, exc.reason)
    except KeyboardInterrupt:
        if _bound_to(name) == target:
            _record_switch(name, source, target, mode, report=report)
            finish(RESULT_SWITCHED, report.reason or "interrupted-after-flip")
        else:
            _restore(name, owner, previous_row, source, resumed=resumed)
            finish(RESULT_INTERRUPTED)
        raise WindowInterrupted(report) from None
    except Exception as exc:  # noqa: BLE001 - keep the collection usable
        if _bound_to(name) == target:
            _record_switch(name, source, target, mode, report=report)
            return finish(RESULT_SWITCHED, report.reason)
        _restore(name, owner, previous_row, source, resumed=resumed)
        logger.exception("%s: switch failed", name)
        return finish(RESULT_FAILED, f"{type(exc).__name__}: {exc}")

    _record_switch(name, source, target, mode, report=report)
    return finish(RESULT_SWITCHED, report.reason)


class WindowInterrupted(KeyboardInterrupt):
    """Ctrl-C inside a window, after its row was restored or its switch recorded."""

    def __init__(self, report: SwitchReport) -> None:
        super().__init__(report.name)
        self.report = report


class _Heartbeat:
    """Renews the frozen row while the window runs; notices a lost ownership."""

    def __init__(self, name: str, owner: str) -> None:
        self.name = name
        self.owner = owner
        self.lost = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"kb-switch-heartbeat-{name}", daemon=True
        )

    def __enter__(self) -> "_Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=HEARTBEAT_SECONDS)

    def beat(self) -> bool:
        try:
            if not heartbeat_freeze(self.name, self.owner):
                self.lost = True
        except Exception:  # noqa: BLE001 - an unrenewable row is a lost one
            logger.exception("%s: heartbeat failed", self.name)
            self.lost = True
        return not self.lost

    def _run(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            if not self.beat():
                return


def _info(name: str) -> Any:
    try:
        return collection_info(name)
    except MigrationCliError:
        return None


def _bound_to(name: str) -> Optional[str]:
    try:
        info = _info(name)
        return None if info is None else source_model_id(name, info)
    except Exception:  # noqa: BLE001 - unknown binding: treat as not flipped
        logger.exception("%s: could not read the binding", name)
        return None


def _may_resume(state: CollectionMigrationState, owner: str) -> bool:
    if state.owner == owner:
        return True
    return _age_seconds(state.updated_at) > STALE_FREEZE_SECONDS


def _frozen_refusal(
    state: CollectionMigrationState, target: str, mode: str
) -> Optional[str]:
    """A frozen row may only be resumed by a window of the same target and mode."""
    frozen_target = (state.target_model_id or "").strip()
    if frozen_target and frozen_target != target:
        return f"frozen-for-another-target: {frozen_target}"
    if (state.mode or MODE_SWITCH) != mode:
        return f"frozen-for-{state.mode or MODE_SWITCH}"
    return None


def _rollback_refusal(
    previous: CollectionMigrationState, source: str, target: str
) -> Optional[str]:
    """Rollback only undoes a recorded switch of this collection back to ``target``.

    A frozen rollback window carries the same shape: source = the switched-to
    model, target = the model it switched from.
    """
    if previous.state == FROZEN:
        if previous.mode == MODE_ROLLBACK and previous.target_model_id == target:
            return None
        return "no-recorded-switch"
    if previous.mode != MODE_SWITCH or previous.target_model_id != source:
        return "no-recorded-switch"
    if previous.source_model_id != target:
        return f"recorded-source-is-{previous.source_model_id}"
    return None


def _run_window(
    report: SwitchReport,
    name: str,
    *,
    source: str,
    target: str,
    embedding_config: EmbeddingModelConfig,
    adapter: BaseEmbedding,
    cfg: IngestionConfig,
    wait_seconds: float,
    poll_seconds: float,
    watcher: InflightJobWatcher,
    acceptance: Callable[..., AcceptanceResult],
    heartbeat: _Heartbeat,
) -> None:
    exists, identity = collection_identity(name)
    if identity is None:
        logger.warning(
            "%s: no stored created_at; a rename during the window is only "
            "detected by the name disappearing",
            name,
        )
    stragglers = _wait_until_idle(
        name, timeout=wait_seconds, poll=poll_seconds, watcher=watcher
    )
    if stragglers:
        raise WindowRefused("in-flight-writers: " + ", ".join(stragglers[:10]))

    # A rename admitted before the freeze moves the row; compare a stored
    # identity, not the name resolving at all.
    still, current = collection_identity(name)
    if not exists or not still or current != identity:
        raise WindowRefused("renamed")

    conn = connection()
    item = embed_collection(
        name,
        model_id=target,
        embedding_config=embedding_config,
        embedding_adapter=adapter,
        cfg=cfg,
        dry_run=False,
        conn=conn,
        source=source,
    )
    report.documents_copied = item.documents_embedded
    report.rows_copied = item.chunks_embedded
    report.excluded_rows = dict(item.skipped)
    if item.collection_skip or item.aborted or item.failures:
        raise WindowRefused(
            "embed-failed: "
            + (item.collection_skip or f"{len(item.failures)} documents failed")
        )

    report.rows_pruned = prune_target_only(name, source, target, conn=conn)

    problem = target_index_problem(target, conn=conn)
    if problem:
        raise WindowRefused(f"target-index-missing: {problem}")

    verdict = acceptance(
        name, source_model_id=source, target_model_id=target, conn=conn
    )
    report.acceptance = dict(verdict.checks)
    if not verdict.ok:
        raise WindowRefused("acceptance-failed: " + "; ".join(verdict.failures))

    stragglers = _wait_until_idle(
        name, timeout=FINAL_LEASE_WAIT_SECONDS, poll=poll_seconds, watcher=watcher
    )
    if stragglers:
        raise WindowRefused("writers-restarted: " + ", ".join(stragglers[:10]))

    # Stop renewing before the flip and check we still own the row: a window
    # whose heartbeat lapsed may already have been resumed by another run.
    heartbeat.stop()
    if heartbeat.lost or not heartbeat.beat():
        raise WindowRefused("ownership-lost")
    switch_collection_embedding_model(name, target)


def collection_identity(name: str) -> Tuple[bool, Optional[Any]]:
    """Whether the metadata row exists, and its stored ``created_at``.

    The stored value, not ``collection_info``: a legacy row without the column
    reads back as ``now()`` there, which would look like a rename every time.
    """
    conn = connection()
    if METADATA_TABLE not in table_names(conn):
        return False, None
    rows = list(
        iter_rows(
            conn,
            METADATA_TABLE,
            f"name = '{escape_lancedb_string(name)}'",
            ["name", "created_at"],
        )
    )
    if not rows:
        return False, None
    return True, rows[0].get("created_at")


def prune_target_only(
    name: str, source: str, target: str, *, conn: Optional[Any] = None
) -> int:
    """Delete target rows whose ``(doc_id, chunk_id)`` the source no longer has.

    Both sides are read in full with the strict reader before anything is
    deleted; a failed read raises and deletes nothing.
    """
    conn = conn if conn is not None else connection()
    source_table = source_embeddings_table(source, conn)
    target_table = embeddings_table(target, conn)
    if source_table is None or target_table is None:
        return 0
    where = collection_filter(name)
    keep = {
        (str(row.get("doc_id") or ""), str(row.get("chunk_id") or ""))
        for row in iter_rows(conn, source_table, where, ["doc_id", "chunk_id"])
    }
    extra: Dict[str, List[str]] = {}
    for row in iter_rows(conn, target_table, where, ["doc_id", "chunk_id"]):
        key = (str(row.get("doc_id") or ""), str(row.get("chunk_id") or ""))
        if key not in keep and key[0] and key[1]:
            extra.setdefault(key[0], []).append(key[1])
    if not extra:
        return 0
    store = get_vector_index_store()
    deleted = 0
    for doc_id, chunk_ids in sorted(extra.items()):
        deleted += store.delete_embedding_records(
            name, doc_id, chunk_ids=chunk_ids, model_tag=target, is_admin=True
        )
    logger.warning("%s: pruned %d target rows the source no longer has", name, deleted)
    return deleted


def target_index_problem(target: str, *, conn: Optional[Any] = None) -> Optional[str]:
    """Why search would degrade on the target table after the flip, if it would."""
    conn = conn if conn is not None else connection()
    table_name = embeddings_table(target, conn)
    if table_name is None:
        return None
    table = conn.open_table(table_name)
    try:
        indexes = table.list_indices()
        rows = table.count_rows()
    finally:
        _safe_close_table(table)
    if rows and not any(
        idx.index_type == "FTS" and "text" in idx.columns for idx in indexes
    ):
        return "no full-text index on text"
    if rows >= IndexPolicy().enable_threshold_rows and not any(
        "vector" in idx.columns for idx in indexes
    ):
        return "no vector index"
    return None


def _wait_until_idle(
    name: str, *, timeout: float, poll: float, watcher: InflightJobWatcher
) -> List[str]:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        jobs = watcher.running(name)
        leases = live_leases(name)
        if not jobs and not leases:
            return []
        if time.monotonic() >= deadline:
            return [f"job:{job}" for job in jobs] + [
                f"lease:{lease.operation}@{lease.owner}" for lease in leases
            ]
        time.sleep(poll)


def _age_seconds(stamp: Any) -> float:
    if not isinstance(stamp, datetime):
        return 0.0
    current = datetime.now(timezone.utc)
    value = stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    return max(0.0, (current - value).total_seconds())


def _record_switch(
    name: str,
    source: Optional[str],
    target: str,
    mode: str,
    *,
    report: Optional[SwitchReport] = None,
) -> None:
    """The completed-switch record the rollback window checks; never masks the flip."""
    try:
        write_migration_state(
            name,
            CollectionMigrationState(
                state=IDLE, source_model_id=source, target_model_id=target, mode=mode
            ),
        )
        logger.info("%s: switched %s -> %s (%s)", name, source, target, mode)
    except Exception as exc:  # noqa: BLE001 - report, never mask the switch
        logger.exception("%s: switched but could not record it", name)
        if report is not None:
            report.reason = f"left-frozen: {type(exc).__name__}: {exc}"


def _restore(
    name: str,
    owner: str,
    previous_row: Optional[Dict[str, Any]],
    source: str,
    *,
    resumed: bool,
) -> None:
    """Put back the row the window replaced, only while this run still owns it."""
    try:
        if read_migration_state(name).owner != owner:
            logger.error("%s: no longer the window's owner; row left as is", name)
            return
        if not resumed:
            restore_state(name, previous_row)
            return
        # A resumed window never saw the pre-freeze row. Unfreeze without a
        # target so it never looks like a completed switch.
        write_migration_state(
            name, CollectionMigrationState(state=IDLE, source_model_id=source)
        )
    except Exception:  # noqa: BLE001 - the freeze write is what matters
        logger.exception("%s: could not unfreeze; the collection stays frozen", name)


def describe(report: SwitchReport) -> None:
    """One line per collection, then excluded rows by reason and acceptance checks."""
    logger.info(
        "%s: %s%s in %.2fs (copied %d rows in %d documents, pruned %d rows)",
        report.name,
        report.result,
        f" ({report.reason})" if report.reason else "",
        report.window_seconds,
        report.rows_copied,
        report.documents_copied,
        report.rows_pruned,
    )
    excluded = {k: v for k, v in sorted(report.excluded_rows.items()) if v}
    if excluded:
        # Not a refusal: these rows keep only their old-model vectors.
        logger.warning(
            "%s: switched with excluded rows (%s); they stay on the old model only",
            report.name,
            ", ".join(f"{k}={v}" for k, v in excluded.items()),
        )
    for key, value in sorted(report.acceptance.items()):
        logger.info("%s: acceptance %s=%s", report.name, key, value)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Target embedding model id")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        metavar="N",
        help=f"Maximum wait for in-flight writers (default {DEFAULT_WAIT_SECONDS})",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.0,
        metavar="N",
        help="Average maximum of texts embedded per second; 0 = unlimited",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        metavar="N",
        help=f"Chunks per embedding request (default {DEFAULT_EMBEDDING_BATCH_SIZE})",
    )
    parser.add_argument(
        "--use-async",
        action="store_true",
        help="Encode chunks concurrently; needed by models without batch support",
    )


def _run_config(args: argparse.Namespace) -> IngestionConfig:
    return IngestionConfig(
        embedding_model_id=args.model,
        embedding_batch_size=args.batch_size,
        embedding_use_async=args.use_async,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Switch one collection to a new embedding model.\n\n"
            "Precompute its vectors first (migrations.embedding.embed). The\n"
            "collection is frozen for the duration of the window; a re-run\n"
            "resumes a window that was interrupted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(parser)
    parser.add_argument(
        "--collection", required=True, metavar="NAME", help="Collection to switch"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    if args.batch_size <= 0:
        logger.error("--batch-size must be positive")
        return 2
    if not math.isfinite(args.rate_limit) or args.rate_limit < 0:
        logger.error("--rate-limit must be a finite, non-negative number")
        return 2

    try:
        report = switch_collection(
            args.collection,
            args.model,
            wait_seconds=args.wait_seconds,
            cfg=_run_config(args),
            rate_limit=args.rate_limit,
        )
    except MigrationCliError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        logger.error("Switch failed: %s", exc, exc_info=True)
        return 2

    describe(report)
    return 0 if report.result in (RESULT_SWITCHED, RESULT_ALREADY) else 1


if __name__ == "__main__":
    sys.exit(main())
