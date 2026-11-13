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
import os
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    IngestionConfig,
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
)
from xagent.migrations.embedding.acceptance import AcceptanceResult, check_collection
from xagent.migrations.embedding.embed import (
    PacedEmbedding,
    RateLimiter,
    embed_collection,
    resolve_target_model,
)
from xagent.migrations.embedding.scope import (
    SKIP_OTHER_ENGINE,
    MigrationCliError,
    collection_engine,
    collection_info,
    connection,
    iter_source_chunks,
    skip_reason,
    source_model_id,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_WAIT_SECONDS = 15 * 60
DEFAULT_POLL_SECONDS = 2.0
# Writers admitted in the race before the freeze can still land; re-check right
# before the flip so they delay the switch instead of racing it.
FINAL_LEASE_WAIT_SECONDS = 30.0
# A window whose owner has not updated its row for this long is treated as
# abandoned and may be resumed.
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
    ownerless_excluded: int = 0
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
    owner: Optional[str] = None,
) -> SwitchReport:
    """Run the switch window for one collection."""
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

    previous = read_migration_state(name)
    resumed = previous.state == FROZEN
    if resumed:
        if not _may_resume(previous, owner):
            return finish(
                RESULT_REFUSED, f"window-in-progress: {previous.owner or 'unknown'}"
            )
        frozen_target = (previous.target_model_id or "").strip()
        if frozen_target and frozen_target != target:
            return finish(RESULT_REFUSED, f"frozen-for-another-target: {frozen_target}")
        logger.warning(
            "%s: resuming a frozen window (owner %s, since %s)",
            name,
            previous.owner or "unknown",
            previous.updated_at,
        )
    if mode == MODE_ROLLBACK:
        refusal = _rollback_refusal(previous, source, target)
        if refusal:
            return finish(RESULT_REFUSED, refusal)

    if source == target:
        if resumed:
            _set_state(name, IDLE, source, None, mode=None)
        return finish(RESULT_ALREADY)

    identity = getattr(info, "created_at", None)
    if not resumed:
        _set_state(name, FROZEN, source, target, mode=mode, owner=owner)
    try:
        _run_window(
            report,
            name,
            owner=owner,
            source=source,
            target=target,
            embedding_config=embedding_config,
            adapter=PacedEmbedding(adapter, RateLimiter(rate_limit)),
            cfg=cfg or IngestionConfig(embedding_model_id=target),
            wait_seconds=wait_seconds,
            poll_seconds=poll_seconds,
            watcher=job_watcher or InflightJobWatcher(),
            acceptance=acceptance,
            identity=identity,
        )
    except WindowRefused as exc:
        _restore(name, previous, source)
        return finish(RESULT_REFUSED, exc.reason)
    except Exception as exc:  # noqa: BLE001 - keep the collection usable
        _restore(name, previous, source)
        logger.exception("%s: switch failed", name)
        return finish(RESULT_FAILED, f"{type(exc).__name__}: {exc}")

    # The row records what this switch did, for the rollback window. Failing to
    # write it does not undo the flip, so the report says both things.
    try:
        _set_state(name, IDLE, source, target, mode=mode, owner=None)
    except Exception as exc:  # noqa: BLE001 - report, never mask the switch
        logger.exception("%s: switched but could not record it", name)
        return finish(RESULT_SWITCHED, f"left-frozen: {type(exc).__name__}: {exc}")
    return finish(RESULT_SWITCHED)


def _info(name: str) -> Any:
    try:
        return collection_info(name)
    except MigrationCliError:
        return None


def _may_resume(state: CollectionMigrationState, owner: str) -> bool:
    if state.owner in (None, owner):
        return True
    return _age_seconds(state.updated_at) > STALE_FREEZE_SECONDS


def _rollback_refusal(
    previous: CollectionMigrationState, source: str, target: str
) -> Optional[str]:
    """Rollback only undoes a recorded switch of this collection back to ``target``."""
    if previous.state == FROZEN:
        return None
    if previous.mode != MODE_SWITCH or previous.target_model_id != source:
        return "no-recorded-switch"
    if previous.source_model_id != target:
        return f"recorded-source-is-{previous.source_model_id}"
    return None


def _run_window(
    report: SwitchReport,
    name: str,
    *,
    owner: str,
    source: str,
    target: str,
    embedding_config: EmbeddingModelConfig,
    adapter: BaseEmbedding,
    cfg: IngestionConfig,
    wait_seconds: float,
    poll_seconds: float,
    watcher: InflightJobWatcher,
    acceptance: Callable[..., AcceptanceResult],
    identity: Any,
) -> None:
    stragglers = _wait_until_idle(
        name, timeout=wait_seconds, poll=poll_seconds, watcher=watcher
    )
    if stragglers:
        raise WindowRefused("in-flight-writers: " + ", ".join(stragglers[:10]))

    # A rename admitted before the freeze moves the row; compare a stable
    # identity, not the name resolving at all.
    current = _info(name)
    if current is None or getattr(current, "created_at", None) != identity:
        raise WindowRefused("renamed")

    conn = connection()
    # Ownerless, textless and unaddressable chunks are excluded from both sides
    # of acceptance: none of them answers a search today either.
    skipped = {
        chunk.doc_id
        for chunk in iter_source_chunks(name, source, conn=conn)
        if skip_reason(chunk) is not None
    }
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
    report.documents_embedded = item.documents_embedded
    report.chunks_embedded = item.chunks_embedded
    report.skipped = dict(item.skipped)
    report.ownerless_excluded = item.skipped.get("no-owner", 0)
    if item.collection_skip or item.aborted or item.failures:
        raise WindowRefused(
            "embed-failed: "
            + (item.collection_skip or f"{len(item.failures)} documents failed")
        )

    verdict = acceptance(
        name,
        source_model_id=source,
        target_model_id=target,
        skipped=skipped,
        conn=conn,
    )
    report.acceptance = dict(verdict.checks)
    if not verdict.ok:
        raise WindowRefused("acceptance-failed: " + "; ".join(verdict.failures))

    stragglers = _wait_until_idle(
        name, timeout=FINAL_LEASE_WAIT_SECONDS, poll=poll_seconds, watcher=watcher
    )
    if stragglers:
        raise WindowRefused("writers-restarted: " + ", ".join(stragglers[:10]))

    switch_collection_embedding_model(name, target)


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


def _set_state(
    name: str,
    state: str,
    source: Optional[str],
    target: Optional[str],
    *,
    mode: Optional[str],
    owner: Optional[str] = None,
) -> None:
    write_migration_state(
        name,
        CollectionMigrationState(
            state=state,
            source_model_id=source,
            target_model_id=target,
            mode=mode,
            owner=owner,
        ),
    )
    logger.info(
        "%s: %s (source %s, target %s, mode %s)", name, state, source, target, mode
    )


def _restore(name: str, previous: CollectionMigrationState, source: str) -> None:
    """Unfreeze without claiming a switch happened."""
    try:
        _set_state(
            name,
            IDLE,
            previous.source_model_id or source,
            None,
            mode=None,
            owner=None,
        )
    except Exception:  # noqa: BLE001 - the freeze write is what matters
        logger.exception("%s: could not unfreeze; the collection stays frozen", name)


def _describe(report: SwitchReport) -> None:
    logger.info(
        "%s: %s%s in %.2fs (copied %d documents/%d chunks)",
        report.name,
        report.result,
        f" ({report.reason})" if report.reason else "",
        report.window_seconds,
        report.documents_embedded,
        report.chunks_embedded,
    )
    if report.ownerless_excluded:
        logger.warning(
            "%s: %d ownerless chunks not migrated (admin-only legacy data)",
            report.name,
            report.ownerless_excluded,
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
    if args.batch_size <= 0:
        logger.error("--batch-size must be positive")
        return 2
    if args.rate_limit < 0:
        logger.error("--rate-limit must not be negative")
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

    _describe(report)
    return 0 if report.result in (RESULT_SWITCHED, RESULT_ALREADY) else 1


if __name__ == "__main__":
    sys.exit(main())
