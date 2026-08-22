from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from ..models.background_job import (
    BackgroundJob,
    BackgroundJobStatus,
    BackgroundJobType,
)
from ..models.database import (
    get_optional_session_local,
    get_session_local,
    init_db,
    session_scope,
)
from ..services.background_jobs import (
    get_background_job,
    mark_job_failed,
    mark_job_running,
    mark_job_succeeded,
)
from .celery_app import celery_app
from .exceptions import BackgroundJobHandlerError

logger = logging.getLogger(__name__)


class _Unset:
    """Sentinel type for "argument not supplied" (bare ``object`` is vacuous)."""


_UNSET = _Unset()

BackgroundJobHandler = Callable[[Session, BackgroundJob], dict[str, Any]]

# Downstream distributions own job types this package must not import. They
# register a handler when the worker imports their task module, so their work
# still flows through execute_background_job and keeps the shared retry and
# stale-requeue behaviour instead of needing a parallel Celery task.
_EXTRA_HANDLERS: dict[str, BackgroundJobHandler] = {}

# Job types this package routes itself in _execute_job_handler, which runs
# before _EXTRA_HANDLERS is consulted.
_BUILTIN_JOB_TYPES = frozenset(job_type.value for job_type in BackgroundJobType)


def register_background_job_handler(
    job_type: str, handler: BackgroundJobHandler, *, replace: bool = False
) -> None:
    """Register a handler for a job type defined outside this package.

    Args:
        job_type: Durable ``BackgroundJob.job_type`` value the handler owns.
        handler: Callable invoked by ``_execute_job_handler`` for that job type.
        replace: Allow replacing an existing registration for ``job_type``.

    Raises:
        ValueError: If ``job_type`` is built into this package, or is already
            registered and ``replace`` is not set.
    """
    key = str(job_type)
    if key in _BUILTIN_JOB_TYPES:
        # _execute_job_handler would never reach such a handler, so accepting
        # the registration would silently do nothing.
        raise ValueError(f"Background job type is built-in: {key}")
    if not replace and key in _EXTRA_HANDLERS:
        raise ValueError(f"Background job handler already registered: {key}")
    _EXTRA_HANDLERS[key] = handler


def is_background_job_handler_registered(job_type: str) -> bool:
    """Return whether an externally registered handler owns ``job_type``.

    Built-in job types are routed directly by ``_execute_job_handler`` and are
    never present in the external registry, so this reports ``False`` for them.
    """
    return str(job_type) in _EXTRA_HANDLERS


def _ensure_db_initialized() -> None:
    if get_optional_session_local() is None:
        init_db()


def _open_worker_session() -> Session:
    _ensure_db_initialized()
    return get_session_local()()


def _execute_job_handler(db: Session, job: BackgroundJob) -> dict[str, Any]:
    if job.job_type == BackgroundJobType.KB_INGEST_DOCUMENT.value:
        from .kb_tasks import handle_kb_ingest_document

        return handle_kb_ingest_document(db, job)
    if job.job_type == BackgroundJobType.KB_INGEST_WEB.value:
        from .kb_tasks import handle_kb_ingest_web

        return handle_kb_ingest_web(db, job)
    if job.job_type == BackgroundJobType.TRIGGER_EVENT.value:
        from .trigger_tasks import handle_trigger_event

        return handle_trigger_event(db, job)
    if job.job_type == BackgroundJobType.TRIGGER_SCAN.value:
        from .trigger_tasks import handle_trigger_scan

        return handle_trigger_scan(db, job)

    handler = _EXTRA_HANDLERS.get(str(job.job_type))
    if handler is not None:
        return handler(db, job)

    # A worker that cannot route this job type will never grow a handler by
    # waiting, so this is permanent: fail fast instead of burning retries.
    raise BackgroundJobHandlerError(
        f"Unsupported background job type: {job.job_type}",
        retryable=False,
    )


@celery_app.task(
    bind=True,
    name="xagent.web.jobs.tasks.execute_background_job",
    retry_backoff=True,
    retry_jitter=True,
)
def execute_background_job(self: Any, job_id: str) -> dict[str, Any]:
    _ensure_db_initialized()

    attempts = mark_job_running(job_id)
    if attempts is None:
        return _settled_job_result(job_id)

    db = _open_worker_session()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        if job is None:
            raise ValueError(f"Background job not found: {job_id}")
        max_attempts = int(job.max_attempts or 1)

        try:
            result = _execute_job_handler(db, job)
        except BackgroundJobHandlerError as exc:
            _end_worker_transaction(db)
            if exc.retryable and attempts < max_attempts:
                _mark_job_for_retry(job_id, error_message=str(exc), result=exc.result)
                raise self.retry(exc=exc, max_retries=max_attempts)
            mark_job_failed(job_id, error_message=str(exc), result=exc.result)
            raise
        except Exception as exc:  # noqa: BLE001
            _end_worker_transaction(db)
            if attempts < max_attempts:
                _mark_job_for_retry(job_id, error_message=str(exc))
                raise self.retry(exc=exc, max_retries=max_attempts)
            mark_job_failed(job_id, error_message=str(exc))
            raise

        _end_worker_transaction(db)
        mark_job_succeeded(job_id, result=result)
        return result
    finally:
        db.close()


def _settled_job_result(job_id: str) -> dict[str, Any]:
    """Report the outcome of a job the claim refused as already settled."""
    with session_scope() as db:
        job = get_background_job(db, job_id)
        if job is not None and job.status == BackgroundJobStatus.SUCCEEDED.value:
            logger.info("Skipping already completed background job %s", job_id)
            return dict(job.result or {"status": "succeeded"})
        if job is not None and job.status == BackgroundJobStatus.FAILED.value:
            logger.info("Skipping already failed background job %s", job_id)
            return dict(job.result or {"status": "failed"})
    logger.info("Skipping cancelled background job %s", job_id)
    return {"status": "cancelled"}


def _end_worker_transaction(db: Session) -> None:
    """End the handler session's transaction before bookkeeping opens its own.

    Bookkeeping now runs on a second connection; leaving this one idle in
    transaction across it is the exposure this whole path exists to remove,
    and a rollback that fails on an already-dead connection must not replace
    the failure being recorded. Handlers must commit their own writes: any
    ORM state still pending here is discarded, not committed.
    """
    try:
        db.rollback()
    except Exception:  # noqa: BLE001
        logger.debug("Failed to end worker session transaction", exc_info=True)


def _mark_job_for_retry(
    job_id: str,
    *,
    error_message: str,
    result: dict[str, Any] | None | _Unset = _UNSET,
) -> None:
    """Hand the job back to the broker for another attempt.

    ``result`` distinguishes "not supplied" from an explicit ``None``: the
    handler-error path overwrites the stored result even when it is None, the
    generic path leaves it untouched.
    """
    with session_scope() as db:
        job = get_background_job(db, job_id)
        if job is None:
            raise ValueError(f"Background job not found: {job_id}")
        setattr(job, "status", BackgroundJobStatus.ENQUEUED.value)
        setattr(job, "error_message", error_message)
        if not isinstance(result, _Unset):
            setattr(job, "result", result)
        db.add(job)
