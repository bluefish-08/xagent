from __future__ import annotations

import logging
from collections.abc import Collection
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, NamedTuple, cast
from urllib.parse import urlsplit

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session
from sqlalchemy.sql import Update

if TYPE_CHECKING:
    from sqlalchemy import CursorResult

from ...config import (
    get_background_job_max_retries,
    get_background_job_stale_seconds,
    get_celery_broker_url,
    get_celery_enabled,
)
from ..models.background_job import (
    BackgroundJob,
    BackgroundJobStatus,
    BackgroundJobType,
)
from ..models.database import session_scope

logger = logging.getLogger(__name__)

QUEUE_DEFAULT = "default"
QUEUE_KB = "kb"
QUEUE_TRIGGERS = "triggers"

NON_TERMINAL_JOB_STATUSES = frozenset(
    {
        BackgroundJobStatus.PENDING.value,
        BackgroundJobStatus.ENQUEUED.value,
        BackgroundJobStatus.RUNNING.value,
    }
)
TERMINAL_JOB_STATUSES = frozenset(
    {
        BackgroundJobStatus.SUCCEEDED.value,
        BackgroundJobStatus.FAILED.value,
        BackgroundJobStatus.CANCELLED.value,
    }
)


def _is_redis_broker_reachable(broker_url: str) -> bool:
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("Redis Celery broker configured but redis package is missing")
        return False

    try:
        client = redis.Redis.from_url(
            broker_url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        client.ping()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Celery Redis broker is unreachable: %s", exc)
        return False
    return True


def is_background_job_enqueue_available(*, check_worker: bool = False) -> bool:
    """Return whether a new durable job can be sent to Celery now."""
    if not get_celery_enabled():
        return False

    broker_url = get_celery_broker_url()
    if broker_url is None:
        return False

    broker_scheme = urlsplit(broker_url).scheme
    if broker_scheme in {"redis", "rediss"} and not _is_redis_broker_reachable(
        broker_url
    ):
        return False

    if not check_worker:
        return True

    try:
        from ..jobs.celery_app import celery_app

        if celery_app.conf.task_always_eager:
            return True
        return bool(celery_app.control.ping(timeout=0.5))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Celery worker health check failed: %s", exc)
        return False


def queue_for_job_type(job_type: str) -> str:
    if job_type.startswith("kb."):
        return QUEUE_KB
    if job_type.startswith("trigger."):
        return QUEUE_TRIGGERS
    return QUEUE_DEFAULT


def create_background_job(
    db: Session,
    *,
    user_id: int,
    job_type: str | BackgroundJobType,
    payload: dict[str, Any],
    queue: str | None = None,
    idempotency_key: str | None = None,
    max_attempts: int | None = None,
    reuse_terminal_idempotency_key: bool = True,
) -> BackgroundJob:
    resolved_job_type = (
        job_type.value if isinstance(job_type, BackgroundJobType) else job_type
    )

    if idempotency_key:
        existing_query = db.query(BackgroundJob).filter(
            BackgroundJob.idempotency_key == idempotency_key
        )
        if not reuse_terminal_idempotency_key:
            existing_query = existing_query.filter(
                BackgroundJob.status.in_(NON_TERMINAL_JOB_STATUSES)
            )
        existing = existing_query.first()
        if existing is not None:
            return existing
        if not reuse_terminal_idempotency_key:
            release_terminal_background_job_idempotency_key(db, idempotency_key)

    job = BackgroundJob(
        user_id=user_id,
        job_type=resolved_job_type,
        queue=queue or queue_for_job_type(resolved_job_type),
        status=BackgroundJobStatus.PENDING.value,
        payload=payload,
        progress={"message": "Queued", "completed": 0, "total": 1},
        idempotency_key=idempotency_key,
        max_attempts=max_attempts or get_background_job_max_retries(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_non_terminal_background_job_by_idempotency_key(
    db: Session,
    idempotency_key: str,
) -> BackgroundJob | None:
    return (
        db.query(BackgroundJob)
        .filter(BackgroundJob.idempotency_key == idempotency_key)
        .filter(BackgroundJob.status.in_(NON_TERMINAL_JOB_STATUSES))
        .first()
    )


def release_terminal_background_job_idempotency_key(
    db: Session,
    idempotency_key: str,
) -> None:
    terminal_jobs = (
        db.query(BackgroundJob)
        .filter(BackgroundJob.idempotency_key == idempotency_key)
        .filter(BackgroundJob.status.in_(TERMINAL_JOB_STATUSES))
        .all()
    )
    if not terminal_jobs:
        return
    for job in terminal_jobs:
        setattr(job, "idempotency_key", None)
        db.add(job)
    db.commit()


def enqueue_background_job(db: Session, job: BackgroundJob) -> BackgroundJob:
    if not get_celery_enabled():
        logger.info("Background job %s created but Celery enqueue is disabled", job.id)
        return job
    if get_celery_broker_url() is None:
        raise RuntimeError(
            "Celery background jobs are enabled but no broker URL is set"
        )

    from ..jobs.tasks import execute_background_job

    setattr(job, "status", BackgroundJobStatus.ENQUEUED.value)
    db.add(job)
    db.commit()
    db.refresh(job)

    async_result = execute_background_job.apply_async(
        args=[job.id],
        queue=str(job.queue or QUEUE_DEFAULT),
    )
    db.refresh(job)
    setattr(job, "celery_task_id", async_result.id)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_background_job(db: Session, job_id: str) -> BackgroundJob | None:
    return db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()


def list_background_jobs(
    db: Session,
    *,
    user_id: int,
    is_admin: bool,
    status: str | None = None,
    job_type: str | None = None,
    limit: int = 50,
) -> list[BackgroundJob]:
    query = db.query(BackgroundJob)
    if not is_admin:
        query = query.filter(BackgroundJob.user_id == user_id)
    if status:
        query = query.filter(BackgroundJob.status == status)
    if job_type:
        query = query.filter(BackgroundJob.job_type == job_type)
    return (
        query.order_by(BackgroundJob.created_at.desc())
        .limit(max(1, min(limit, 200)))
        .all()
    )


# mark_job_running / mark_job_failed each take a job id
# and open a session of their own: sharing the caller's is what let a poisoned
# connection take the failure record down with it. One contract for the
# family -- ValueError when the row is gone, never a silent no-op.


class JobClaim(NamedTuple):
    """Immutable snapshot of a won claim, enough to decide about a retry."""

    attempts: int
    max_attempts: int


class SettledJob(NamedTuple):
    """Immutable snapshot of a job the claim refused as already settled."""

    status: str
    result: dict[str, Any] | None


def mark_job_running(job_id: str) -> JobClaim | SettledJob:
    """Claim the job for this attempt, or report the state that refused it.

    The terminal check and the claim are one conditional UPDATE so no cancel
    can land between them and be clobbered back to RUNNING. Row locking is not
    involved, so the guarantee holds on SQLite too.
    """
    with session_scope() as db:
        claimed = _rowcount(
            db,
            update(BackgroundJob)
            .where(
                BackgroundJob.id == job_id,
                BackgroundJob.status.not_in(TERMINAL_JOB_STATUSES),
            )
            .values(
                status=BackgroundJobStatus.RUNNING.value,
                attempts=BackgroundJob.attempts + 1,
                started_at=datetime.now(timezone.utc),
                error_message=None,
                progress={"message": "Running", "completed": 0, "total": 1},
            )
            .execution_options(synchronize_session=False),
        )
        row = db.execute(
            select(
                BackgroundJob.status,
                BackgroundJob.attempts,
                BackgroundJob.max_attempts,
                BackgroundJob.result,
            ).where(BackgroundJob.id == job_id)
        ).first()
        if row is None:
            raise ValueError(f"Background job not found: {job_id}")
        if not claimed:
            return SettledJob(status=str(row.status), result=row.result)
        return JobClaim(
            attempts=int(row.attempts or 0), max_attempts=int(row.max_attempts or 1)
        )


def update_job_progress(
    db: Session,
    job: BackgroundJob,
    *,
    message: str,
    completed: int | None = None,
    total: int | None = None,
    extra: dict[str, Any] | None = None,
) -> BackgroundJob:
    progress = dict(job.progress or {})
    progress["message"] = message
    if completed is not None:
        progress["completed"] = completed
    if total is not None:
        progress["total"] = total
    if extra:
        progress.update(extra)
    setattr(job, "progress", progress)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _rowcount(db: Session, statement: Update) -> int:
    return cast("CursorResult[Any]", db.execute(statement)).rowcount


def _fenced_job_update(
    db: Session,
    job_id: str,
    *,
    expected: Collection[str],
    unchanged_since: datetime | None = None,
    **values: Any,
) -> bool:
    """Write only while the row is still the one this pass observed.

    A worker or a cancellation can settle the row between the requeue's steps;
    a zero rowcount means one did, and the row is left alone instead of being
    resurrected. ``unchanged_since`` additionally pins the ``updated_at`` the
    scan read, so a live worker that reports progress after the SELECT takes
    the row out of this sweep. Two sweepers reading the same snapshot still
    both match -- that needs a claim token, not a state predicate.
    """
    predicates = [
        BackgroundJob.id == job_id,
        BackgroundJob.status.in_(tuple(expected)),
    ]
    if unchanged_since is not None:
        predicates.append(BackgroundJob.updated_at == unchanged_since)
    return bool(
        _rowcount(
            db,
            update(BackgroundJob)
            .where(*predicates)
            .values(**values)
            .execution_options(synchronize_session=False),
        )
    )


class RequeuedJob(NamedTuple):
    """Immutable snapshot of one requeue claim, not of the row's later state."""

    id: str
    queue: str


def requeue_stale_background_jobs(
    db: Session,
    *,
    stale_after_seconds: int | None = None,
    limit: int = 100,
) -> list[RequeuedJob]:
    """Requeue non-terminal jobs whose durable DB state is stale.

    Redis/Celery can lose in-flight delivery state during broker loss or worker
    crashes. The database row remains authoritative, so the scheduler can safely
    put old pending/enqueued/running jobs back on the broker.

    Staleness is judged on ``updated_at`` for every status, never ``started_at``:
    the latter is written once and never advances, so a RUNNING job judged by it
    is requeued for being long, not for being dead.
    """
    stale_seconds = stale_after_seconds or get_background_job_stale_seconds()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
    requeue_statuses = {
        BackgroundJobStatus.PENDING.value,
        BackgroundJobStatus.ENQUEUED.value,
        BackgroundJobStatus.RUNNING.value,
    }

    stale_jobs = (
        db.query(BackgroundJob)
        .filter(BackgroundJob.status.in_(requeue_statuses))
        .filter(
            or_(
                and_(
                    BackgroundJob.updated_at.is_not(None),
                    BackgroundJob.updated_at <= cutoff,
                ),
                and_(
                    BackgroundJob.updated_at.is_(None),
                    BackgroundJob.created_at.is_not(None),
                    BackgroundJob.created_at <= cutoff,
                ),
            )
        )
        .order_by(BackgroundJob.created_at.asc())
        .limit(max(1, min(limit, 500)))
        .all()
    )

    if not stale_jobs:
        return []

    # Snapshot dispatch identities and the observed state while the attributes
    # are still loaded -- after the commit each read would lazy-load on a
    # reopened transaction.
    targets = [
        (
            str(job.id),
            str(job.queue or QUEUE_DEFAULT),
            str(job.job_type),
            str(job.status),
            cast("datetime | None", job.updated_at),
        )
        for job in stale_jobs
    ]

    claimed: list[RequeuedJob] = []
    for job_id, queue, job_type, observed_status, observed_updated_at in targets:
        logger.warning(
            "Requeueing stale background job %s type=%s status=%s",
            job_id,
            job_type,
            observed_status,
        )
        if _fenced_job_update(
            db,
            job_id,
            expected=(observed_status,),
            unchanged_since=observed_updated_at,
            status=BackgroundJobStatus.PENDING.value,
            celery_task_id=None,
            started_at=None,
            error_message="Requeued stale background job",
            progress={
                "message": "Requeued stale background job",
                "completed": 0,
                "total": 1,
            },
        ):
            claimed.append(RequeuedJob(id=job_id, queue=queue))

    db.commit()

    if not claimed or not get_celery_enabled():
        return claimed

    if get_celery_broker_url() is None:
        error_message = "Celery background jobs are enabled but no broker URL is set"
        for job_id, _queue in claimed:
            _fenced_job_update(
                db,
                job_id,
                expected=(BackgroundJobStatus.PENDING.value,),
                error_message=f"Failed to requeue stale job: {error_message}",
            )
        db.commit()
        return claimed

    from ..jobs.tasks import execute_background_job

    dispatch_targets = [
        (job_id, queue)
        for job_id, queue in claimed
        if _fenced_job_update(
            db,
            job_id,
            expected=(BackgroundJobStatus.PENDING.value,),
            status=BackgroundJobStatus.ENQUEUED.value,
        )
    ]
    db.commit()

    task_ids: dict[str, str] = {}
    dispatch_errors: dict[str, str] = {}
    for job_id, queue in dispatch_targets:
        try:
            async_result = execute_background_job.apply_async(
                args=[job_id],
                queue=queue,
            )
            task_ids[job_id] = async_result.id
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to requeue stale background job %s", job_id)
            dispatch_errors[job_id] = str(exc)

    for job_id, _queue in dispatch_targets:
        error = dispatch_errors.get(job_id)
        if error is None:
            _fenced_job_update(
                db,
                job_id,
                expected=(BackgroundJobStatus.ENQUEUED.value,),
                celery_task_id=task_ids.get(job_id),
            )
        else:
            _fenced_job_update(
                db,
                job_id,
                expected=(BackgroundJobStatus.ENQUEUED.value,),
                status=BackgroundJobStatus.PENDING.value,
                error_message=f"Failed to requeue stale job: {error}",
            )

    db.commit()

    return claimed


def mark_job_failed(
    job_id: str,
    *,
    error_message: str,
    result: dict[str, Any] | None = None,
) -> None:
    """Record the terminal failure on a session of its own.

    This must not share a session with the work that failed: a poisoned
    session takes the failure record down with it, leaving the row ``running``
    forever and bypassing ``max_attempts``.
    """
    with session_scope() as db:
        job = get_background_job(db, job_id)
        if job is None:
            raise ValueError(f"Background job not found: {job_id}")
        setattr(job, "status", BackgroundJobStatus.FAILED.value)
        setattr(job, "error_message", error_message)
        setattr(job, "result", result)
        setattr(job, "finished_at", datetime.now(timezone.utc))
        setattr(job, "progress", {"message": error_message, "completed": 0, "total": 1})
        db.add(job)
