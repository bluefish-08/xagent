"""Postgres-only regression for the idle-in-transaction failure mode (#1535).

The SQLite suite pins the mechanism (no transaction spans the handler body).
Only a real server enforces ``idle_in_transaction_session_timeout``, which is
what actually killed the connection in production, so the end-to-end
consequence -- a job stuck at ``running`` with no error recorded -- can only be
reproduced here. Skipped unless XAGENT_TEST_POSTGRES_URL is set.
"""

from __future__ import annotations

import time

import pytest

from tests.shared.postgres_disposable import disposable_database_url

pytestmark = pytest.mark.integration

IDLE_TIMEOUT = "1s"


def _create_schema() -> None:
    import xagent.web.models.background_job  # noqa: F401
    import xagent.web.models.uploaded_file  # noqa: F401
    import xagent.web.models.user  # noqa: F401
    from xagent.web.models.database import Base, get_engine

    Base.metadata.create_all(get_engine())


def test_failure_is_recorded_after_idle_transaction_timeout(monkeypatch):
    """A job whose handler idles past the timeout must still reach FAILED.

    Before the per-operation sessions, the worker held one session open across
    the handler body; the server terminated it, and the exception path then
    tried to record the failure on that dead session. The row stayed
    ``running`` with ``error_message`` NULL, which also meant max_attempts
    never applied.
    """
    from xagent.config import CELERY_BROKER_URL, CELERY_ENABLED

    monkeypatch.setenv(CELERY_ENABLED, "true")
    monkeypatch.setenv(CELERY_BROKER_URL, "memory://")

    with disposable_database_url(
        "xagent_jobs_idle_txn",
        settings={"idle_in_transaction_session_timeout": IDLE_TIMEOUT},
    ) as db_url:
        from xagent.web.jobs import tasks as tasks_module
        from xagent.web.jobs.celery_app import celery_app
        from xagent.web.models.background_job import BackgroundJob, BackgroundJobStatus
        from xagent.web.models.database import configure_db, get_session_local
        from xagent.web.models.user import User
        from xagent.web.services.background_jobs import create_background_job

        configure_db(db_url)
        _create_schema()

        celery_app.conf.task_always_eager = True
        celery_app.conf.task_eager_propagates = False

        SessionLocal = get_session_local()
        setup = SessionLocal()
        try:
            user = User(username="idle-txn", password_hash="x")
            setup.add(user)
            setup.commit()
            setup.refresh(user)
            job = create_background_job(
                setup,
                user_id=int(user.id),
                job_type="kb.team.idle_txn",
                payload={},
                max_attempts=1,
            )
            setup.commit()
            job_id = str(job.id)
        finally:
            setup.close()

        def handler(_ref):
            # Stand in for the gap between two progress updates during a
            # multi-hour crawl -- long enough for the server to reap an idle
            # transaction, if one were open.
            time.sleep(2.5)
            raise RuntimeError("ingestion died mid-run")

        tasks_module.register_background_job_handler("kb.team.idle_txn", handler)
        try:
            tasks_module.execute_background_job.apply(args=[job_id])
        finally:
            tasks_module._EXTRA_HANDLERS.pop("kb.team.idle_txn", None)
            celery_app.conf.task_always_eager = False
            celery_app.conf.task_eager_propagates = False

        verify = SessionLocal()
        try:
            row = verify.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
            assert row.status == BackgroundJobStatus.FAILED.value, (
                f"job did not reach a terminal state: status={row.status!r} "
                f"error={row.error_message!r}"
            )
            assert "ingestion died mid-run" in str(row.error_message)
            assert row.finished_at is not None
            assert int(row.attempts) == 1
        finally:
            verify.close()
