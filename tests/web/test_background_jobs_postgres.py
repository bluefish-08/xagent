"""Postgres-only regression for the idle-in-transaction failure mode (#1535).

The SQLite suite pins the mechanism (bookkeeping runs on a session of its own).
Only a real server enforces ``idle_in_transaction_session_timeout``, which is
what actually killed the connection in production, so the end-to-end
consequence -- a job stuck at ``running`` with no error recorded -- can only be
reproduced here. Skipped unless XAGENT_TEST_POSTGRES_URL is set.
"""

from __future__ import annotations

import os
import time
import uuid

import psycopg2
import pytest

from tests.shared.postgres_disposable import (
    disposable_database_url,
    psycopg2_kwargs,
)

pytestmark = pytest.mark.postgresql

IDLE_TIMEOUT = "1s"


def _create_schema() -> None:
    import xagent.web.models.background_job  # noqa: F401
    import xagent.web.models.uploaded_file  # noqa: F401
    import xagent.web.models.user  # noqa: F401
    from xagent.web.models.database import Base, get_engine

    Base.metadata.create_all(get_engine())


def test_failure_is_recorded_after_idle_transaction_timeout(monkeypatch):
    """A job whose handler idles past the timeout must still reach FAILED.

    Before bookkeeping got its own sessions, the exception path recorded the
    failure on the same session the server had just terminated. The row stayed
    ``running`` with ``error_message`` NULL, which also meant max_attempts
    never applied.
    """
    from xagent.config import CELERY_BROKER_URL, CELERY_ENABLED
    from xagent.web.models import database as database_module

    monkeypatch.setenv(CELERY_ENABLED, "true")
    monkeypatch.setenv(CELERY_BROKER_URL, "memory://")

    with disposable_database_url(
        "xagent_jobs_idle_txn",
        settings={"idle_in_transaction_session_timeout": IDLE_TIMEOUT},
    ) as db_url:
        from xagent.web.jobs import tasks as tasks_module
        from xagent.web.jobs.celery_app import celery_app
        from xagent.web.models.background_job import BackgroundJob, BackgroundJobStatus
        from xagent.web.models.database import (
            configure_db,
            get_engine,
            get_session_local,
        )
        from xagent.web.models.user import User
        from xagent.web.services.background_jobs import create_background_job

        previous_engine = database_module._engine
        previous_session_local = database_module._SessionLocal
        previous_eager = celery_app.conf.task_always_eager
        previous_propagates = celery_app.conf.task_eager_propagates
        try:
            configure_db(db_url)
            _create_schema()

            # The regression only reproduces if the timeout the test asked for
            # is what the application's own connections actually get: a role or
            # session-level override would leave the sleep below harmless and
            # the assertions green for the wrong reason.
            with get_engine().connect() as probe:
                effective = probe.exec_driver_sql(
                    "SHOW idle_in_transaction_session_timeout"
                ).scalar()
            assert effective == IDLE_TIMEOUT, (
                "idle_in_transaction_session_timeout on the application "
                f"connection is {effective!r}, not {IDLE_TIMEOUT!r}; this test "
                "would pass without exercising the regression"
            )

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

            def handler(_db, _job):
                # Stand in for the gap between two progress updates during a
                # multi-hour crawl -- long enough for the server to reap an
                # idle transaction, which is what the worker session holds.
                time.sleep(2.5)
                raise RuntimeError("ingestion died mid-run")

            tasks_module.register_background_job_handler("kb.team.idle_txn", handler)
            try:
                tasks_module.execute_background_job.apply(args=[job_id])
            finally:
                tasks_module._EXTRA_HANDLERS.pop("kb.team.idle_txn", None)

            verify = SessionLocal()
            try:
                row = (
                    verify.query(BackgroundJob)
                    .filter(BackgroundJob.id == job_id)
                    .first()
                )
                assert row.status == BackgroundJobStatus.FAILED.value, (
                    f"job did not reach a terminal state: status={row.status!r} "
                    f"error={row.error_message!r}"
                )
                assert "ingestion died mid-run" in str(row.error_message)
                assert row.finished_at is not None
                assert int(row.attempts) == 1
            finally:
                verify.close()
        finally:
            celery_app.conf.task_always_eager = previous_eager
            celery_app.conf.task_eager_propagates = previous_propagates
            # configure_db rebinds process-global state; leaving it pointed at
            # a dropped database breaks whatever test runs next.
            if database_module._engine is not previous_engine:
                database_module._engine.dispose()
            database_module._engine = previous_engine
            database_module._SessionLocal = previous_session_local


def test_disposable_database_is_dropped_when_setup_fails():
    """A failing ALTER after CREATE DATABASE must not leak the database.

    The setup statements run between CREATE and the yield, so without a drop
    guard registered right after CREATE, a rejected setting leaves an orphan
    database on a shared server for every run.
    """
    base_url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not base_url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")

    # Unique per run: a bare prefix would also match databases minted by a
    # concurrent run against the same server.
    prefix = f"xagent_jobs_setup_fail_{uuid.uuid4().hex[:8]}"
    with pytest.raises(psycopg2.Error):
        with disposable_database_url(prefix, settings={"xagent_not_a_real_guc": "1"}):
            pytest.fail("the context manager must not yield after a failed setup")

    admin = psycopg2.connect(**psycopg2_kwargs(base_url))
    try:
        cursor = admin.cursor()
        cursor.execute(
            "SELECT datname FROM pg_database WHERE datname LIKE %s", (f"{prefix}_%",)
        )
        leaked = [row[0] for row in cursor.fetchall()]
    finally:
        admin.close()

    assert leaked == [], f"disposable databases left behind: {leaked}"
