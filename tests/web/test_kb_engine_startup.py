"""The backend, the agent worker and the Celery worker refuse a mismatched KB engine."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from cryptography.fernet import Fernet

from xagent.core.tools.core.RAG_tools.core.exceptions import ConfigurationError
from xagent.core.tools.core.RAG_tools.storage.vector_backend import KB_ENGINE_RECORD


@pytest.fixture(autouse=True)
def milvus_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path))
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "lancedb")
    (tmp_path / KB_ENGINE_RECORD).write_text("milvus\n")


@pytest.mark.asyncio
async def test_backend_startup_refuses_before_touching_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web import app

    initialize = AsyncMock()
    monkeypatch.setattr(app, "_initialize_database_and_admit_runtime", initialize)

    with pytest.raises(ConfigurationError, match="KB engine is milvus"):
        await app.startup_event()
    initialize.assert_not_called()


@pytest.mark.asyncio
async def test_agent_worker_refuses_before_accepting_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web import worker

    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "worker")
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    for name in (
        "configure_db",
        "validate_worker_schema",
        "validate_interaction_rollout_at_startup",
        "register_local_browser_runtime",
    ):
        monkeypatch.setattr(worker, name, Mock())

    with pytest.raises(ConfigurationError, match="KB engine is milvus"):
        await worker.run_worker(stop=asyncio.Event())
    worker.register_local_browser_runtime.assert_not_called()


def test_celery_worker_exits_at_worker_init() -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    with pytest.raises(
        SystemExit,
        match="Refusing to start the Celery worker: This deployment's KB engine is milvus",
    ):
        worker_init.send(sender=None)


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "message"),
    [
        (3, "disk unavailable\n", "", "disk unavailable"),
        (
            3,
            "KB engine detected\nThis deployment's KB engine is milvus\n",
            "Cannot record the KB engine: [Errno 30] Read-only file system\n",
            "This deployment's KB engine is milvus",
        ),
        (-11, "", "Fatal Python error\nSegmentation fault\n", "Segmentation fault"),
        (-11, "", "", "the engine check was killed by signal 11"),
        (1, "", "", "the engine check exited with code 1"),
    ],
)
def test_celery_worker_exits_on_any_engine_check_failure(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    stderr: str,
    message: str,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    monkeypatch.setattr(
        celery_app.subprocess,
        "run",
        lambda *_, **__: subprocess.CompletedProcess([], returncode, stdout, stderr),
    )

    with pytest.raises(
        SystemExit, match=f"Refusing to start the Celery worker: {message}$"
    ):
        worker_init.send(sender=None)


def test_celery_worker_exits_when_the_engine_check_cannot_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    monkeypatch.setattr(celery_app.sys, "executable", str(tmp_path / "no-python"))

    with pytest.raises(SystemExit, match="cannot run the engine check: .*no-python"):
        worker_init.send(sender=None)


def test_celery_worker_reads_an_engine_check_that_prints_undecodable_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    script = (
        "import sys\n"
        "sys.stdout.buffer.write(b'\\xff the path is broken\\n')\n"
        "raise SystemExit(3)\n"
    )
    monkeypatch.setattr(celery_app, "_LOCK_KB_ENGINE", script)

    with pytest.raises(
        SystemExit, match="Refusing to start the Celery worker: . the path is broken$"
    ):
        worker_init.send(sender=None)


def test_celery_worker_exits_when_the_engine_check_output_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    def undecodable(*_: object, **__: object) -> None:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(celery_app.subprocess, "run", undecodable)

    with pytest.raises(SystemExit, match="cannot run the engine check: .*0xff"):
        worker_init.send(sender=None)


def test_celery_parent_leaves_the_engine_check_to_a_child_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from celery.signals import worker_init

    from xagent.core.tools.core.RAG_tools.storage import vector_backend
    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    def in_the_parent() -> None:
        # Celery swallows an Exception raised by a signal handler, so exit instead.
        raise SystemExit("the engine check opened LanceDB in the Celery parent")

    monkeypatch.setattr(vector_backend, "lock_deployment_kb_engine", in_the_parent)
    (tmp_path / KB_ENGINE_RECORD).write_text("lancedb\n")

    worker_init.send(sender=None)
