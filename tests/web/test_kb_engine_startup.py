"""The backend, the agent worker and the Celery worker check the KB engine at start."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import lancedb
import pytest
from cryptography.fernet import Fernet

from xagent.core.tools.core.RAG_tools.core.exceptions import ConfigurationError
from xagent.core.tools.core.RAG_tools.storage.vector_backend import KB_ENGINE_RECORD


class _Started(Exception):
    pass


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

    with pytest.raises(SystemExit, match="KB engine is milvus"):
        worker_init.send(sender=None)


def test_celery_worker_exits_on_any_engine_check_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celery.signals import worker_init

    from xagent.core.tools.core.RAG_tools.storage import vector_backend
    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    def unwritable() -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(vector_backend, "lock_deployment_kb_engine", unwritable)

    with pytest.raises(SystemExit, match="disk unavailable"):
        worker_init.send(sender=None)


@pytest.fixture
def add_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The Milvus add-on applied to a deployment that has no engine record yet."""
    (tmp_path / KB_ENGINE_RECORD).unlink()
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")
    return tmp_path


def _seed_lancedb(directory: Path) -> None:
    lancedb.connect(str(directory)).create_table("documents", [{"id": "x"}])


@pytest.mark.asyncio
async def test_backend_starts_with_the_add_on_on_a_new_deployment(
    monkeypatch: pytest.MonkeyPatch, add_on: Path
) -> None:
    from xagent.web import app

    initialize = AsyncMock(side_effect=_Started)
    monkeypatch.setattr(app, "_initialize_database_and_admit_runtime", initialize)

    with pytest.raises(_Started):
        await app.startup_event()
    initialize.assert_awaited_once()
    assert (add_on / KB_ENGINE_RECORD).read_text() == "milvus\n"


@pytest.mark.asyncio
async def test_backend_refuses_the_add_on_over_existing_lancedb_data(
    monkeypatch: pytest.MonkeyPatch, add_on: Path
) -> None:
    from xagent.web import app

    _seed_lancedb(add_on)
    initialize = AsyncMock()
    monkeypatch.setattr(app, "_initialize_database_and_admit_runtime", initialize)

    with pytest.raises(ConfigurationError, match=r"is lancedb \(documents hold data\)"):
        await app.startup_event()
    initialize.assert_not_called()


def test_celery_worker_starts_with_the_add_on_on_a_new_deployment(
    add_on: Path,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    worker_init.send(sender=None)
    assert (add_on / KB_ENGINE_RECORD).read_text() == "milvus\n"


def test_celery_worker_exits_with_the_add_on_over_existing_lancedb_data(
    add_on: Path,
) -> None:
    from celery.signals import worker_init

    from xagent.web.jobs import celery_app  # noqa: F401 - connects the handler

    _seed_lancedb(add_on)

    with pytest.raises(SystemExit, match=r"is lancedb \(documents hold data\)"):
        worker_init.send(sender=None)
