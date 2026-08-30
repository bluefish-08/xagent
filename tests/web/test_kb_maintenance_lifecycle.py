"""In-process KB index maintenance loop wiring (#1557).

``get_celery_enabled`` defaults to False, so Gunicorn-only and local
deployments have no worker and no Beat; without this loop they would get no
KB maintenance at all once it stopped hanging off the ingestion hook.
"""

from __future__ import annotations

import asyncio

import pytest

from xagent.web import app as app_module


def _patch_loop(monkeypatch: pytest.MonkeyPatch, started: asyncio.Event) -> None:
    async def fake_loop(*, poll_interval_seconds: int) -> None:
        assert poll_interval_seconds == 11
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        app_module, "get_kb_index_maintenance_interval_seconds", lambda: 11
    )
    monkeypatch.setattr(app_module, "run_kb_maintenance_loop", fake_loop)


@pytest.mark.asyncio
async def test_loop_runs_when_celery_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_module, "get_celery_enabled", lambda: False)
    _patch_loop(monkeypatch, started)

    task = app_module.start_kb_maintenance_task(app_module.app)

    assert task is app_module.app.state.kb_maintenance_task
    await asyncio.wait_for(started.wait(), timeout=1)

    await app_module.stop_kb_maintenance_task(app_module.app)
    assert task.cancelled()
    assert app_module.app.state.kb_maintenance_task is None


@pytest.mark.asyncio
async def test_loop_stays_off_when_celery_beat_owns_maintenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both running would fight over the same per-table compaction lock."""
    started = asyncio.Event()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_module, "get_celery_enabled", lambda: True)
    _patch_loop(monkeypatch, started)

    assert app_module.start_kb_maintenance_task(app_module.app) is None

    assert not started.is_set()
    assert app_module.app.state.kb_maintenance_task is None


@pytest.mark.asyncio
async def test_loop_stays_off_under_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    monkeypatch.setattr(app_module, "get_celery_enabled", lambda: False)
    _patch_loop(monkeypatch, started)

    assert app_module.start_kb_maintenance_task(app_module.app) is None

    assert not started.is_set()


@pytest.mark.asyncio
async def test_loop_survives_a_failing_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """A permanently broken database must not silently end the loop."""
    from xagent.web.services import kb_maintenance

    calls = 0
    third = asyncio.Event()

    def boom() -> dict:
        nonlocal calls
        calls += 1
        if calls >= 3:
            third.set()
        raise RuntimeError("disk full")

    monkeypatch.setattr(kb_maintenance, "sweep_kb_storage", boom)
    task = asyncio.create_task(
        kb_maintenance.run_kb_maintenance_loop(poll_interval_seconds=0)
    )
    try:
        await asyncio.wait_for(third.wait(), timeout=2)
    finally:
        task.cancel()

    assert calls >= 3
