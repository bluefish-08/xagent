"""In-process KB index maintenance loop wiring (#1557).

The loop runs in every deployment: XAGENT_CELERY_ENABLED says durable jobs go
to a worker, not that Beat exists (``dev_background_jobs.py --no-beat`` sets
one without the other), so gating on it would leave that deployment with no
maintenance at all once it stopped hanging off the ingestion hook.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from xagent.web import app as app_module


def _patch_loop(monkeypatch: pytest.MonkeyPatch, started: asyncio.Event) -> None:
    async def fake_loop(*, poll_interval_seconds: int, stop_event=None) -> None:
        assert poll_interval_seconds == 11
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        app_module, "get_kb_index_maintenance_interval_seconds", lambda: 11
    )
    monkeypatch.setattr(app_module, "run_kb_maintenance_loop", fake_loop)


@pytest.mark.parametrize("celery_enabled", [False, True])
@pytest.mark.asyncio
async def test_loop_runs_whatever_celery_is_set_to(
    monkeypatch: pytest.MonkeyPatch, celery_enabled: bool
) -> None:
    """Celery being enabled is not evidence that Beat is running.

    ``dev_background_jobs.py --no-beat`` sets XAGENT_CELERY_ENABLED=true and
    starts no Beat; gating on it left that deployment with zero maintenance.
    """
    started = asyncio.Event()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("XAGENT_CELERY_ENABLED", str(celery_enabled).lower())
    _patch_loop(monkeypatch, started)

    task = app_module.start_kb_maintenance_task(app_module.app)

    assert task is app_module.app.state.kb_maintenance_task
    await asyncio.wait_for(started.wait(), timeout=1)

    await app_module.stop_kb_maintenance_task(app_module.app)
    assert app_module.app.state.kb_maintenance_task is None


@pytest.mark.asyncio
async def test_loop_stays_off_under_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    _patch_loop(monkeypatch, started)

    assert app_module.start_kb_maintenance_task(app_module.app) is None

    assert not started.is_set()


@pytest.mark.asyncio
async def test_shutdown_sets_the_stop_flag_the_sweep_can_see(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the task cannot stop a sweep already in the executor.

    The flag is the only thing that thread can observe, so shutdown has to set
    it -- cancelling alone leaves the sweep running to completion.
    """
    order: list[str] = []
    running = asyncio.Event()

    async def fake_loop(*, poll_interval_seconds: int, stop_event) -> None:
        running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            order.append("cancelled" if not stop_event.is_set() else "stop-then-cancel")
            raise

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        app_module, "get_kb_index_maintenance_interval_seconds", lambda: 11
    )
    monkeypatch.setattr(app_module, "run_kb_maintenance_loop", fake_loop)

    app_module.start_kb_maintenance_task(app_module.app)
    await asyncio.wait_for(running.wait(), timeout=1)

    await app_module.stop_kb_maintenance_task(app_module.app)

    assert order == ["stop-then-cancel"]
    assert app_module.app.state.kb_maintenance_stop is None


@pytest.mark.asyncio
async def test_shutdown_wait_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unfinished sweep must not hold shutdown past the grace period."""

    async def stubborn_loop(*, poll_interval_seconds: int, stop_event) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Stands in for an executor thread that ignores the cancellation.
            await asyncio.sleep(3600)

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        app_module, "get_kb_index_maintenance_interval_seconds", lambda: 11
    )
    monkeypatch.setattr(app_module, "run_kb_maintenance_loop", stubborn_loop)
    monkeypatch.setattr(app_module, "KB_MAINTENANCE_SHUTDOWN_TIMEOUT_SECONDS", 0.1)

    task = app_module.start_kb_maintenance_task(app_module.app)
    await asyncio.sleep(0)

    await asyncio.wait_for(
        app_module.stop_kb_maintenance_task(app_module.app), timeout=2
    )

    assert task is not None and not task.done()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_loop_survives_a_failing_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """A permanently broken database must not silently end the loop."""
    from xagent.web.services import kb_maintenance

    calls = 0
    third = asyncio.Event()

    def boom(stop_event=None) -> dict:
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


@pytest.mark.asyncio
async def test_loop_sweeps_before_its_first_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker recycled inside one interval must still get a sweep.

    Gunicorn ``max_requests`` recycling is routine in exactly the deployments
    this loop exists for, and sleeping first would mean they never sweep.
    """
    from xagent.web.services import kb_maintenance

    swept = asyncio.Event()
    monkeypatch.setattr(
        kb_maintenance, "sweep_kb_storage", lambda stop_event=None: swept.set()
    )
    task = asyncio.create_task(
        kb_maintenance.run_kb_maintenance_loop(poll_interval_seconds=3600)
    )
    try:
        await asyncio.wait_for(swept.wait(), timeout=2)
    finally:
        task.cancel()


def test_startup_and_shutdown_wire_the_loop() -> None:
    """The loop is dead code unless the lifespan starts and stops it."""
    import inspect

    assert "start_kb_maintenance_task(app)" in inspect.getsource(
        app_module.startup_event
    )
    assert "await stop_kb_maintenance_task(app)" in inspect.getsource(
        app_module.shutdown_event
    )
