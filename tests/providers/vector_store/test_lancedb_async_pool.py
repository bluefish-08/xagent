"""Async LanceDB connections must be reachable from every event loop.

The stores that use them are process-wide singletons, while the codebase keeps
creating fresh event loops (``asyncio.run`` in the coordinator's sync wrappers,
the parse path, collection_manager). Guarding a per-instance ``_async_conn``
with an ``asyncio.Lock`` deadlocked that shape: the lock binds to whichever
loop first contended it, and the release path wakes that loop's future without
``call_soon_threadsafe``, so every other loop waits forever (#2200).

Every test here joins with a timeout and asserts the thread finished, so a
regression fails the suite instead of hanging CI.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, List
from unittest.mock import patch

import pytest

from xagent.providers.vector_store import lancedb as lancedb_module
from xagent.providers.vector_store.lancedb import (
    clear_connection_cache,
    get_async_connection_from_env,
)

JOIN_TIMEOUT = 30


@pytest.fixture(autouse=True)
def _isolate_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path))
    clear_connection_cache()
    yield
    clear_connection_cache()


class _FakeAsyncConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _slow_connect_async(delay: float = 0.2):
    """A connect_async whose latency is wide enough for openers to collide."""
    created: List[_FakeAsyncConnection] = []

    async def connect_async(_uri: str) -> _FakeAsyncConnection:
        await asyncio.sleep(delay)
        conn = _FakeAsyncConnection()
        created.append(conn)
        return conn

    return connect_async, created


def _run_in_own_loop(coro_factory, results: list, errors: list) -> threading.Thread:
    def target() -> None:
        try:
            results.append(asyncio.run(coro_factory()))
        except BaseException as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_concurrent_init_from_two_loops_does_not_deadlock() -> None:
    """Two threads, two loops, one uninitialized pool entry: both must return."""
    connect_async, created = _slow_connect_async()
    results: List[Any] = []
    errors: List[str] = []

    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        threads = [
            _run_in_own_loop(get_async_connection_from_env, results, errors)
            for _ in range(2)
        ]
        for thread in threads:
            thread.join(timeout=JOIN_TIMEOUT)

    assert not [t for t in threads if t.is_alive()], (
        "a thread never returned from async connection init -- deadlock"
    )
    assert not errors, f"async connection init raised: {errors}"
    assert len(results) == 2
    assert results[0] is results[1], "both loops must share one pooled connection"
    # The opener that lost the insert race must not leak.
    leaked = [c for c in created if c is not results[0] and not c.closed]
    assert not leaked, f"{len(leaked)} superseded connections were left open"


def test_many_loops_reuse_one_pooled_connection() -> None:
    """A cached connection is handed to loops that had no part in creating it."""
    connect_async, created = _slow_connect_async(delay=0.0)
    results: List[Any] = []
    errors: List[str] = []

    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        for _ in range(6):
            thread = _run_in_own_loop(get_async_connection_from_env, results, errors)
            thread.join(timeout=JOIN_TIMEOUT)
            assert not thread.is_alive()

    assert not errors, f"async connection init raised: {errors}"
    assert len(results) == 6
    assert all(conn is results[0] for conn in results)
    assert len(created) == 1, f"connect_async ran {len(created)} times, expected 1"


def test_clear_connection_cache_drops_and_closes_async_connections() -> None:
    """``reset_rag_storage_for_tests`` relies on this to reset *all* state.

    Before the pool existed, the async connection died with the store instance
    that `StorageFactory.reset_all()` threw away; now it outlives that, so the
    cache clear has to take it down.
    """
    connect_async, created = _slow_connect_async(delay=0.0)
    results: List[Any] = []
    errors: List[str] = []

    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        thread = _run_in_own_loop(get_async_connection_from_env, results, errors)
        thread.join(timeout=JOIN_TIMEOUT)
        assert not errors and len(results) == 1

        clear_connection_cache()
        assert created[0].closed, "cleared async connection was not closed"

        thread = _run_in_own_loop(get_async_connection_from_env, results, errors)
        thread.join(timeout=JOIN_TIMEOUT)

    assert not errors, f"async connection init raised: {errors}"
    assert len(created) == 2, "cache clear must force a fresh connect_async"
    assert results[1] is not results[0]


def test_store_async_methods_share_the_pool_across_loops() -> None:
    """The shape #2200 actually breaks: one store singleton, several loops."""
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        LanceDBIngestionStatusStore,
        LanceDBVectorIndexStore,
    )

    connect_async, created = _slow_connect_async()
    results: List[Any] = []
    errors: List[str] = []
    vector_store = LanceDBVectorIndexStore()
    status_store = LanceDBIngestionStatusStore()

    with patch.object(lancedb_module.lancedb, "connect_async", connect_async):
        threads = [
            _run_in_own_loop(store._get_async_connection, results, errors)
            for store in (vector_store, status_store, vector_store)
        ]
        for thread in threads:
            thread.join(timeout=JOIN_TIMEOUT)

    assert not [t for t in threads if t.is_alive()], (
        "a store's async connection init never returned -- deadlock"
    )
    assert not errors, f"store async init raised: {errors}"
    assert len(results) == 3
    assert all(conn is results[0] for conn in results)
    assert not hasattr(vector_store, "_async_lock")
    assert not hasattr(status_store, "_async_lock")
