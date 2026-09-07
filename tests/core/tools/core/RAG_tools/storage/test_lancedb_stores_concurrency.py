"""The store data plane must survive concurrent handle access.

The coordinator offloads blocking handle calls with ``asyncio.to_thread``. Once
those are awaited concurrently on a shared loop, several worker threads reach a
single process-wide store instance at the same time. These tests pin the two
pieces of *synchronous* shared state that made that unsafe: the table-handle
cache, and the per-instance sync connection cache that no longer exists.

Async connection init is NOT covered here and is not safe yet: ``_async_conn``
is still guarded by an ``asyncio.Lock`` that deadlocks when reached from more
than one event loop. Tracked in #2200.

The table-cache assertion is a conservation law rather than a race detector:
every handle ``open_table`` returns must end up either in the cache or closed.
What actually trips these tests on an unguarded cache is the crash -- a
``move_to_end`` racing a concurrent ``clear``/``pop`` raises ``KeyError``.
The conservation law is what catches a dropped ``_safe_close_table``.

Note what is NOT reachable from a test: the cache-insert section's
read-modify-write is atomic under the GIL, so removing its lock cannot be made
to fail on CPython 3.12. That lock earns its place under free-threading only.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, List
from unittest.mock import Mock, patch

import pytest

from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBIngestionStatusStore,
    LanceDBMainPointerStore,
    LanceDBMetadataStore,
    LanceDBPromptTemplateStore,
    LanceDBVectorIndexStore,
)


class _FakeTable:
    def __init__(self, name: str, *, close_delay: float = 0.0) -> None:
        self.name = name
        self.closed = False
        self._close_delay = close_delay

    def close(self) -> None:
        # A slow close widens the window an unguarded invalidate leaves open
        # between snapshotting the cache and clearing it.
        threading.Event().wait(self._close_delay)
        self.closed = True


class _RacyConnection:
    """Hands out a distinct handle per ``open_table``, slowly enough to overlap."""

    def __init__(self, *, delay: float = 0.005, close_delay: float = 0.0) -> None:
        self._delay = delay
        self._close_delay = close_delay
        self._lock = threading.Lock()
        self.opened: List[_FakeTable] = []

    def open_table(self, name: str) -> _FakeTable:
        # The delay is what makes the interleaving reproducible rather than rare.
        threading.Event().wait(self._delay)
        table = _FakeTable(name, close_delay=self._close_delay)
        with self._lock:
            self.opened.append(table)
        return table


def _store_on(connection: _RacyConnection) -> LanceDBVectorIndexStore:
    store = LanceDBVectorIndexStore()
    store._get_connection = lambda: connection  # type: ignore[method-assign]
    return store


def _assert_every_handle_cached_or_closed(
    connection: _RacyConnection, store: LanceDBVectorIndexStore
) -> None:
    cached = set(id(table) for table in store._table_cache.values())
    leaked = [
        table
        for table in connection.opened
        if id(table) not in cached and not table.closed
    ]
    assert not leaked, (
        f"{len(leaked)} of {len(connection.opened)} opened handles are neither "
        f"cached nor closed"
    )


def test_concurrent_open_of_one_table_leaks_no_handle() -> None:
    """Threads racing on the same table must not drop an opened handle."""
    connection = _RacyConnection()
    store = _store_on(connection)
    barrier = threading.Barrier(8)
    errors: List[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=30)
            store._get_table("documents")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"concurrent _get_table raised: {errors}"
    assert len(store._table_cache) == 1
    _assert_every_handle_cached_or_closed(connection, store)


def test_concurrent_get_and_invalidate_leaks_no_handle() -> None:
    """Interleaved cache fills and invalidations must stay consistent."""
    connection = _RacyConnection(delay=0.001, close_delay=0.002)
    store = _store_on(connection)
    names = [f"table_{index}" for index in range(6)]
    errors: List[BaseException] = []
    stop = threading.Event()

    def filler(name: str) -> None:
        try:
            while not stop.is_set():
                store._get_table(name)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def invalidator() -> None:
        try:
            while not stop.is_set():
                store.invalidate_table_cache()
                store.invalidate_table_cache("table_0")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=filler, args=(name,)) for name in names]
    threads += [threading.Thread(target=invalidator) for _ in range(2)]
    for thread in threads:
        thread.start()
    threading.Event().wait(0.5)
    stop.set()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"concurrent cache access raised: {errors}"
    _assert_every_handle_cached_or_closed(connection, store)


def test_cache_never_exceeds_maxsize_under_concurrency() -> None:
    """LRU eviction must hold when many threads fill the cache at once."""
    connection = _RacyConnection(delay=0.0)
    store = _store_on(connection)
    over = store._TABLE_CACHE_MAXSIZE * 2
    errors: List[BaseException] = []

    def worker(index: int) -> None:
        try:
            store._get_table(f"table_{index}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(over)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"concurrent cache fill raised: {errors}"
    assert len(store._table_cache) <= store._TABLE_CACHE_MAXSIZE
    _assert_every_handle_cached_or_closed(connection, store)


@pytest.mark.asyncio
async def test_concurrent_to_thread_handle_calls_share_no_mutable_connection() -> None:
    """The shape #657 names: many to_thread storage calls on one shared loop."""
    connection = _RacyConnection(delay=0.002)
    store = _store_on(connection)

    results = await asyncio.gather(
        *(asyncio.to_thread(store._get_table, "documents") for _ in range(12))
    )

    assert all(table is results[0] for table in results)
    assert len(store._table_cache) == 1
    _assert_every_handle_cached_or_closed(connection, store)


@pytest.mark.parametrize(
    ("store_class", "getter_name", "cache_attr"),
    [
        (LanceDBMetadataStore, "get_raw_connection", "_conn"),
        (LanceDBVectorIndexStore, "_get_connection", "_conn"),
        (LanceDBIngestionStatusStore, "_get_sync_connection", "_sync_conn"),
        (LanceDBPromptTemplateStore, "_get_sync_connection", "_sync_conn"),
        (LanceDBMainPointerStore, "_get_sync_connection", "_sync_conn"),
    ],
)
@patch(
    "xagent.core.tools.core.RAG_tools.storage.lancedb_stores.get_connection_from_env"
)
def test_connection_is_not_cached_on_the_instance(
    mock_get_connection: Mock,
    store_class: type,
    getter_name: str,
    cache_attr: str,
) -> None:
    """Connections come from the process-wide pool, which holds its own lock.

    A per-instance cache would be unguarded shared state and would also outlive
    both the pool's TTL and ``clear_connection_cache()``. Every store that was
    carrying one is covered, so re-adding a cache to any of them fails here.
    """
    conn: Any = Mock()
    mock_get_connection.return_value = conn
    store = store_class()
    getter = getattr(store, getter_name)

    assert not hasattr(store, cache_attr)
    assert getter() is conn
    assert getter() is conn
    assert mock_get_connection.call_count == 2
