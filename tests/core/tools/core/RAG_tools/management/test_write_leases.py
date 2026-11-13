"""Write leases: admitted operations stay visible to a migration across processes."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import CollectionMigratingError
from xagent.core.tools.core.RAG_tools.management import write_leases as leases
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    CollectionMigrationState,
    admitted_async,
    ensure_collection_writable,
    ensure_collection_writable_async,
    pending_collection,
    write_migration_state,
)
from xagent.core.tools.core.RAG_tools.management.write_leases import (
    DEFAULT_LEASE_TTL_SECONDS,
    LEASES_TABLE,
    LeaseSet,
    acquire_lease,
    live_leases,
    purge_expired_leases,
    release_leases,
    renew_leases,
    wait_for_no_leases,
)


def _freeze(name: str = "kb") -> None:
    write_migration_state(name, CollectionMigrationState(state=FROZEN))


def _insert_lease(name: str, lease_id: str, expires_at: datetime) -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    conn = get_metadata_store().get_raw_connection()
    leases._ensure_table(conn)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    table = conn.open_table(LEASES_TABLE)
    try:
        table.add(
            [
                {
                    "name": name,
                    "lease_id": lease_id,
                    "operation": "old",
                    "owner": "host:1",
                    "created_at": now,
                    "expires_at": expires_at,
                }
            ]
        )
    finally:
        _safe_close_table(table)


def test_a_lease_is_live_until_it_is_released() -> None:
    lease_id = acquire_lease("kb", "test")

    live = live_leases("kb")
    assert [item.lease_id for item in live] == [lease_id]
    assert live[0].operation == "test"

    assert release_leases([lease_id]) == 1
    assert live_leases("kb") == []


def test_the_default_ttl_is_short_enough_for_a_crashed_writer_to_clear() -> None:
    assert DEFAULT_LEASE_TTL_SECONDS <= 300


def test_expired_leases_are_not_live() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    _insert_lease("kb", "expired", now - timedelta(seconds=1))
    _insert_lease("kb", "live", now + timedelta(minutes=5))

    assert [lease.lease_id for lease in live_leases("kb")] == ["live"]


def test_renewal_extends_a_lease() -> None:
    lease_id = acquire_lease("kb", "test", ttl_seconds=60)

    assert renew_leases([lease_id], ttl_seconds=600) == 1
    assert live_leases("kb")[0].expires_at is not None


def test_renewal_never_shortens_a_lease() -> None:
    lease_id = acquire_lease("kb", "test", ttl_seconds=600)
    before = live_leases("kb")[0].expires_at

    assert renew_leases([lease_id], ttl_seconds=60) == 0

    assert live_leases("kb")[0].expires_at == before


def test_purging_removes_only_expired_rows_of_that_collection() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    _insert_lease("kb", "expired", now - timedelta(seconds=1))
    _insert_lease("kb", "live", now + timedelta(minutes=5))
    _insert_lease("other", "expired-other", now - timedelta(seconds=1))

    assert purge_expired_leases("kb") == 1

    assert [lease.lease_id for lease in live_leases("kb")] == ["live"]
    assert live_leases("other") == []  # expired, still present, ignored


def test_waits_return_what_is_left_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquire_lease("kb", "test")
    clock = {"now": 100.0}
    monkeypatch.setattr(leases.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        leases.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds)
    )

    remaining = wait_for_no_leases("kb", timeout_seconds=2.0, poll_seconds=1.0)

    assert [item.operation for item in remaining] == ["test"]


def test_a_waiter_returns_once_the_lease_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_id = acquire_lease("kb", "test")
    clock = {"now": 100.0}
    monkeypatch.setattr(leases.time, "monotonic", lambda: clock["now"])

    def _sleep(seconds: float) -> None:
        clock["now"] += seconds
        release_leases([lease_id])

    monkeypatch.setattr(leases.time, "sleep", _sleep)

    assert wait_for_no_leases("kb", timeout_seconds=10.0, poll_seconds=1.0) == []


def test_the_guard_writes_the_lease_before_it_reads_the_freeze_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state as state

    observed: list = []

    def _read(name: str, store=None):
        observed.append([lease.lease_id for lease in live_leases(name)])
        return CollectionMigrationState()

    monkeypatch.setattr(state, "read_migration_state", _read)

    with pending_collection("kb"):
        ensure_collection_writable("kb")

    assert observed and len(observed[0]) == 1


async def test_the_async_guard_writes_the_lease_before_it_reads_the_freeze_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state as state

    observed: list = []

    def _read(name: str, store=None):
        observed.append([lease.lease_id for lease in live_leases(name)])
        return CollectionMigrationState()

    monkeypatch.setattr(state, "read_migration_state", _read)

    with pending_collection("kb"):
        await ensure_collection_writable_async("kb")

    assert observed and len(observed[0]) == 1


def test_a_frozen_collection_refuses_and_leaves_no_lease() -> None:
    _freeze("kb")

    with pytest.raises(CollectionMigratingError):
        with pending_collection("kb"):
            ensure_collection_writable("kb")

    assert live_leases("kb") == []


def test_one_lease_covers_every_guard_call_of_the_operation() -> None:
    with pending_collection("kb"):
        ensure_collection_writable("kb")
        ensure_collection_writable("kb")
        ensure_collection_writable("kb")
        assert len(live_leases("kb")) == 1

    assert live_leases("kb") == []


def test_a_rename_scope_leases_both_names() -> None:
    with pending_collection("old", "new"):
        ensure_collection_writable("old")
        ensure_collection_writable("new")
        assert len(live_leases("old")) == 1
        assert len(live_leases("new")) == 1

    assert live_leases("old") == []
    assert live_leases("new") == []


def test_a_scope_that_raises_releases_its_leases() -> None:
    with pytest.raises(RuntimeError):
        with pending_collection("kb"):
            ensure_collection_writable("kb")
            raise RuntimeError("boom")

    assert live_leases("kb") == []


def test_a_guard_outside_a_pending_scope_takes_no_lease() -> None:
    ensure_collection_writable("kb")

    assert live_leases("kb") == []


async def test_admitted_async_holds_a_lease_until_the_scope_exits() -> None:
    async with admitted_async("kb"):
        assert len(live_leases("kb")) == 1

    assert live_leases("kb") == []


async def test_admitted_async_takes_the_lease_before_it_reads_the_freeze_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import migration_state as state

    observed: list = []

    def _read(name: str, store=None):
        observed.append([lease.lease_id for lease in live_leases(name)])
        return CollectionMigrationState()

    monkeypatch.setattr(state, "read_migration_state", _read)

    async with admitted_async("kb"):
        pass

    assert observed and len(observed[0]) == 1


def test_a_waiter_clears_expired_rows_it_finds_nothing_live_for() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    _insert_lease("kb", "expired", now - timedelta(seconds=1))

    assert wait_for_no_leases("kb", timeout_seconds=0.0) == []

    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    conn = get_metadata_store().get_raw_connection()
    table = conn.open_table(LEASES_TABLE)
    try:
        assert table.count_rows() == 0
    finally:
        _safe_close_table(table)


async def test_admitted_async_inside_a_pending_scope_keeps_one_lease() -> None:
    with pending_collection("kb"):
        async with admitted_async("kb"):
            assert len(live_leases("kb")) == 1

    assert live_leases("kb") == []


def test_the_heartbeat_extends_a_held_lease() -> None:
    held = LeaseSet(ttl_seconds=0.5)
    try:
        held.add(acquire_lease("kb", "test", ttl_seconds=0.5))
        before = live_leases("kb")[0].expires_at
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            rows = live_leases("kb")
            if rows and rows[0].expires_at > before:
                break
            time.sleep(0.05)
        assert live_leases("kb")[0].expires_at > before
    finally:
        held.release()


def test_the_lease_table_is_compacted_periodically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compacted: list = []

    class _Store:
        def trigger_reindex(self, table: str) -> bool:
            compacted.append(table)
            return True

    monkeypatch.setattr(leases, "get_vector_index_store", lambda: _Store())
    monkeypatch.setattr(leases, "_acquire_count", leases.COMPACT_EVERY_ACQUIRES - 1)

    acquire_lease("kb", "test")

    assert compacted == [LEASES_TABLE]
