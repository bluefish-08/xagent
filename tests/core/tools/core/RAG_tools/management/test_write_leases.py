"""Write leases: admitted operations stay visible to a migration across processes."""

from __future__ import annotations

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
    LEASES_TABLE,
    acquire_lease,
    live_leases,
    release_leases,
    wait_for_no_leases,
)


def _freeze(name: str = "kb") -> None:
    write_migration_state(name, CollectionMigrationState(state=FROZEN))


def _insert_expired(name: str = "kb") -> None:
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
                    "lease_id": "expired",
                    "operation": "old",
                    "owner": "host:1",
                    "created_at": now - timedelta(hours=2),
                    "expires_at": now - timedelta(hours=1),
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


def test_expired_leases_are_not_live() -> None:
    _insert_expired("kb")

    assert live_leases("kb") == []


def test_leases_are_scoped_to_their_collection() -> None:
    acquire_lease("kb", "test")
    acquire_lease("other", "test")

    assert len(live_leases("kb")) == 1
    assert len(live_leases("other")) == 1
    assert live_leases("missing") == []


def test_wait_returns_immediately_without_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(leases.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert wait_for_no_leases("kb", timeout_seconds=5.0) == []
    assert sleeps == []


def test_wait_returns_what_is_left_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquire_lease("kb", "test")
    clock = {"now": 100.0}
    monkeypatch.setattr(leases.time, "monotonic", lambda: clock["now"])

    def _sleep(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(leases.time, "sleep", _sleep)

    remaining = wait_for_no_leases("kb", timeout_seconds=2.0, poll_seconds=1.0)

    assert [item.operation for item in remaining] == ["test"]


def test_releasing_leases_makes_a_waiter_return(
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


def test_a_guard_inside_a_pending_scope_holds_one_lease() -> None:
    with pending_collection("kb"):
        ensure_collection_writable("kb")
        ensure_collection_writable("kb")
        assert len(live_leases("kb")) == 1

    assert live_leases("kb") == []


def test_a_guard_releases_its_lease_when_the_scope_raises() -> None:
    with pytest.raises(RuntimeError):
        with pending_collection("kb"):
            ensure_collection_writable("kb")
            raise RuntimeError("boom")

    assert live_leases("kb") == []


def test_a_guard_outside_a_pending_scope_takes_no_lease() -> None:
    ensure_collection_writable("kb")

    assert live_leases("kb") == []


def test_a_frozen_collection_gets_no_lease() -> None:
    _freeze("kb")

    with pytest.raises(CollectionMigratingError):
        with pending_collection("kb"):
            ensure_collection_writable("kb")

    assert live_leases("kb") == []


async def test_the_async_guard_holds_a_lease_until_the_scope_exits() -> None:
    with pending_collection("kb"):
        await ensure_collection_writable_async("kb")
        assert len(live_leases("kb")) == 1

    assert live_leases("kb") == []


async def test_admitted_async_holds_a_lease_until_the_scope_exits() -> None:
    async with admitted_async("kb"):
        assert len(live_leases("kb")) == 1

    assert live_leases("kb") == []
