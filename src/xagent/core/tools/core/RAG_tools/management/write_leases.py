"""Short-lived write leases for operations admitted before a migration freeze.

The freeze guard admits an operation once per scope, and that admission lives in
process-local context variables, so a migration running in another process cannot
see a writer that was already admitted. Each admission records a lease row here
instead, released when the scope exits; a scope that outlives the TTL renews its
rows, and a crashed process leaves rows that expire.

Ordering is the point: the writer inserts its lease *before* it reads the freeze
state, and the migration writes the freeze *before* it reads leases. Either the
writer sees the freeze and refuses, or the migration sees the lease and waits.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Optional

import pyarrow as pa  # type: ignore

from xagent.core.utils.blocking import run_blocking

from ..LanceDB.schema_manager import _safe_close_table
from ..storage.factory import get_metadata_store, get_vector_index_store
from ..utils.lancedb_query_utils import list_table_names
from ..utils.string_utils import escape_lancedb_string

LEASES_TABLE = "collection_write_leases"
# Short enough that a crashed operation stops delaying a migration quickly; a
# live scope renews its rows, so long ingests do not lose their lease.
DEFAULT_LEASE_TTL_SECONDS = 120.0
DEFAULT_WAIT_POLL_SECONDS = 1.0
COMPACT_EVERY_ACQUIRES = 200

logger = logging.getLogger(__name__)

_SCHEMA = pa.schema(
    [
        ("name", pa.string()),
        ("lease_id", pa.string()),
        ("operation", pa.string()),
        ("owner", pa.string()),
        ("created_at", pa.timestamp("us")),
        ("expires_at", pa.timestamp("us")),
    ]
)

_acquire_count = 0
_acquire_lock = threading.Lock()


@dataclass(frozen=True)
class WriteLease:
    """One admitted operation that may still be writing to a collection."""

    name: str
    lease_id: str
    operation: str
    owner: str
    created_at: Optional[datetime]
    expires_at: Optional[datetime]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _store_connection(store: Optional[Any]) -> Any:
    return (store or get_metadata_store()).get_raw_connection()


def _ensure_table(conn: Any) -> None:
    if LEASES_TABLE in list_table_names(conn):
        return
    try:
        conn.create_table(LEASES_TABLE, schema=_SCHEMA)
    except Exception:
        if LEASES_TABLE not in list_table_names(conn):
            raise


def _maybe_compact() -> None:
    """Merge the lease table's fragments once every few hundred operations.

    Rows are deleted on release, so the table only ever holds in-flight leases;
    what accumulates is fragments and versions from the add/delete/heartbeat
    writes. `trigger_reindex` compacts and prunes versions for us.
    """
    global _acquire_count
    with _acquire_lock:
        _acquire_count += 1
        due = _acquire_count % COMPACT_EVERY_ACQUIRES == 0
    if not due:
        return
    try:
        get_vector_index_store().trigger_reindex(LEASES_TABLE)
    except Exception:  # noqa: BLE001 - compaction must never fail a write
        logger.exception("Failed to compact %s", LEASES_TABLE)


def acquire_lease(
    name: str,
    operation: str,
    *,
    ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
    store: Optional[Any] = None,
) -> str:
    """Record that this process may write ``name`` until it releases the lease.

    Raises on failure: an admission the migration cannot see must not proceed
    silently, the same way the freeze read fails closed.
    """
    conn = _store_connection(store)
    _ensure_table(conn)
    now = _now()
    lease_id = uuid.uuid4().hex
    row = {
        "name": name,
        "lease_id": lease_id,
        "operation": operation,
        "owner": f"{socket.gethostname()}:{os.getpid()}",
        "created_at": now,
        "expires_at": now + timedelta(seconds=max(1.0, float(ttl_seconds))),
    }
    table = conn.open_table(LEASES_TABLE)
    try:
        table.add([row])
    finally:
        _safe_close_table(table)
    _maybe_compact()
    return lease_id


def renew_leases(
    lease_ids: Iterable[str],
    *,
    ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
    store: Optional[Any] = None,
) -> int:
    """Extend these leases; returns how many rows were updated."""
    ids = [lease_id for lease_id in lease_ids if lease_id]
    if not ids:
        return 0
    conn = _store_connection(store)
    if LEASES_TABLE not in list_table_names(conn):
        return 0
    table = conn.open_table(LEASES_TABLE)
    try:
        updated = 0
        for lease_id in ids:
            where = f"lease_id = '{escape_lancedb_string(lease_id)}'"
            rows = table.search().where(where).to_arrow().to_pylist()
            if not rows:
                continue
            expires_at = rows[0].get("expires_at")
            target = _now() + timedelta(seconds=max(1.0, float(ttl_seconds)))
            if isinstance(expires_at, datetime) and expires_at >= target:
                continue
            updated += int(table.update(where, {"expires_at": target}).rows_updated)
        return updated
    finally:
        _safe_close_table(table)


def release_leases(lease_ids: Iterable[str], *, store: Optional[Any] = None) -> int:
    """Delete these leases; a missing table or row is a no-op."""
    ids = [lease_id for lease_id in lease_ids if lease_id]
    if not ids:
        return 0
    conn = _store_connection(store)
    if LEASES_TABLE not in list_table_names(conn):
        return 0
    table = conn.open_table(LEASES_TABLE)
    try:
        for lease_id in ids:
            table.delete(f"lease_id = '{escape_lancedb_string(lease_id)}'")
    finally:
        _safe_close_table(table)
    return len(ids)


def live_leases(name: str, *, store: Optional[Any] = None) -> List[WriteLease]:
    """The unexpired leases for ``name``; a missing table means none."""
    conn = _store_connection(store)
    if LEASES_TABLE not in list_table_names(conn):
        return []
    table = conn.open_table(LEASES_TABLE)
    try:
        where = f"name = '{escape_lancedb_string(name)}'"
        rows = table.search().where(where).to_arrow().to_pylist()
    finally:
        _safe_close_table(table)
    now = _now()
    live: List[WriteLease] = []
    for row in rows:
        expires_at = row.get("expires_at")
        if not isinstance(expires_at, datetime) or expires_at <= now:
            continue
        live.append(
            WriteLease(
                name=str(row.get("name") or ""),
                lease_id=str(row.get("lease_id") or ""),
                operation=str(row.get("operation") or ""),
                owner=str(row.get("owner") or ""),
                created_at=row.get("created_at"),
                expires_at=expires_at,
            )
        )
    return live


def purge_expired_leases(name: str, *, store: Optional[Any] = None) -> int:
    """Delete ``name``'s expired rows only; live rows of any process stay."""
    conn = _store_connection(store)
    if LEASES_TABLE not in list_table_names(conn):
        return 0
    table = conn.open_table(LEASES_TABLE)
    try:
        where = f"name = '{escape_lancedb_string(name)}'"
        rows = table.search().where(where).to_arrow().to_pylist()
        now = _now()
        expired = [
            str(row.get("lease_id") or "")
            for row in rows
            if isinstance(row.get("expires_at"), datetime) and row["expires_at"] <= now
        ]
        for lease_id in expired:
            if lease_id:
                table.delete(f"lease_id = '{escape_lancedb_string(lease_id)}'")
        return len([lease_id for lease_id in expired if lease_id])
    finally:
        _safe_close_table(table)


def wait_for_no_leases(
    name: str,
    *,
    timeout_seconds: float,
    poll_seconds: float = DEFAULT_WAIT_POLL_SECONDS,
    store: Optional[Any] = None,
) -> List[WriteLease]:
    """Poll until ``name`` has no live lease; returns what is left at the deadline."""
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        remaining = live_leases(name, store=store)
        if not remaining:
            purge_expired_leases(name, store=store)
            return []
        if time.monotonic() >= deadline:
            return remaining
        time.sleep(max(0.01, min(poll_seconds, max(0.01, deadline - time.monotonic()))))


async def acquire_lease_async(
    name: str,
    operation: str,
    *,
    ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
    store: Optional[Any] = None,
) -> str:
    return await run_blocking(
        acquire_lease, name, operation, ttl_seconds=ttl_seconds, store=store
    )


async def release_leases_async(
    lease_ids: Iterable[str], *, store: Optional[Any] = None
) -> None:
    """Release off the event loop; a failure is logged, never raised."""
    ids = list(lease_ids)
    try:
        await run_blocking(release_leases, ids, store=store)
    except Exception:  # noqa: BLE001 - the write already happened
        logger.exception("Failed to release write leases: %s", sorted(ids))


class LeaseSet:
    """The leases one operation holds, kept alive for as long as it runs."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        store: Optional[Any] = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._store = store
        self._ids: List[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def lease_ids(self) -> List[str]:
        with self._lock:
            return list(self._ids)

    def add(self, lease_id: str) -> None:
        with self._lock:
            self._ids.append(lease_id)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._renew_loop,
                    name="kb-write-lease-heartbeat",
                    daemon=True,
                )
                self._thread.start()

    def _renew_loop(self) -> None:
        interval = max(0.05, float(self._ttl) / 3.0)
        while not self._stop.wait(interval):
            try:
                renew_leases(self.lease_ids, ttl_seconds=self._ttl, store=self._store)
            except Exception:  # noqa: BLE001 - keep trying until the scope ends
                logger.exception("Failed to renew write leases")

    def stop(self) -> List[str]:
        """Stop renewing and return the ids; safe to call more than once."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        return self.lease_ids

    def release(self) -> None:
        layer = self.stop()
        with self._lock:
            self._ids.clear()
        try:
            release_leases(layer, store=self._store)
        except Exception:  # noqa: BLE001 - the write already happened
            logger.exception("Failed to release write leases: %s", sorted(layer))
