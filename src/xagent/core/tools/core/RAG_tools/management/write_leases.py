"""Short-lived write leases for operations admitted before a migration freeze.

The freeze guard admits an operation once per scope, and that admission lives in
process-local context variables, so a migration running in another process
cannot see a writer that was already admitted. Each admission records a lease
row here instead, released when the scope exits; a crashed process leaves a row
that expires. A migration waits until no unexpired lease remains before it
switches a collection.
"""

from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Optional

import pyarrow as pa  # type: ignore

from xagent.core.utils.blocking import run_blocking

from ..LanceDB.schema_manager import _safe_close_table
from ..storage.factory import get_metadata_store
from ..utils.lancedb_query_utils import list_table_names
from ..utils.string_utils import escape_lancedb_string

LEASES_TABLE = "collection_write_leases"
DEFAULT_LEASE_TTL_SECONDS = 30 * 60
DEFAULT_WAIT_POLL_SECONDS = 1.0

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


def _ensure_table(conn: Any) -> None:
    if LEASES_TABLE in list_table_names(conn):
        return
    try:
        conn.create_table(LEASES_TABLE, schema=_SCHEMA)
    except Exception:
        if LEASES_TABLE not in list_table_names(conn):
            raise


def _store_connection(store: Optional[Any]) -> Any:
    return (store or get_metadata_store()).get_raw_connection()


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
    return lease_id


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
            return []
        if time.monotonic() >= deadline:
            return remaining
        time.sleep(max(0.01, min(poll_seconds, max(0.01, deadline - time.monotonic()))))
