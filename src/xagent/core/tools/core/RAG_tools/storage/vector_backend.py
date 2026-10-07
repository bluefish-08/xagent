"""KB engine selection and the per-deployment engine lock.

``XAGENT_VECTOR_BACKEND`` names this deployment's KB engine; the
:class:`~.contracts.VectorIndexStore` is always the LanceDB ledger. The engine is
recorded in the LanceDB data directory at first start, and startup is refused
whenever the setting differs from the record.

Test isolation: pytest should call ``reset_rag_storage_for_tests`` in
``storage.factory`` instead of importing a specific provider.
"""

from __future__ import annotations

import logging
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from filelock import FileLock, Timeout

from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Primary env var (namespaced to avoid collisions with other libs).
VECTOR_BACKEND_ENV: Final[str] = "XAGENT_VECTOR_BACKEND"

# Backward-compatible alias used in some deployments / docs.
VECTOR_BACKEND_ENV_LEGACY: Final[str] = "VECTOR_STORE_BACKEND"

KB_ENGINE_RECORD: Final[str] = ".kb-engine"
_KB_DATA_TABLES: Final = ("documents", "collection_config", "collection_metadata")
_LOCK_TIMEOUT_SECONDS: Final = 120


class KBStorageBackend(StrEnum):
    """KB engine of a deployment; collection bindings use the same values."""

    LANCEDB = "lancedb"
    MILVUS = "milvus"
    QDRANT = "qdrant"


VectorBackend = KBStorageBackend


def _parse_backend(raw: str) -> VectorBackend:
    """Parse and validate backend string."""
    key = raw.strip().lower()
    if not key:
        return VectorBackend.LANCEDB
    try:
        return VectorBackend(key)
    except ValueError as exc:
        allowed = ", ".join(sorted(b.value for b in VectorBackend))
        raise ConfigurationError(
            f"Invalid {VECTOR_BACKEND_ENV}={raw!r}. Choose one of: {allowed}."
        ) from exc


def get_configured_vector_backend() -> VectorBackend:
    """Read configured vector backend from the environment.

    Precedence: ``XAGENT_VECTOR_BACKEND``, then ``VECTOR_STORE_BACKEND``,
    then default ``lancedb``.

    Returns:
        Selected :class:`VectorBackend`.

    Raises:
        ConfigurationError: If the value is not a known backend name.
    """
    raw = os.environ.get(VECTOR_BACKEND_ENV)
    if raw is None or raw.strip() == "":
        raw = os.environ.get(VECTOR_BACKEND_ENV_LEGACY, "")
    return _parse_backend(raw)


def require_implemented_vector_backend(backend: VectorBackend) -> None:
    """Refuse a KB engine that is not implemented yet; Qdrant is a reserved value.

    Args:
        backend: Resolved backend.

    Raises:
        ConfigurationError: If the backend is known but not implemented yet.
    """
    if backend is VectorBackend.LANCEDB:
        return
    raise ConfigurationError(
        f"KB engine {backend.value!r} is not implemented yet. "
        f"Set {VECTOR_BACKEND_ENV}=lancedb (default)."
    )


def _tables_with_rows(conn: Any, names: list[str]) -> list[str]:
    blocking = []
    for name in names:
        try:
            if conn.open_table(name).count_rows() > 0:
                blocking.append(name)
        except Exception:  # noqa: BLE001 - an unreadable table counts as data
            blocking.append(name)
    return blocking


def _detect_engine(
    db_dir: str, configured: KBStorageBackend
) -> tuple[KBStorageBackend, list[str]]:
    import lancedb

    from ..utils.lancedb_query_utils import list_table_names

    # Uncached, so a Celery parent does not fork with an open connection.
    conn = lancedb.connect(db_dir)
    names = list_table_names(conn)
    blocking = _tables_with_rows(conn, [name for name in names if name == "kb_ids"])
    if blocking:
        return KBStorageBackend.MILVUS, blocking
    blocking = _tables_with_rows(
        conn,
        [
            name
            for name in names
            if name in _KB_DATA_TABLES or name.startswith("embeddings_")
        ],
    )
    return (KBStorageBackend.LANCEDB if blocking else configured), blocking


def _read_record(path: Path) -> KBStorageBackend:
    try:
        return KBStorageBackend(path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"KB engine record {path} does not name a known engine ({exc}); "
            "delete the record file and restart to detect the engine again."
        ) from exc


def _write_record(path: Path, engine: KBStorageBackend) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(f"{engine.value}\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def lock_deployment_kb_engine() -> KBStorageBackend:
    """Record the KB engine at first start and refuse a setting that differs.

    Without a record, rows in ``kb_ids`` mean Milvus and rows in the KB data
    tables mean an upgraded LanceDB deployment; otherwise the setting is
    recorded. Detection and the write share one file lock. When the directory
    cannot be written, the detected engine is compared without a record.

    Raises:
        ConfigurationError: If the setting is not implemented, the record does
            not name a known engine, the lock is not acquired in time, or the
            setting differs from the record.
    """
    from ......providers.vector_store.lancedb import LanceDBConnectionManager

    configured = get_configured_vector_backend()
    require_implemented_vector_backend(configured)
    db_dir = LanceDBConnectionManager().resolve_dir_from_env()
    path = Path(db_dir) / KB_ENGINE_RECORD
    lock_path = path.with_name(f"{path.name}.lock")
    blocking: list[str] = []
    source = f"recorded in {path}"
    # os.replace publishes the record whole, so reading it needs no lock.
    if path.exists():
        recorded = _read_record(path)
    else:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_SECONDS):
                if path.exists():
                    recorded = _read_record(path)
                else:
                    recorded, blocking = _detect_engine(db_dir, configured)
                    _write_record(path, recorded)
                    logger.info("Recorded KB engine %s in %s", recorded.value, path)
        except Timeout as exc:
            raise ConfigurationError(
                f"Timed out after {_LOCK_TIMEOUT_SECONDS}s waiting for {lock_path}, "
                "held by another process detecting the KB engine."
            ) from exc
        except OSError as exc:
            logger.warning("Cannot record the KB engine in %s: %s", path, exc)
            recorded, blocking = _detect_engine(db_dir, configured)
            source = f"detected because {path} cannot be written"
    if recorded is not configured:
        held = f" ({', '.join(blocking)} hold data)" if blocking else ""
        raise ConfigurationError(
            f"This deployment's KB engine is {recorded.value}{held}, {source}, "
            f"but {VECTOR_BACKEND_ENV} is {configured.value}. To change "
            "the engine of an empty deployment, delete the record file and restart."
        )
    return recorded
