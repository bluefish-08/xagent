"""Scope, document enumeration and pacing shared by the embedding-migration commands."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from xagent.core.model.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)


class MigrationCliError(RuntimeError):
    """A scope or configuration problem that must stop the command."""


@dataclass(frozen=True)
class DocumentScope:
    """One ``documents`` row the migration may embed."""

    doc_id: str
    user_id: Optional[int]
    source_path: Optional[str]


def _owner(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def iter_documents(
    collection: str, *, store: Optional[Any] = None
) -> Iterator[DocumentScope]:
    """Yield every ``documents`` row of ``collection``, across all owners.

    ``CollectionHandle.list_documents`` stops at 100 rows, so the migration
    streams the table in batches instead.
    """
    if store is None:
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_vector_index_store,
        )

        store = get_vector_index_store()
    for batch in store.iter_batches(
        table_name="documents",
        columns=["doc_id", "user_id", "source_path"],
        batch_size=1000,
        filters={"collection": collection},
        user_id=None,
        is_admin=True,
    ):
        for row in batch.to_pylist():
            doc_id = row.get("doc_id")
            if not doc_id:
                continue
            yield DocumentScope(
                doc_id=str(doc_id),
                user_id=_owner(row.get("user_id")),
                source_path=row.get("source_path"),
            )


def load_successful_parse_hashes(
    collection: str, *, store: Optional[Any] = None
) -> Dict[str, Optional[str]]:
    """Map doc_id to the parse_hash of its latest successful ingestion.

    A key present with an empty value means the run succeeded without recording
    a hash. ``ingestion_runs`` holds one row per document, so a later failure
    erases the earlier success; those documents stay out of the map and are
    reported instead of being embedded from an older parse.
    """
    from xagent.core.tools.core.RAG_tools.core.schemas import DocumentProcessingStatus

    if store is None:
        from xagent.core.tools.core.RAG_tools.management.status import (
            load_ingestion_status,
        )

        rows = load_ingestion_status(collection=collection, is_admin=True)
    else:
        rows = store.load_ingestion_status(
            collection=collection, user_id=None, is_admin=True
        )
    hashes: Dict[str, Optional[str]] = {}
    for row in rows:
        if str(row.get("status") or "") != DocumentProcessingStatus.SUCCESS.value:
            continue
        doc_id = row.get("doc_id")
        if not doc_id:
            continue
        parse_hash = row.get("parse_hash")
        hashes[str(doc_id)] = str(parse_hash) if parse_hash else None
    return hashes


def resolve_collection_names(
    *, collections: Optional[Sequence[str]] = None, team: Optional[int] = None
) -> List[str]:
    """The collections this run covers, in the order they are processed."""
    if collections and team is not None:
        raise MigrationCliError("pass --collection or --team, not both")
    if collections:
        names: List[str] = []
        for raw in collections:
            name = raw.strip()
            if not name:
                raise MigrationCliError("--collection must not be empty")
            if name not in names:
                names.append(name)
        return names
    if team is not None:
        return _team_collection_names(team)
    return _all_collection_names()


def collection_backend(collection: str) -> str:
    """The engine a collection is bound to; the deployment's when unbound."""
    from xagent.core.tools.core.RAG_tools.kb.coordinator import (
        resolve_collection_backend,
    )
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        get_collection_sync,
    )

    try:
        info: object | None = get_collection_sync(collection)
    except ValueError:
        info = None
    return str(resolve_collection_backend(info).value)


def _all_collection_names() -> List[str]:
    from xagent.core.tools.core.RAG_tools.management.collections import (
        list_collections,
    )

    result = asyncio.run(list_collections(user_id=None, is_admin=True))
    return sorted({info.name for info in result.collections if info.name})


def _team_collection_names(team_id: int) -> List[str]:
    # Teams live outside this package: the application installs the hook that
    # maps a team to its knowledge bases, and the command cannot invent one.
    from xagent.web.models.database import configure_db, get_session_local
    from xagent.web.services.knowledge_base_team_scope import (
        team_knowledge_base_hook_installed,
        team_knowledge_bases,
    )

    if not team_knowledge_base_hook_installed():
        raise MigrationCliError(
            "--team needs an application-installed team visibility hook; "
            "pass --collection NAME instead"
        )
    configure_db(read_only=True)
    session = get_session_local()()
    try:
        access = team_knowledge_bases(session, team_id=int(team_id))
    finally:
        session.close()
    return sorted({item.name for item in access if item.name})


class RateLimiter:
    """Spaces embedding calls so a run never exceeds ``rate`` items per second."""

    def __init__(self, rate: float) -> None:
        self._rate = float(rate)
        self._next_start = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, count: int) -> None:
        if self._rate <= 0 or count <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + count / self._rate
        wait = start - now
        if wait > 0:
            time.sleep(wait)


class PacedEmbedding(BaseEmbedding):
    """Spends rate-limit budget for every encode call the stage makes."""

    def __init__(self, inner: BaseEmbedding, limiter: RateLimiter) -> None:
        self._inner = inner
        self._limiter = limiter

    def encode(
        self,
        text: Union[str, List[str]],
        dimension: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        self._limiter.acquire(1 if isinstance(text, str) else len(text))
        return self._inner.encode(text, dimension=dimension, instruct=instruct)

    def get_dimension(self) -> Optional[int]:
        return self._inner.get_dimension()

    @property
    def abilities(self) -> List[str]:
        return self._inner.abilities
