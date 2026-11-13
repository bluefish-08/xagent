"""Read-only scope of an embedding migration: collections, documents, skips.

The precompute and switch commands both start here, and it runs on its own to
report what a migration would cover:

    python -m xagent.migrations.embedding.scope [--collection NAME ... | --team ID]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, List, Optional, Sequence

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SKIP_NO_OWNER = "no-owner"
SKIP_NO_SUCCESSFUL_INGEST = "no-successful-ingest"
SKIP_NO_PARSE_HASH = "no-parse-hash"
SKIP_OTHER_ENGINE = "other-engine"


class MigrationCliError(RuntimeError):
    """A scope or configuration problem that must stop the command."""


@dataclass(frozen=True)
class DocumentScope:
    """One ``documents`` row and, when it is usable, its latest parse hash."""

    doc_id: str
    user_id: Optional[int]
    source_path: Optional[str]
    parse_hash: Optional[str] = None
    skip_reason: Optional[str] = None


@dataclass
class CollectionScope:
    """What the migration would do with one collection, without doing it."""

    name: str
    engine: str = ""
    collection_skip: Optional[str] = None
    documents_total: int = 0
    eligible: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)


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
    erases the earlier success; those documents stay out of the map.
    """
    from xagent.core.tools.core.RAG_tools.core.schemas import (
        DocumentProcessingStatus,
    )

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


def iter_scope_documents(
    collection: str, *, store: Optional[Any] = None
) -> Iterator[DocumentScope]:
    """The collection's documents, each marked eligible or with its skip reason."""
    hashes = load_successful_parse_hashes(collection, store=store)
    for doc in iter_documents(collection, store=store):
        if doc.user_id is None:
            yield replace(doc, skip_reason=SKIP_NO_OWNER)
            continue
        if doc.doc_id not in hashes:
            yield replace(doc, skip_reason=SKIP_NO_SUCCESSFUL_INGEST)
            continue
        parse_hash = hashes[doc.doc_id]
        if not parse_hash:
            yield replace(doc, skip_reason=SKIP_NO_PARSE_HASH)
            continue
        yield replace(doc, parse_hash=parse_hash)


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


def scope_report(
    *, collections: Optional[Sequence[str]] = None, team: Optional[int] = None
) -> List[CollectionScope]:
    """Count what a migration would cover, per collection, changing nothing."""
    names = resolve_collection_names(collections=collections, team=team)
    report: List[CollectionScope] = []
    for name in names:
        item = CollectionScope(name=name, engine=collection_backend(name))
        if item.engine != "lancedb":
            item.collection_skip = SKIP_OTHER_ENGINE
            report.append(item)
            continue
        for doc in iter_scope_documents(name):
            item.documents_total += 1
            if doc.skip_reason:
                item.skipped[doc.skip_reason] = item.skipped.get(doc.skip_reason, 0) + 1
            else:
                item.eligible += 1
        report.append(item)
    return report


def add_scope_args(parser: argparse.ArgumentParser) -> None:
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--collection",
        action="append",
        metavar="NAME",
        help="Restrict to this collection; repeatable",
    )
    scope.add_argument(
        "--team",
        type=int,
        metavar="ID",
        help="Restrict to the knowledge bases of this team",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report what an embedding migration would cover.\n\n"
            "Read-only: lists the collections in scope, their engine, the\n"
            "documents eligible for the target model, and why the rest are\n"
            "skipped."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_scope_args(parser)
    args = parser.parse_args(argv)

    try:
        report = scope_report(collections=args.collection, team=args.team)
    except MigrationCliError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        logger.error("Scope report failed: %s", exc, exc_info=True)
        return 2

    if not report:
        logger.warning("No collections in scope; check --collection/--team")
        return 1
    for item in report:
        if item.collection_skip:
            logger.info("%s: skipped (%s engine)", item.name, item.engine)
            continue
        skipped = ", ".join(f"{k}={v}" for k, v in sorted(item.skipped.items()))
        logger.info(
            "%s: %d documents, %d eligible%s",
            item.name,
            item.documents_total,
            item.eligible,
            f", skipped ({skipped})" if skipped else "",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
