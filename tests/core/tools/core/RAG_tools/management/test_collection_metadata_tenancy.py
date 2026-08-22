"""The collection_metadata row is owner-neutral (#1567).

Against a real LanceDB directory, because the hazard is what survives a write
and comes back out: ``merge_insert(["name"])`` merges on the name alone, so two
tenants owning a same-named collection share one row. A tenant-scoped ingestion
config -- which can carry an embedding_api_key and a model binding that would
outrank the caller's own config -- must never land in it, and must not be
hydrated back out of it either.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkStrategy,
    CollectionInfo,
    IngestionConfig,
    ListCollectionsResult,
    ParseMethod,
)


@pytest.fixture
def metadata_store(tmp_path, monkeypatch) -> Iterator[Any]:
    pytest.importorskip("lancedb")
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
    )
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        reset_locks_for_testing,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_metadata_store,
        reset_rag_storage_for_tests,
    )

    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / ".lancedb"))
    reset_rag_storage_for_tests()
    reset_locks_for_testing()
    store = get_metadata_store()
    try:
        yield store
    finally:
        # LanceDB holds file handles per open table/connection; a test that
        # leaks them keeps the tmp dir mapped for the rest of the session.
        _safe_close_table(getattr(store, "_conn", None))
        reset_rag_storage_for_tests()
        reset_locks_for_testing()


def _tenant_config(model_id: str, api_key: str) -> IngestionConfig:
    return IngestionConfig(
        parse_method=ParseMethod.PYPDF,
        chunk_strategy=ChunkStrategy.MARKDOWN,
        embedding_model_id=model_id,
        embedding_api_key=api_key,
    )


def _raw_metadata_rows(metadata_store) -> str:
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
    )

    table = None
    try:
        table = metadata_store.get_raw_connection().open_table("collection_metadata")
        return json.dumps(table.to_arrow().to_pylist(), default=str)
    finally:
        _safe_close_table(table)


def test_same_named_tenants_share_a_row_without_sharing_config(metadata_store):
    """Two tenants, one collection name, no config or credential in the row."""
    tenant_a = CollectionInfo(
        name="shared-name",
        embedding_model_id="model-a",
        ingestion_config=_tenant_config("model-a-from-config", "sk-tenant-a"),
    )
    tenant_b = CollectionInfo(
        name="shared-name",
        embedding_model_id="model-b",
        ingestion_config=_tenant_config("model-b-from-config", "sk-tenant-b"),
    )

    asyncio.run(metadata_store.save_collection(tenant_a))
    asyncio.run(metadata_store.save_collection(tenant_b))

    readback = asyncio.run(metadata_store.get_collection("shared-name"))
    assert readback.ingestion_config is None

    raw = _raw_metadata_rows(metadata_store)
    assert "sk-tenant-a" not in raw
    assert "sk-tenant-b" not in raw
    assert "model-a-from-config" not in raw
    assert "model-b-from-config" not in raw


def test_rebuild_style_save_keeps_a_binding_it_could_not_infer(metadata_store):
    """A rebuild that infers no embedding model must not erase the stored one."""
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    bound = CollectionInfo(
        name="bound",
        embedding_model_id="text-embedding-v4",
        embedding_dimension=1024,
    )
    asyncio.run(metadata_store.save_collection(bound))

    # What the rebuild holds when inference yielded nothing: stats, no binding.
    inferred_nothing = CollectionInfo(name="bound", documents=3)
    asyncio.run(
        cm.collection_manager.save_collection_fields(inferred_nothing, ("documents",))
    )

    readback = asyncio.run(metadata_store.get_collection("bound"))
    assert readback.documents == 3
    assert readback.embedding_model_id == "text-embedding-v4"
    assert readback.embedding_dimension == 1024


def test_unknown_owned_fields_are_rejected(metadata_store):
    """A typo would otherwise be an AttributeError the rebuild loop swallows."""
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    with pytest.raises(ValueError, match="not_a_field"):
        asyncio.run(
            cm.collection_manager.save_collection_fields(
                CollectionInfo(name="typo"), ("not_a_field",)
            )
        )


def _seed_two_tenant_configs(metadata_store, name: str) -> None:
    """Two collection_config rows for one name; tenant B is the latest."""
    older = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    asyncio.run(
        metadata_store.save_collection_config(
            name,
            _tenant_config("model-a-from-config", "sk-tenant-a").model_dump_json(),
            101,
        )
    )
    asyncio.run(
        metadata_store.save_collection_config(
            name,
            _tenant_config("model-b-from-config", "sk-tenant-b").model_dump_json(),
            202,
        )
    )
    # Age tenant A so admin's latest-row pick is deterministically tenant B.
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
    )

    table = None
    try:
        table = metadata_store.get_raw_connection().open_table("collection_config")
        table.update("user_id = 101", {"updated_at": older})
    finally:
        _safe_close_table(table)


def test_rebuild_does_not_leak_a_tenant_config_into_the_shared_row(metadata_store):
    """Full chain: two tenant configs -> admin rebuild -> production readback.

    The rebuild lists collections as admin, which attaches the latest tenant's
    ingestion config to the in-memory CollectionInfo it then saves. Nothing of
    that config may reach the shared row, and the resolver must fall through to
    the caller's own model rather than the tenant's.
    """
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    name = "shared-name"
    asyncio.run(metadata_store.save_collection(CollectionInfo(name=name)))
    _seed_two_tenant_configs(metadata_store, name)

    from xagent.core.tools.core.RAG_tools.management import collections

    listed = asyncio.run(
        collections.list_collections(is_admin=True, force_realtime=True)
    )
    attached = next(c for c in listed.collections if c.name == name)
    # Precondition: the rebuild really does hold a tenant's config in memory.
    assert attached.ingestion_config is not None
    assert attached.ingestion_config.embedding_api_key == "sk-tenant-b"

    asyncio.run(cm.rebuild_collection_metadata())

    readback = asyncio.run(metadata_store.get_collection(name))
    assert readback.ingestion_config is None
    assert readback.embedding_model_id is None

    # The rebuild's no-stored-row branch inserts the listed object whole, and is
    # the only place a config-carrying CollectionInfo reaches storage as a full
    # row; the merge branch above never passes the column along at all.
    asyncio.run(metadata_store.delete_collection(name))
    asyncio.run(cm.collection_manager.save_collection_fields(attached, ("documents",)))
    assert asyncio.run(metadata_store.get_collection(name)).ingestion_config is None

    raw = _raw_metadata_rows(metadata_store)
    for secret in (
        "sk-tenant-a",
        "sk-tenant-b",
        "model-a-from-config",
        "model-b-from-config",
    ):
        assert secret not in raw

    # Resolver priority: with nothing bound on the shared row and no tenant
    # config to hydrate, the caller's own model is what wins.
    assert (
        cm.resolve_effective_embedding_model_sync(name, config_model_id="caller-model")
        == "caller-model"
    )


def test_rebuild_only_writes_the_fields_it_recomputed(metadata_store):
    """A stale pre-loop snapshot must not roll back another writer's columns.

    The rebuild reads every collection once, then saves them one by one; the
    storage merge overwrites every column, so anything a concurrent writer
    changed in between would be reverted to the snapshot.
    """
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    name = "raced"
    snapshot = CollectionInfo(name=name, documents=7, chunks=21)
    asyncio.run(metadata_store.save_collection(snapshot))

    # What the concurrent writer landed after the rebuild's snapshot read.
    concurrent = CollectionInfo(
        name=name,
        documents=1,
        chunks=1,
        collection_locked=True,
        extra_metadata={"pinned": "by-other-writer"},
    )
    asyncio.run(metadata_store.save_collection(concurrent))

    async def stale_listing(**kwargs):
        return ListCollectionsResult(
            status="success",
            collections=[snapshot],
            total_count=1,
            message="stale snapshot",
        )

    with patch(
        "xagent.core.tools.core.RAG_tools.management.collections.list_collections",
        stale_listing,
    ):
        asyncio.run(cm.rebuild_collection_metadata())

    readback = asyncio.run(metadata_store.get_collection(name))
    assert readback.documents == 7
    assert readback.chunks == 21
    assert readback.collection_locked is True
    assert readback.extra_metadata == {"pinned": "by-other-writer"}


def test_rebuild_skips_the_write_when_nothing_changed(metadata_store):
    """A no-op rebuild must not touch the row at all, not even updated_at."""
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    name = "unchanged"
    asyncio.run(metadata_store.save_collection(CollectionInfo(name=name, documents=3)))
    before = asyncio.run(metadata_store.get_collection(name))

    async def same_listing(**kwargs):
        return ListCollectionsResult(
            status="success",
            collections=[before],
            total_count=1,
            message="unchanged",
        )

    with patch(
        "xagent.core.tools.core.RAG_tools.management.collections.list_collections",
        same_listing,
    ):
        asyncio.run(cm.rebuild_collection_metadata())

    after = asyncio.run(metadata_store.get_collection(name))
    assert after.updated_at == before.updated_at


def test_a_written_row_carries_the_migrated_schema_version(metadata_store):
    """schema_version rides along on any write without being an owned field.

    The re-read migrates in memory, so the merged row already holds the current
    version; owning the field would change nothing, and would not rewrite a
    legacy row whose recomputed stats happen to match either.
    """
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
    )
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    name = "legacy-version"
    asyncio.run(metadata_store.save_collection(CollectionInfo(name=name, documents=3)))

    table = None
    try:
        table = metadata_store.get_raw_connection().open_table("collection_metadata")
        table.update(f"name = '{name}'", {"schema_version": "0.0.0"})
    finally:
        _safe_close_table(table)

    migrated = asyncio.run(metadata_store.get_collection(name))
    assert migrated.schema_version == "1.0.0"

    recomputed = migrated.model_copy(update={"documents": 4})
    asyncio.run(
        cm.collection_manager.save_collection_fields(
            recomputed, cm._REBUILD_OWNED_FIELDS
        )
    )

    raw = json.loads(_raw_metadata_rows(metadata_store))
    row = next(row for row in raw if row["name"] == name)
    assert row["schema_version"] == "1.0.0"


def test_a_brand_new_collection_is_inserted_whole(metadata_store):
    """Nothing stored yet -- not even the table -- so the whole row goes in."""
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    fresh = CollectionInfo(name="brand-new", documents=4)
    asyncio.run(cm.collection_manager.save_collection_fields(fresh, ("documents",)))

    assert asyncio.run(metadata_store.get_collection("brand-new")).documents == 4


def test_a_read_failure_does_not_become_a_full_row_overwrite(metadata_store):
    """Only "not found" may reach the insert-whole path.

    Treating any read failure as "no row" would let one flaky read turn a stale
    caller snapshot into a full-row overwrite, silently reinstating exactly the
    concurrent rollback the field merge exists to prevent.
    """
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    name = "flaky"
    asyncio.run(
        metadata_store.save_collection(
            CollectionInfo(name=name, documents=9, collection_locked=True)
        )
    )
    stale = CollectionInfo(name=name, documents=1)

    async def unavailable(_self, _name):
        raise RuntimeError("table under compaction")

    with patch.object(type(metadata_store), "get_collection", unavailable):
        with pytest.raises(RuntimeError):
            asyncio.run(
                cm.collection_manager.save_collection_fields(stale, ("documents",))
            )

    survived = asyncio.run(metadata_store.get_collection(name))
    assert survived.documents == 9
    assert survived.collection_locked is True


def test_a_corrupt_row_is_not_mistaken_for_a_missing_one(metadata_store):
    """A row that exists but will not parse is a read failure, not "no row".

    Both raise from the same call and json/pydantic errors are ValueError
    subclasses, so only a dedicated not-found error can tell them apart -- and
    getting it wrong turns a stale snapshot into a full-row overwrite.
    """
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        _safe_close_table,
    )
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    name = "corrupt"
    asyncio.run(
        metadata_store.save_collection(
            CollectionInfo(name=name, documents=9, collection_locked=True)
        )
    )

    table = None
    try:
        table = metadata_store.get_raw_connection().open_table("collection_metadata")
        table.update(f"name = '{name}'", {"extra_metadata": "{not json"})
    finally:
        _safe_close_table(table)

    stale = CollectionInfo(name=name, documents=1)
    with pytest.raises(ValueError):
        asyncio.run(cm.collection_manager.save_collection_fields(stale, ("documents",)))

    raw = json.loads(_raw_metadata_rows(metadata_store))
    survived = next(row for row in raw if row["name"] == name)
    assert survived["documents"] == 9
    assert survived["collection_locked"] is True
