"""The collection_metadata row is owner-neutral (#1567).

Against a real LanceDB table, because the hazard is what survives a write and
comes back out: `merge_insert(["name"])` merges on the name alone, so two
tenants owning a same-named collection share one row. A tenant-scoped
ingestion config -- which can carry an embedding_api_key and a model binding
that outranks the caller's own config -- must never land in it.

Until the enum TypeError in to_storage() was fixed, it was the only thing
stopping that write.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkStrategy,
    CollectionInfo,
    IngestionConfig,
    ParseMethod,
)


@pytest.fixture
def metadata_store(tmp_path):
    lancedb = pytest.importorskip("lancedb")
    from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
        LanceDBMetadataStore,
    )

    store = LanceDBMetadataStore()
    store._conn = lancedb.connect(str(tmp_path / "meta"))
    return store


def _tenant_config(model_id: str, api_key: str) -> IngestionConfig:
    return IngestionConfig(
        parse_method=ParseMethod.PYPDF,
        chunk_strategy=ChunkStrategy.MARKDOWN,
        embedding_model_id=model_id,
        embedding_api_key=api_key,
    )


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

    # Neither tenant's config reaches the shared row, so the resolver's
    # "use the model stored in ingestion_config" step cannot outrank either
    # caller's own config.
    assert readback.ingestion_config is None

    table = metadata_store._conn.open_table("collection_metadata")
    raw = json.dumps(table.to_arrow().to_pylist(), default=str)
    assert "sk-tenant-a" not in raw
    assert "sk-tenant-b" not in raw
    assert "model-a-from-config" not in raw
    assert "model-b-from-config" not in raw


def test_rebuild_style_save_keeps_a_binding_it_could_not_infer(metadata_store):
    """A save that infers no embedding model must not erase the stored one.

    ``_rebuild_collection_metadata_impl`` leaves both embedding fields None for
    a collection with no embeddings yet, and merge_insert updates every column,
    so passing them through would blank the binding. The enum TypeError used to
    abort that write before it landed.
    """
    bound = CollectionInfo(
        name="bound",
        embedding_model_id="text-embedding-v4",
        embedding_dimension=1024,
    )
    asyncio.run(metadata_store.save_collection(bound))

    # What the rebuild loop now hands to save_collection: the row as listed,
    # with nothing overwritten because nothing was inferred.
    listed = asyncio.run(metadata_store.get_collection("bound"))
    asyncio.run(metadata_store.save_collection(listed))

    readback = asyncio.run(metadata_store.get_collection("bound"))
    assert readback.embedding_model_id == "text-embedding-v4"
    assert readback.embedding_dimension == 1024
