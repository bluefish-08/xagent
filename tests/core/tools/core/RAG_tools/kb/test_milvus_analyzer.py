"""Milvus keyword search across scripts, on the server at ``MILVUS_URI``.

The analyzer is fixed when the collection is created, so these tests cover the
one ``ensure_milvus_collection`` installs and the one old collections keep.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    KBCollectionHandle,
    KBHandleProvider,
    ensure_milvus_collection,
    milvus_collection_name,
)
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_main_pointer_store,
    get_metadata_store,
    get_vector_index_store,
)

pytestmark = pytest.mark.milvus

PARSE = "ph-1"
COLLECTION = "kb"
DIM = 3
ONE = [1.0, 0.0, 0.0]


@pytest.fixture
def client() -> Any:
    from pymilvus import MilvusClient

    return MilvusClient(uri=os.environ["MILVUS_URI"])


@pytest.fixture
def model(client: Any) -> Iterator[str]:
    model = f"analyzer-{uuid.uuid4().hex[:12]}"
    yield model
    client.drop_collection(milvus_collection_name(model))


@pytest.fixture(autouse=True)
def milvus_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", "milvus")


def _open() -> KBCollectionHandle:
    return KBHandleProvider().open(
        KBCollectionContext(
            collection=COLLECTION,
            user_scope=KBUserScope(user_id=None, is_admin=True),
            access_mode=KBAccessMode.WRITE,
            allow_create=True,
            hide_missing=True,
            metadata_store=get_metadata_store(),
            vector_index_store=get_vector_index_store(),
            ingestion_status_store=get_ingestion_status_store(),
            main_pointer_store=get_main_pointer_store(),
            backend=KBStorageBackend.MILVUS,
            capabilities=KBBackendCapabilities.milvus(),
        )
    )


Chunk = tuple[str, str, list[float], dict[str, Any]]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return chunk_id, text, ONE, {}


def _ingest(
    handle: KBCollectionHandle, model: str, doc_id: str, chunks: list[Chunk]
) -> None:
    now = datetime.now(timezone.utc)
    handle.write_chunks(
        doc_id,
        PARSE,
        "cfg",
        {},
        [
            {
                "chunk_id": chunk_id,
                "index": index,
                "text": text,
                "created_at": now,
                "metadata": metadata,
            }
            for index, (chunk_id, text, _, metadata) in enumerate(chunks)
        ],
        user_id=1,
    )
    vectors = {chunk_id: vector for chunk_id, _, vector, _ in chunks}
    pending = handle.read_chunks_needing_embedding(
        doc_id, PARSE, model, user_id=1, is_admin=False
    ).chunks
    handle.write_embeddings(
        [
            ChunkEmbeddingData(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                parse_hash=chunk.parse_hash,
                model=model,
                vector=vectors[chunk.chunk_id],
                text=chunk.text,
                chunk_hash=chunk.chunk_hash,
                metadata=chunk.metadata,
            )
            for chunk in pending
        ],
        user_id=1,
    )
    handle.commit_embeddings(doc_id, PARSE, model, user_id=1, is_admin=False)


def _sparse(handle: KBCollectionHandle, model: str, query: str) -> Any:
    return handle.search_sparse(model, query, top_k=5, user_id=1, is_admin=False)


def _found(response: Any) -> set[str]:
    assert response.status == "success", response.warnings
    return {result.chunk_id for result in response.results}


def test_keyword_search_is_case_insensitive(model: str) -> None:
    handle = _open()
    _ingest(handle, model, "d", [_chunk("c0", "Milvus 2.6 向量数据库")])

    for query in ("milvus", "MILVUS", "Milvus"):
        response = _sparse(handle, model, query)
        assert _found(response) == {"c0"}, query
        assert response.warnings == [], query


@pytest.mark.parametrize(
    ("text", "query"),
    [
        ("ＭｉｌｖｕｓＤＢ ２０２６", "ＭｉｌｖｕｓＤＢ"),
        ("こんにちは世界", "こんにちは"),
        ("안녕하세요 세계", "안녕하세요"),
    ],
)
def test_keyword_search_keeps_full_width_kana_and_hangul_tokens(
    text: str, query: str, model: str
) -> None:
    handle = _open()
    _ingest(handle, model, "d", [_chunk("c0", text)])

    response = _sparse(handle, model, query)

    assert _found(response) == {"c0"}
    assert response.warnings == []


def test_full_width_text_is_matched_one_character_at_a_time(model: str) -> None:
    """Document character-level matching of full-width text, not a feature to extend."""
    handle = _open()
    _ingest(handle, model, "d", [_chunk("c0", "ＡＢＣ"), _chunk("c1", "ＣＢＡ")])

    response = _sparse(handle, model, "ＡＢＣ")

    assert _found(response) == {"c0", "c1"}
    assert response.warnings == []


def test_chinese_multi_word_queries_still_match(model: str) -> None:
    handle = _open()
    _ingest(handle, model, "d", [_chunk("c0", "知识库检索与向量数据库")])

    for query in ("向量数据库", "知识库"):
        response = _sparse(handle, model, query)
        assert "c0" in _found(response), query
        assert response.warnings == [], query


def test_punctuation_only_queries_match_no_bm25_tokens(client: Any, model: str) -> None:
    handle = _open()
    _ingest(handle, model, "d", [_chunk("c0", "hello ， world!! 🎉 结束")])
    name = milvus_collection_name(model)

    for query in ("，", "!!"):
        hits = client.search(
            name,
            data=[query],
            anns_field="sparse",
            limit=5,
            search_params={"metric_type": "BM25"},
            output_fields=["chunk_id"],
            consistency_level="Strong",
        )[0]
        assert hits == [], query


@pytest.mark.parametrize(
    ("text", "queries"),
    [
        ("Milvus2.6 release", ("Milvus2.6", "milvus2.6")),
        ("v2.6.25 发布", ("v2.6",)),
        ("version 2.6 notes", ("2.6",)),
    ],
)
def test_mixed_alphanumeric_tokens_stay_whole(
    text: str, queries: tuple[str, ...], model: str
) -> None:
    handle = _open()
    _ingest(handle, model, "d", [_chunk("c0", text)])

    for query in queries:
        response = _sparse(handle, model, query)
        assert _found(response) == {"c0"}, query
        assert response.warnings == [], query


def test_whitespace_is_not_indexed_as_a_token(model: str) -> None:
    handle = _open()
    _ingest(
        handle,
        model,
        "d",
        [_chunk("c0", "alpha beta"), _chunk("c1", "gamma delta")],
    )

    response = _sparse(handle, model, "zzz qqq")

    assert _found(response) == set()
    assert response.warnings == []


def test_a_punctuation_only_query_matches_no_bm25_token(model: str) -> None:
    handle = _open()
    _ingest(handle, model, "d", [_chunk("c0", "全角逗号，与感叹号！！")])

    response = _sparse(handle, model, "，！！")

    assert _found(response) == set()
    assert response.warnings == []


def test_an_existing_collection_of_the_old_analyzer_is_kept(
    client: Any, model: str
) -> None:
    name = _create_old_collection(client, model)
    assert ensure_milvus_collection(client, model, DIM) == name
    fields = {
        field["name"]: field for field in client.describe_collection(name)["fields"]
    }
    assert fields["text"]["params"]["analyzer_params"] == '{"type":"chinese"}'

    handle = _open()
    _ingest(
        handle,
        model,
        "d",
        [
            _chunk("c0", "Milvus 2.6 向量数据库"),
            _chunk("f0", "ＭｉｌｖｕｓＤＢ ２０２６"),
        ],
    )
    lower = _sparse(handle, model, "milvus")
    upper = _sparse(handle, model, "Milvus")
    full = _sparse(handle, model, "ＭｉｌｖｕｓＤＢ")

    assert (lower.results, lower.warnings) == ([], [])
    assert (_found(upper), upper.warnings) == ({"c0"}, [])
    assert (_found(full), [w.code for w in full.warnings]) == (
        {"f0"},
        ["FTS_FALLBACK"],
    )


def _create_old_collection(client: Any, model: str) -> str:
    """Build the pre-change collection directly, before the new analyzer existed."""
    from pymilvus import DataType, Function, FunctionType

    name = milvus_collection_name(model)
    schema = client.create_schema(auto_id=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field("kb_id", DataType.VARCHAR, max_length=512, is_partition_key=True)
    schema.add_field("user_id", DataType.INT64, nullable=True)
    for field in ("doc_id", "parse_hash", "config_hash"):
        schema.add_field(field, DataType.VARCHAR, max_length=512)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=65_535,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("visible", DataType.BOOL)
    schema.add_field("created_at", DataType.INT64)
    schema.add_field("metadata", DataType.JSON)
    schema.add_function(
        Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )
    )
    client.create_collection(name, schema=schema, properties={"xagent.model_id": model})
    indexes = client.prepare_index_params()
    indexes.add_index(
        "dense",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    indexes.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    for field in ("doc_id", "user_id"):
        indexes.add_index(field, index_type="INVERTED")
    client.create_index(name, indexes)
    client.load_collection(name)
    return name
