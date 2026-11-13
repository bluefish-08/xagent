"""Tests for the embedding migration's read-only scope."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from xagent.migrations.embedding import scope as mod
from xagent.migrations.embedding.scope import DocumentScope, MigrationCliError


class _Batch:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    def to_pylist(self) -> List[Dict[str, Any]]:
        return self._rows


class _Store:
    def __init__(
        self,
        batches: List[List[Dict[str, Any]]],
        status: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._batches = batches
        self._status = status or []
        self.calls: List[Dict[str, Any]] = []
        self.status_calls: List[Dict[str, Any]] = []

    def iter_batches(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        for rows in self._batches:
            yield _Batch(rows)

    def load_ingestion_status(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.status_calls.append(kwargs)
        return self._status


def test_documents_are_streamed_across_batches() -> None:
    store = _Store(
        [
            [{"doc_id": f"doc-{index}", "user_id": 1} for index in range(150)],
            [{"doc_id": "doc-150", "user_id": 2}],
        ]
    )

    docs = list(mod.iter_documents("kb", store=store))

    assert len(docs) == 151
    assert store.calls == [
        {
            "table_name": "documents",
            "columns": ["doc_id", "user_id", "source_path"],
            "batch_size": 1000,
            "filters": {"collection": "kb"},
            "user_id": None,
            "is_admin": True,
        }
    ]


def test_documents_without_a_doc_id_or_with_a_bad_owner_are_handled() -> None:
    store = _Store(
        [
            [
                {"doc_id": "", "user_id": 1},
                {"doc_id": "doc-1", "user_id": "not-a-number"},
                {"doc_id": "doc-2", "user_id": None},
                {"doc_id": "doc-3", "user_id": 7},
            ]
        ]
    )

    docs = list(mod.iter_documents("kb", store=store))

    assert [(doc.doc_id, doc.user_id) for doc in docs] == [
        ("doc-1", None),
        ("doc-2", None),
        ("doc-3", 7),
    ]


def test_only_successful_runs_contribute_a_parse_hash() -> None:
    store = _Store(
        [],
        [
            {"doc_id": "doc-1", "status": "success", "parse_hash": "hash-1"},
            {"doc_id": "doc-2", "status": "failed", "parse_hash": "hash-2"},
            {"doc_id": "doc-3", "status": "success", "parse_hash": ""},
            {"doc_id": "", "status": "success", "parse_hash": "hash-4"},
        ],
    )

    hashes = mod.load_successful_parse_hashes("kb", store=store)

    assert hashes == {"doc-1": "hash-1", "doc-3": None}
    assert store.status_calls == [
        {"collection": "kb", "user_id": None, "is_admin": True}
    ]


def test_each_document_gets_its_hash_or_exactly_one_skip_reason() -> None:
    store = _Store(
        [
            [
                {"doc_id": "ownerless", "user_id": None},
                {"doc_id": "never-ok", "user_id": 1},
                {"doc_id": "no-hash", "user_id": 1},
                {"doc_id": "ok", "user_id": 1, "source_path": "/a.pdf"},
            ]
        ],
        [
            {"doc_id": "ownerless", "status": "success", "parse_hash": "h0"},
            {"doc_id": "never-ok", "status": "failed", "parse_hash": "h1"},
            {"doc_id": "no-hash", "status": "success", "parse_hash": None},
            {"doc_id": "ok", "status": "success", "parse_hash": "h3"},
        ],
    )

    docs = {doc.doc_id: doc for doc in mod.iter_scope_documents("kb", store=store)}

    assert docs["ownerless"].skip_reason == mod.SKIP_NO_OWNER
    assert docs["ownerless"].parse_hash is None
    assert docs["never-ok"].skip_reason == mod.SKIP_NO_SUCCESSFUL_INGEST
    assert docs["no-hash"].skip_reason == mod.SKIP_NO_PARSE_HASH
    assert docs["ok"] == DocumentScope("ok", 1, "/a.pdf", parse_hash="h3")


def test_explicit_collections_are_trimmed_and_deduplicated() -> None:
    assert mod.resolve_collection_names(collections=[" kb ", "kb", "other"]) == [
        "kb",
        "other",
    ]


def test_collection_and_team_are_mutually_exclusive() -> None:
    with pytest.raises(MigrationCliError, match="not both"):
        mod.resolve_collection_names(collections=["kb"], team=3)


def test_an_empty_collection_name_is_rejected() -> None:
    with pytest.raises(MigrationCliError, match="must not be empty"):
        mod.resolve_collection_names(collections=["  "])


def test_all_collections_are_listed_as_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    result = SimpleNamespace(
        collections=[
            SimpleNamespace(name="b"),
            SimpleNamespace(name="a"),
            SimpleNamespace(name=""),
        ]
    )

    async def _list(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return result

    from xagent.core.tools.core.RAG_tools.management import (
        collections as collections_mod,
    )

    monkeypatch.setattr(collections_mod, "list_collections", _list)

    assert mod.resolve_collection_names() == ["a", "b"]
    assert calls == [{"user_id": None, "is_admin": True}]


def test_team_scope_requires_the_application_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import knowledge_base_team_scope as team_scope

    monkeypatch.setattr(team_scope, "team_knowledge_base_hook_installed", lambda: False)

    with pytest.raises(MigrationCliError, match="--collection"):
        mod.resolve_collection_names(team=7)


def test_team_scope_maps_hook_answers_to_collection_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.models import database
    from xagent.web.services import knowledge_base_team_scope as team_scope
    from xagent.web.services.knowledge_base_team_scope import KnowledgeBaseAccess

    seen: Dict[str, Any] = {}

    def _configure_db(*args: Any, **kwargs: Any) -> None:
        seen["configure_db"] = kwargs

    class _Session:
        def close(self) -> None:
            seen["closed"] = True

    def _team_knowledge_bases(
        session: Any, *, team_id: int
    ) -> List[KnowledgeBaseAccess]:
        seen["team_id"] = team_id
        return [
            KnowledgeBaseAccess(name="b", storage_user_id=11),
            KnowledgeBaseAccess(name="a", storage_user_id=12),
        ]

    monkeypatch.setattr(team_scope, "team_knowledge_base_hook_installed", lambda: True)
    monkeypatch.setattr(database, "configure_db", _configure_db)
    monkeypatch.setattr(database, "get_session_local", lambda: _Session)
    monkeypatch.setattr(team_scope, "team_knowledge_bases", _team_knowledge_bases)

    assert mod.resolve_collection_names(team=7) == ["a", "b"]
    assert seen["team_id"] == 7
    assert seen["configure_db"] == {"read_only": True}
    assert seen["closed"] is True


def _patch_backend(monkeypatch: pytest.MonkeyPatch, lookup: Any) -> Dict[str, Any]:
    from xagent.core.tools.core.RAG_tools.kb import coordinator as coordinator_mod
    from xagent.core.tools.core.RAG_tools.management import (
        collection_manager as collection_manager_mod,
    )
    from xagent.core.tools.core.RAG_tools.storage.vector_backend import (
        KBStorageBackend,
    )

    seen: Dict[str, Any] = {}

    def _resolve(collection_info: object | None) -> Any:
        seen["info"] = collection_info
        return KBStorageBackend.MILVUS if collection_info else KBStorageBackend.LANCEDB

    monkeypatch.setattr(collection_manager_mod, "get_collection_sync", lookup)
    monkeypatch.setattr(coordinator_mod, "resolve_collection_backend", _resolve)
    return seen


def test_collection_backend_reads_the_stored_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = SimpleNamespace(extra_metadata={"kb_storage": {"backend": "milvus"}})
    seen = _patch_backend(monkeypatch, lambda name: info)

    assert mod.collection_backend("kb") == "milvus"
    assert seen["info"] is info


def test_collection_backend_defaults_for_unknown_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(name: str) -> Any:
        raise ValueError("Collection not found")

    seen = _patch_backend(monkeypatch, _missing)

    assert mod.collection_backend("kb") == "lancedb"
    assert seen["info"] is None


def test_the_public_backend_resolver_delegates_to_the_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.kb import coordinator as coordinator_mod

    seen: List[object] = []
    fake = SimpleNamespace(_resolve_backend=lambda info: seen.append(info) or "x")
    monkeypatch.setattr(coordinator_mod, "get_kb_coordinator", lambda: fake)

    assert coordinator_mod.resolve_collection_backend("info") == "x"
    assert seen == ["info"]


def _wire_report(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {
        "kb": [
            DocumentScope("a", 1, None, parse_hash="h"),
            DocumentScope("b", None, None, skip_reason=mod.SKIP_NO_OWNER),
            DocumentScope("c", None, None, skip_reason=mod.SKIP_NO_OWNER),
            DocumentScope("d", 1, None, skip_reason=mod.SKIP_NO_PARSE_HASH),
        ],
    }
    monkeypatch.setattr(
        mod, "resolve_collection_names", lambda **kwargs: ["milvus-kb", "kb"]
    )
    monkeypatch.setattr(
        mod,
        "collection_backend",
        lambda name: "milvus" if name == "milvus-kb" else "lancedb",
    )

    def _docs(name: str, **kwargs: Any) -> Any:
        assert name != "milvus-kb", "another engine's documents must not be read"
        return iter(docs[name])

    monkeypatch.setattr(mod, "iter_scope_documents", _docs)


def test_the_report_counts_eligible_and_skipped_documents_per_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_report(monkeypatch)

    report = mod.scope_report()

    other, kb = report
    assert other.collection_skip == mod.SKIP_OTHER_ENGINE
    assert other.engine == "milvus"
    assert other.documents_total == 0
    assert kb.engine == "lancedb"
    assert kb.collection_skip is None
    assert kb.documents_total == 4
    assert kb.eligible == 1
    assert kb.skipped == {mod.SKIP_NO_OWNER: 2, mod.SKIP_NO_PARSE_HASH: 1}


def test_main_prints_the_report_and_returns_0(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _wire_report(monkeypatch)

    with caplog.at_level("INFO", logger=mod.__name__):
        assert mod.main([]) == 0

    assert "milvus-kb: skipped (milvus engine)" in caplog.text
    assert "kb: 4 documents, 1 eligible, skipped (no-owner=2, no-parse-hash=1)" in (
        caplog.text
    )


def test_main_returns_1_when_no_collection_is_in_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "scope_report", lambda **kwargs: [])

    assert mod.main([]) == 1


def test_main_returns_2_when_the_scope_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(**kwargs: Any) -> Any:
        raise MigrationCliError("no hook")

    monkeypatch.setattr(mod, "scope_report", _fail)

    assert mod.main(["--team", "3"]) == 2


def test_main_forwards_the_scope_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: List[Dict[str, Any]] = []
    monkeypatch.setattr(mod, "scope_report", lambda **kwargs: seen.append(kwargs) or [])

    mod.main(["--collection", "a", "--collection", "b"])
    mod.main(["--team", "4"])

    assert seen == [
        {"collections": ["a", "b"], "team": None},
        {"collections": None, "team": 4},
    ]


def test_main_rejects_collection_with_team() -> None:
    with pytest.raises(SystemExit) as exc:
        mod.main(["--collection", "a", "--team", "4"])
    assert exc.value.code == 2
