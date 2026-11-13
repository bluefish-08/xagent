"""Tests for the embedding migration's phase-1 ``embed`` command."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import EmbeddingReadResponse
from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
    EmbeddingStageResult,
)
from xagent.migrations.embedding import common
from xagent.migrations.embedding import embed as mod
from xagent.migrations.embedding.common import (
    DocumentScope,
    MigrationCliError,
    PacedEmbedding,
    RateLimiter,
)


class _StubAdapter(BaseEmbedding):
    def __init__(self) -> None:
        self.calls: List[Union[str, List[str]]] = []

    def encode(  # type: ignore[override]
        self,
        text: Union[str, List[str]],
        dimension: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        self.calls.append(text)
        if isinstance(text, str):
            return [0.0, 0.0]
        return [[0.0, 0.0] for _ in text]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


def _stage(pending: int = 1) -> EmbeddingStageResult:
    return EmbeddingStageResult(pending_count=pending, embedding_count=pending)


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    *,
    docs: List[DocumentScope],
    hashes: Dict[str, Optional[str]],
    names: tuple = ("kb",),
) -> None:
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: list(names))
    monkeypatch.setattr(
        mod,
        "_resolve_target_model",
        lambda model_id: (
            EmbeddingModelConfig(id=model_id, model_name="stub", dimension=2),
            _StubAdapter(),
        ),
    )
    monkeypatch.setattr(mod, "iter_documents", lambda name, **kwargs: iter(docs))
    monkeypatch.setattr(
        mod, "load_successful_parse_hashes", lambda name, **kwargs: dict(hashes)
    )


def _record_stage(
    monkeypatch: pytest.MonkeyPatch,
    calls: List[Dict[str, Any]],
    *,
    pending: int = 1,
) -> None:
    def _embed(**kwargs: Any) -> EmbeddingStageResult:
        calls.append(kwargs)
        return _stage(pending)

    monkeypatch.setattr(mod, "embed_document_chunks", _embed)


def test_embeds_the_successful_parse_hash_under_the_document_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _wire(
        monkeypatch,
        docs=[DocumentScope("doc-1", 42, "/data/a.pdf")],
        hashes={"doc-1": "hash-1"},
    )
    _record_stage(monkeypatch, calls, pending=3)

    report = mod.embed_collections("target-model")

    assert len(calls) == 1
    assert calls[0]["parse_hash"] == "hash-1"
    assert calls[0]["user_id"] == 42
    assert calls[0]["is_admin"] is False
    assert calls[0]["is_spreadsheet"] is False
    item = report.collections[0]
    assert item.documents_total == 1
    assert item.documents_embedded == 1
    assert item.chunks_embedded == 3
    assert report.failed is False


def test_spreadsheet_sources_take_the_safeguarded_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _wire(
        monkeypatch,
        docs=[DocumentScope("doc-1", 42, "/data/sheet.XLSX")],
        hashes={"doc-1": "hash-1"},
    )
    _record_stage(monkeypatch, calls)

    mod.embed_collections("target-model")

    assert calls[0]["is_spreadsheet"] is True


def test_a_rerun_over_embedded_chunks_writes_nothing_and_reports_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _wire(
        monkeypatch,
        docs=[DocumentScope("doc-1", 42, "/data/a.pdf")],
        hashes={"doc-1": "hash-1"},
    )
    _record_stage(monkeypatch, calls, pending=0)

    report = mod.embed_collections("target-model")

    item = report.collections[0]
    assert item.documents_unchanged == 1
    assert item.documents_embedded == 0
    assert item.chunks_embedded == 0


def test_documents_without_an_owner_are_skipped_before_anything_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _wire(
        monkeypatch,
        docs=[DocumentScope("doc-1", None, "/data/a.pdf")],
        hashes={},
    )
    _record_stage(monkeypatch, calls)

    report = mod.embed_collections("target-model")

    assert calls == []
    assert report.collections[0].skipped == {mod.SKIP_NO_OWNER: 1}


def test_documents_without_a_successful_run_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _wire(
        monkeypatch,
        docs=[DocumentScope("doc-1", 42, "/data/a.pdf")],
        hashes={},
    )
    _record_stage(monkeypatch, calls)

    report = mod.embed_collections("target-model")

    assert calls == []
    assert report.collections[0].skipped == {mod.SKIP_NO_SUCCESSFUL_INGEST: 1}


def test_a_successful_run_without_a_parse_hash_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _wire(
        monkeypatch,
        docs=[DocumentScope("doc-1", 42, "/data/a.pdf")],
        hashes={"doc-1": None},
    )
    _record_stage(monkeypatch, calls)

    report = mod.embed_collections("target-model")

    assert calls == []
    assert report.collections[0].skipped == {mod.SKIP_NO_PARSE_HASH: 1}


def test_a_failed_document_does_not_stop_the_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _wire(
        monkeypatch,
        docs=[
            DocumentScope("doc-1", 42, "/data/a.pdf"),
            DocumentScope("doc-2", 42, "/data/b.pdf"),
        ],
        hashes={"doc-1": "hash-1", "doc-2": "hash-2"},
    )

    def _embed(**kwargs: Any) -> EmbeddingStageResult:
        if kwargs["doc_id"] == "doc-1":
            raise RuntimeError("provider exploded")
        calls.append(kwargs)
        return _stage(pending=2)

    monkeypatch.setattr(mod, "embed_document_chunks", _embed)

    report = mod.embed_collections("target-model")

    item = report.collections[0]
    assert item.documents_total == 2
    assert item.documents_embedded == 1
    assert [failure.doc_id for failure in item.failures] == ["doc-1"]
    assert item.failures[0].error.startswith("RuntimeError")
    assert report.failed is True


def test_consecutive_failures_abort_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(
        monkeypatch,
        docs=[
            DocumentScope(f"doc-{index}", 42, "/data/a.pdf")
            for index in range(mod.MAX_CONSECUTIVE_DOCUMENT_FAILURES + 2)
        ],
        hashes={
            f"doc-{index}": "hash"
            for index in range(mod.MAX_CONSECUTIVE_DOCUMENT_FAILURES + 2)
        },
        names=("kb-1", "kb-2"),
    )

    def _embed(**kwargs: Any) -> EmbeddingStageResult:
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(mod, "embed_document_chunks", _embed)

    report = mod.embed_collections("target-model")

    assert report.aborted is True
    assert [item.name for item in report.collections] == ["kb-1"]
    item = report.collections[0]
    assert item.aborted is True
    assert item.documents_total == mod.MAX_CONSECUTIVE_DOCUMENT_FAILURES


def test_a_collection_level_failure_is_reported_and_the_run_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(
        monkeypatch,
        docs=[DocumentScope("doc-1", 42, "/data/a.pdf")],
        hashes={"doc-1": "hash-1"},
        names=("kb-1", "kb-2"),
    )

    def _hashes(name: str, **kwargs: Any) -> Dict[str, Optional[str]]:
        if name == "kb-1":
            raise RuntimeError("unreadable table")
        return {"doc-1": "hash-1"}

    monkeypatch.setattr(mod, "load_successful_parse_hashes", _hashes)
    _record_stage(monkeypatch, [], pending=1)

    report = mod.embed_collections("target-model")

    assert [item.name for item in report.collections] == ["kb-1", "kb-2"]
    assert report.collections[0].failures[0].doc_id == "*"
    assert report.collections[1].documents_embedded == 1
    assert report.failed is True


def test_dry_run_counts_pending_chunks_without_calling_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: List[Dict[str, Any]] = []
    _wire(
        monkeypatch,
        docs=[DocumentScope("doc-1", 42, "/data/a.pdf")],
        hashes={"doc-1": "hash-1"},
    )
    monkeypatch.setattr(
        mod,
        "read_chunks_for_embedding",
        lambda **kwargs: (
            reads.append(kwargs)
            or EmbeddingReadResponse(chunks=[], total_count=0, pending_count=4)
        ),
    )
    monkeypatch.setattr(
        mod,
        "embed_document_chunks",
        lambda **kwargs: pytest.fail("dry run must not embed"),
    )

    report = mod.embed_collections("target-model", dry_run=True)

    item = report.collections[0]
    assert item.pending_chunks == 4
    assert item.documents_embedded == 1
    assert reads[0]["user_id"] == 42
    assert reads[0]["is_admin"] is False
    assert reads[0]["model"] == "target-model"


def test_reads_use_the_canonical_model_id_the_adapter_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: List[Dict[str, Any]] = []
    _wire(
        monkeypatch,
        docs=[DocumentScope("doc-1", 42, "/data/a.pdf")],
        hashes={"doc-1": "hash-1"},
    )
    monkeypatch.setattr(
        mod,
        "_resolve_target_model",
        lambda model_id: (
            EmbeddingModelConfig(id="canonical", model_name="stub", dimension=2),
            _StubAdapter(),
        ),
    )
    monkeypatch.setattr(
        mod,
        "read_chunks_for_embedding",
        lambda **kwargs: (
            reads.append(kwargs)
            or EmbeddingReadResponse(chunks=[], total_count=0, pending_count=1)
        ),
    )

    mod.embed_collections("alias", dry_run=True)

    assert reads[0]["model"] == "canonical"


def test_main_rejects_collection_with_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        mod.main(["--model", "m", "--collection", "kb", "--team", "3"])

    assert exit_info.value.code == 2


def test_main_returns_2_when_the_scope_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**kwargs: Any) -> mod.EmbedReport:
        raise MigrationCliError("--team needs an application-installed hook")

    monkeypatch.setattr(mod, "embed_collections", _raise)

    assert mod.main(["--model", "m", "--team", "3"]) == 2


def test_main_returns_2_when_the_model_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**kwargs: Any) -> mod.EmbedReport:
        raise ValueError("model 'nope' not found")

    monkeypatch.setattr(mod, "embed_collections", _raise)

    assert mod.main(["--model", "nope"]) == 2


def test_main_returns_1_when_no_collection_is_in_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "embed_collections",
        lambda *args, **kwargs: mod.EmbedReport(model_id="m", dry_run=False),
    )

    assert mod.main(["--model", "m"]) == 1


def test_main_returns_1_when_a_document_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = mod.EmbedReport(
        model_id="m",
        dry_run=False,
        collections=[
            mod.CollectionReport(
                name="kb",
                documents_total=1,
                failures=[mod.DocumentFailure("doc-1", "RuntimeError: boom")],
            )
        ],
    )
    monkeypatch.setattr(mod, "embed_collections", lambda *args, **kwargs: report)

    assert mod.main(["--model", "m"]) == 1


def test_main_returns_0_on_a_clean_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = mod.EmbedReport(
        model_id="m",
        dry_run=False,
        collections=[mod.CollectionReport(name="kb", documents_total=1)],
    )
    monkeypatch.setattr(mod, "embed_collections", lambda *args, **kwargs: report)

    assert mod.main(["--model", "m"]) == 0


def test_main_forwards_scope_and_pacing_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: Dict[str, Any] = {}

    def _capture(model_id: str, **kwargs: Any) -> mod.EmbedReport:
        seen["model_id"] = model_id
        seen.update(kwargs)
        return mod.EmbedReport(model_id=model_id, dry_run=kwargs["dry_run"])

    monkeypatch.setattr(mod, "embed_collections", _capture)

    mod.main(
        [
            "--model",
            "m",
            "--collection",
            "kb-1",
            "--collection",
            "kb-2",
            "--dry-run",
            "--rate-limit",
            "12.5",
            "--batch-size",
            "7",
            "--use-async",
        ]
    )

    assert seen == {
        "model_id": "m",
        "collections": ["kb-1", "kb-2"],
        "team": None,
        "dry_run": True,
        "rate_limit": 12.5,
        "batch_size": 7,
        "use_async": True,
    }


def test_main_rejects_a_non_positive_batch_size() -> None:
    assert mod.main(["--model", "m", "--batch-size", "0"]) == 2


# --- common.py ---


class _Batch:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    def to_pylist(self) -> List[Dict[str, Any]]:
        return self._rows


class _DocumentStore:
    def __init__(self, batches: List[List[Dict[str, Any]]]) -> None:
        self._batches = batches
        self.calls: List[Dict[str, Any]] = []

    def iter_batches(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        for rows in self._batches:
            yield _Batch(rows)


def test_documents_are_streamed_across_batches() -> None:
    store = _DocumentStore(
        [
            [{"doc_id": f"doc-{index}", "user_id": 1} for index in range(150)],
            [{"doc_id": "doc-150", "user_id": 2}],
        ]
    )

    docs = list(common.iter_documents("kb", store=store))

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
    store = _DocumentStore(
        [
            [
                {"doc_id": "", "user_id": 1},
                {"doc_id": "doc-1", "user_id": "not-a-number"},
                {"doc_id": "doc-2", "user_id": None},
                {"doc_id": "doc-3", "user_id": 7},
            ]
        ]
    )

    docs = list(common.iter_documents("kb", store=store))

    assert [(doc.doc_id, doc.user_id) for doc in docs] == [
        ("doc-1", None),
        ("doc-2", None),
        ("doc-3", 7),
    ]


class _StatusStore:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows
        self.calls: List[Dict[str, Any]] = []

    def load_ingestion_status(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.calls.append(kwargs)
        return self._rows


def test_only_successful_runs_contribute_a_parse_hash() -> None:
    store = _StatusStore(
        [
            {"doc_id": "doc-1", "status": "success", "parse_hash": "hash-1"},
            {"doc_id": "doc-2", "status": "failed", "parse_hash": "hash-2"},
            {"doc_id": "doc-3", "status": "success", "parse_hash": ""},
            {"doc_id": "", "status": "success", "parse_hash": "hash-4"},
        ]
    )

    hashes = common.load_successful_parse_hashes("kb", store=store)

    assert hashes == {"doc-1": "hash-1", "doc-3": None}
    assert store.calls == [{"collection": "kb", "user_id": None, "is_admin": True}]


def test_explicit_collections_are_trimmed_and_deduplicated() -> None:
    assert common.resolve_collection_names(collections=[" kb ", "kb", "other"]) == [
        "kb",
        "other",
    ]


def test_collection_and_team_are_mutually_exclusive() -> None:
    with pytest.raises(MigrationCliError, match="not both"):
        common.resolve_collection_names(collections=["kb"], team=3)


def test_an_empty_collection_name_is_rejected() -> None:
    with pytest.raises(MigrationCliError, match="must not be empty"):
        common.resolve_collection_names(collections=["  "])


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

    assert common.resolve_collection_names() == ["a", "b"]
    assert calls == [{"user_id": None, "is_admin": True}]


def test_team_scope_requires_the_application_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import knowledge_base_team_scope as scope

    monkeypatch.setattr(scope, "team_knowledge_base_hook_installed", lambda: False)

    with pytest.raises(MigrationCliError, match="--collection"):
        common.resolve_collection_names(team=7)


def test_team_scope_maps_hook_answers_to_collection_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.models import database
    from xagent.web.services import knowledge_base_team_scope as scope
    from xagent.web.services.knowledge_base_team_scope import KnowledgeBaseAccess

    seen: Dict[str, Any] = {}

    def _configure_db(*args: Any, **kwargs: Any) -> None:
        seen["configure_db"] = kwargs

    class _Session:
        def close(self) -> None:
            seen["closed"] = True

    class _Factory:
        def __call__(self) -> _Session:
            return _Session()

    def _team_knowledge_bases(
        session: Any, *, team_id: int
    ) -> List[KnowledgeBaseAccess]:
        seen["team_id"] = team_id
        return [
            KnowledgeBaseAccess(name="b", storage_user_id=11),
            KnowledgeBaseAccess(name="a", storage_user_id=12),
        ]

    monkeypatch.setattr(scope, "team_knowledge_base_hook_installed", lambda: True)
    monkeypatch.setattr(database, "configure_db", _configure_db)
    monkeypatch.setattr(database, "get_session_local", lambda: _Factory())
    monkeypatch.setattr(scope, "team_knowledge_bases", _team_knowledge_bases)

    assert common.resolve_collection_names(team=7) == ["a", "b"]
    assert seen["team_id"] == 7
    assert seen["configure_db"] == {"read_only": True}
    assert seen["closed"] is True


def test_rate_limiter_spaces_calls_at_the_configured_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    sleeps: List[float] = []
    monkeypatch.setattr(common.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(common.time, "sleep", lambda seconds: sleeps.append(seconds))

    limiter = RateLimiter(2.0)
    limiter.acquire(1)
    limiter.acquire(1)

    assert sleeps == [pytest.approx(0.5)]


def test_rate_limiter_never_sleeps_without_a_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: List[float] = []
    monkeypatch.setattr(common.time, "sleep", lambda seconds: sleeps.append(seconds))

    RateLimiter(0.0).acquire(10)

    assert sleeps == []


def test_paced_embedding_charges_one_unit_per_text() -> None:
    charged: List[int] = []

    class _Limiter:
        def acquire(self, count: int) -> None:
            charged.append(count)

    inner = _StubAdapter()
    paced = PacedEmbedding(inner, _Limiter())  # type: ignore[arg-type]

    assert paced.encode(["a", "b", "c"]) == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert paced.encode("a") == [0.0, 0.0]

    assert charged == [3, 1]
    assert inner.calls == [["a", "b", "c"], "a"]
    assert paced.abilities == ["embedding"]
    assert paced.get_dimension() == 2
