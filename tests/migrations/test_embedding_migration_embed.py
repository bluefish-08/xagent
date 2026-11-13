"""Tests for the embedding migration's phase-1 ``embed`` command."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import EmbeddingReadResponse
from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
    EmbeddingStageResult,
)
from xagent.migrations.embedding import embed as mod
from xagent.migrations.embedding import scope
from xagent.migrations.embedding.embed import PacedEmbedding, RateLimiter
from xagent.migrations.embedding.scope import DocumentScope, MigrationCliError


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
        "resolve_target_model",
        lambda model_id: (
            EmbeddingModelConfig(id=model_id, model_name="stub", dimension=2),
            _StubAdapter(),
        ),
    )
    monkeypatch.setattr(scope, "iter_documents", lambda name, **kwargs: iter(docs))
    monkeypatch.setattr(
        scope, "load_successful_parse_hashes", lambda name, **kwargs: dict(hashes)
    )
    monkeypatch.setattr(mod, "collection_backend", lambda name: "lancedb")


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


def test_collections_on_another_engine_are_skipped_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    _wire(
        monkeypatch,
        docs=[DocumentScope("doc-1", 42, "/data/a.pdf")],
        hashes={"doc-1": "hash-1"},
        names=("milvus-kb", "kb"),
    )
    monkeypatch.setattr(
        mod,
        "collection_backend",
        lambda name: "milvus" if name == "milvus-kb" else "lancedb",
    )
    _record_stage(monkeypatch, calls)

    report = mod.embed_collections("target-model")

    skipped = report.collections[0]
    assert skipped.collection_skip == mod.SKIP_OTHER_ENGINE
    assert skipped.engine == "milvus"
    assert skipped.documents_total == 0
    assert report.collections[1].documents_embedded == 1
    assert [call["collection"] for call in calls] == ["kb"]
    assert report.failed is False


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
    assert report.collections[0].skipped == {scope.SKIP_NO_OWNER: 1}


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
    assert report.collections[0].skipped == {scope.SKIP_NO_SUCCESSFUL_INGEST: 1}


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
    assert report.collections[0].skipped == {scope.SKIP_NO_PARSE_HASH: 1}


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

    monkeypatch.setattr(scope, "load_successful_parse_hashes", _hashes)
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
        "resolve_target_model",
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


# --- pacing ---
def test_rate_limiter_spaces_calls_at_the_configured_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    sleeps: List[float] = []
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    limiter = RateLimiter(2.0)
    limiter.acquire(1)
    limiter.acquire(1)

    assert sleeps == [pytest.approx(0.5)]


def test_rate_limiter_never_sleeps_without_a_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: List[float] = []
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))

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
