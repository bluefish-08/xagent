"""Tests for the per-collection embedding switch window."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    IDLE,
    CollectionMigrationState,
)
from xagent.migrations.embedding import switch as mod
from xagent.migrations.embedding.acceptance import AcceptanceResult
from xagent.migrations.embedding.embed import CollectionReport, DocumentFailure
from xagent.migrations.embedding.scope import (
    SKIP_NO_OWNER,
    DocumentScope,
)

ORIGINAL_WAIT = mod._wait_until_idle


class _StubAdapter(BaseEmbedding):
    def encode(self, text, dimension=None, instruct=None):  # type: ignore[override]
        return [0.0, 0.0]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


class _FakeWatcher:
    def __init__(self, jobs: Optional[List[str]] = None) -> None:
        self.jobs = jobs or []
        self.calls: List[str] = []

    def running(self, collection: str) -> List[str]:
        self.calls.append(collection)
        return list(self.jobs)


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str = "lancedb",
    current: Optional[str] = "old-model",
    target: str = "new-model",
    state: Optional[CollectionMigrationState] = None,
    info_error: Optional[BaseException] = None,
    skipped_docs: Sequence[DocumentScope] = (),
) -> Dict[str, List[Any]]:
    calls: Dict[str, List[Any]] = {"freeze": [], "unfreeze": [], "switch": []}

    monkeypatch.setattr(mod, "collection_backend", lambda name: backend)

    def _get_collection(name: str) -> Any:
        if info_error is not None:
            raise info_error
        return SimpleNamespace(embedding_model_id=current)

    monkeypatch.setattr(mod, "get_collection_sync", _get_collection)
    monkeypatch.setattr(
        mod,
        "_resolve_target",
        lambda model_id: (
            EmbeddingModelConfig(id=target, model_name="stub", dimension=2),
            _StubAdapter(),
        ),
    )
    monkeypatch.setattr(
        mod, "read_migration_state", lambda name: state or CollectionMigrationState()
    )

    def _write_state(name: str, value: CollectionMigrationState) -> Any:
        key = "freeze" if value.state == FROZEN else "unfreeze"
        calls[key].append(
            (name, value.state, value.source_model_id, value.target_model_id)
        )
        return value

    monkeypatch.setattr(mod, "write_migration_state", _write_state)
    monkeypatch.setattr(mod, "_wait_until_idle", lambda *args, **kwargs: [])
    monkeypatch.setattr(mod, "wait_for_no_leases", lambda *args, **kwargs: [])
    monkeypatch.setattr(mod, "prune_orphan_documents", lambda *args, **kwargs: 0)
    monkeypatch.setattr(mod, "embed_collection", _happy_embed)
    monkeypatch.setattr(mod, "iter_scope_documents", lambda name: iter(skipped_docs))
    monkeypatch.setattr(
        mod,
        "switch_collection_embedding_model",
        lambda name, model: calls["switch"].append((name, model)),
    )
    return calls


def _happy_embed(*args: Any, **kwargs: Any) -> CollectionReport:
    return CollectionReport(
        name="kb", documents_embedded=2, chunks_embedded=5, documents_total=2
    )


def _ok_acceptance(*args: Any, **kwargs: Any) -> AcceptanceResult:
    return AcceptanceResult(ok=True, checks={"source_count": 5, "target_count": 5})


def test_a_healthy_collection_is_frozen_switched_and_unfrozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_FakeWatcher(), acceptance=_ok_acceptance
    )

    assert report.result == mod.RESULT_SWITCHED
    assert report.source_model_id == "old-model"
    assert report.documents_embedded == 2
    assert report.chunks_embedded == 5
    assert calls["switch"] == [("kb", "new-model")]
    assert calls["freeze"] == [("kb", FROZEN, "old-model", "new-model")]
    assert calls["unfreeze"] == [("kb", IDLE, "old-model", "new-model")]
    assert report.window_seconds >= 0.0


def test_a_collection_already_on_the_target_needs_no_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch, current="new-model")

    report = mod.switch_collection("kb", "new-model", job_watcher=_FakeWatcher())

    assert report.result == mod.RESULT_ALREADY
    assert calls["freeze"] == []
    assert calls["switch"] == []


def test_a_resumed_window_does_not_freeze_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(
        monkeypatch,
        state=CollectionMigrationState(
            state=FROZEN, source_model_id="old-model", target_model_id="new-model"
        ),
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_FakeWatcher(), acceptance=_ok_acceptance
    )

    assert report.result == mod.RESULT_SWITCHED
    assert calls["freeze"] == []
    assert calls["switch"] == [("kb", "new-model")]


def test_a_resumed_switch_that_already_landed_is_unfrozen_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(
        monkeypatch,
        current="new-model",
        state=CollectionMigrationState(
            state=FROZEN, source_model_id="old-model", target_model_id="new-model"
        ),
    )

    report = mod.switch_collection("kb", "new-model", job_watcher=_FakeWatcher())

    assert report.result == mod.RESULT_ALREADY
    assert calls["unfreeze"] == [("kb", IDLE, "old-model", "new-model")]
    assert calls["switch"] == []


def test_a_collection_frozen_for_another_target_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(
        monkeypatch,
        state=CollectionMigrationState(
            state=FROZEN, source_model_id="old-model", target_model_id="third-model"
        ),
    )

    report = mod.switch_collection("kb", "new-model", job_watcher=_FakeWatcher())

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("frozen-for-another-target")
    assert calls["freeze"] == []
    assert calls["switch"] == []


@pytest.mark.parametrize(
    ("backend", "current", "info_error", "reason"),
    [
        ("milvus", "old-model", None, mod.SKIP_OTHER_ENGINE),
        ("lancedb", None, None, "unknown-collection"),
        ("lancedb", "", None, "unbound"),
    ],
)
def test_the_window_refuses_collections_it_cannot_switch(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    current: Optional[str],
    info_error: Optional[BaseException],
    reason: str,
) -> None:
    if reason == "unknown-collection":
        info_error = ValueError("Collection not found")
    calls = _wire(monkeypatch, backend=backend, current=current, info_error=info_error)

    report = mod.switch_collection("kb", "new-model", job_watcher=_FakeWatcher())

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == reason
    assert calls["freeze"] == []
    assert calls["switch"] == []


def test_an_unresolvable_target_is_refused_before_freezing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    def _raise(model_id: str) -> Any:
        raise mod.MigrationCliError("'alias' resolves to 'new-model'")

    monkeypatch.setattr(mod, "_resolve_target", _raise)

    report = mod.switch_collection("kb", "alias", job_watcher=_FakeWatcher())

    assert report.result == mod.RESULT_REFUSED
    assert "resolves to" in report.reason
    assert calls["freeze"] == []


def test_in_flight_writers_refuse_the_window_and_unfreeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    monkeypatch.setattr(mod, "_wait_until_idle", lambda *args, **kwargs: ["job:job-1"])

    report = mod.switch_collection("kb", "new-model", job_watcher=_FakeWatcher())

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("in-flight-writers")
    assert calls["freeze"] == [("kb", FROZEN, "old-model", "new-model")]
    assert calls["unfreeze"] == [("kb", IDLE, "old-model", "new-model")]
    assert calls["switch"] == []


def test_a_rename_admitted_before_the_freeze_refuses_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch)
    calls = {"n": 0}
    original = SimpleNamespace(embedding_model_id="old-model")

    def _get_collection(name: str) -> Any:
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("Collection not found")
        return original

    monkeypatch.setattr(mod, "get_collection_sync", _get_collection)

    report = mod.switch_collection("kb", "new-model", job_watcher=_FakeWatcher())

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == "renamed"


def test_document_failures_refuse_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    def _failing_embed(*args: Any, **kwargs: Any) -> CollectionReport:
        return CollectionReport(
            name="kb", documents_total=1, failures=[DocumentFailure("d", "boom")]
        )

    monkeypatch.setattr(mod, "embed_collection", _failing_embed)

    report = mod.switch_collection("kb", "new-model", job_watcher=_FakeWatcher())

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("embed-failed")
    assert calls["switch"] == []


def test_skipped_documents_reach_acceptance_and_ownerless_ones_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(
        monkeypatch,
        skipped_docs=[
            DocumentScope("legacy", None, None, skip_reason=SKIP_NO_OWNER),
            DocumentScope("ok", 1, None, parse_hash="h"),
        ],
    )
    seen: List[Dict[str, Any]] = []

    def _acceptance(*args: Any, **kwargs: Any) -> AcceptanceResult:
        seen.append(kwargs)
        return AcceptanceResult(ok=True, checks={"ownerless_excluded": 1})

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_FakeWatcher(), acceptance=_acceptance
    )

    assert seen[0]["skipped"] == {"legacy": SKIP_NO_OWNER}
    assert report.result == mod.RESULT_SWITCHED
    assert report.ownerless_excluded == 1
    assert calls["switch"] == [("kb", "new-model")]


def test_uncovered_documents_refuse_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    def _acceptance(*args: Any, **kwargs: Any) -> AcceptanceResult:
        return AcceptanceResult(ok=False, failures=["x"], uncovered=["doc-9"])

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_FakeWatcher(), acceptance=_acceptance
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == "uncovered-documents: doc-9"
    assert calls["switch"] == []
    assert calls["unfreeze"] == [("kb", IDLE, "old-model", "new-model")]


def test_the_freeze_is_written_before_leases_are_first_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch)
    events: List[str] = []

    def _write_state(name: str, value: CollectionMigrationState) -> Any:
        events.append(f"state:{value.state}")
        return value

    def _leases(name: str) -> List[Any]:
        events.append("leases")
        return []

    monkeypatch.setattr(mod, "write_migration_state", _write_state)
    monkeypatch.setattr(mod, "live_leases", _leases)
    monkeypatch.setattr(mod, "_wait_until_idle", ORIGINAL_WAIT)

    mod.switch_collection(
        "kb", "new-model", job_watcher=_FakeWatcher(), acceptance=_ok_acceptance
    )

    assert events[:2] == [f"state:{FROZEN}", "leases"]


def test_a_failed_acceptance_refuses_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    def _bad_acceptance(*args: Any, **kwargs: Any) -> AcceptanceResult:
        return AcceptanceResult(
            ok=False, checks={"source_count": 5, "target_count": 4}, failures=["count"]
        )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_FakeWatcher(), acceptance=_bad_acceptance
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("acceptance-failed")
    assert report.acceptance == {"source_count": 5, "target_count": 4}
    assert calls["switch"] == []
    assert calls["unfreeze"] == [("kb", IDLE, "old-model", "new-model")]


def test_writers_reappearing_before_the_flip_refuse_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    monkeypatch.setattr(
        mod,
        "wait_for_no_leases",
        lambda *args, **kwargs: [SimpleNamespace(operation="late", owner="h:1")],
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_FakeWatcher(), acceptance=_ok_acceptance
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("writers-restarted")
    assert calls["switch"] == []


def test_a_switch_error_unfreezes_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    def _explode(name: str, model: str) -> None:
        calls["switch"].append((name, model))
        raise RuntimeError("binding write exploded")

    monkeypatch.setattr(mod, "switch_collection_embedding_model", _explode)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_FakeWatcher(), acceptance=_ok_acceptance
    )

    assert report.result == mod.RESULT_FAILED
    assert "RuntimeError" in report.reason
    assert calls["unfreeze"] == [("kb", IDLE, "old-model", "new-model")]


def test_prune_deletes_rows_for_documents_that_no_longer_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Store:
        def __init__(self) -> None:
            self.deleted: List[tuple] = []

        def iter_batches(self, **kwargs: Any) -> Any:
            table = kwargs["table_name"]
            ids = (
                ["doc-a", "doc-gone"] if table.startswith("embeddings_") else ["doc-a"]
            )
            yield SimpleNamespace(
                to_pylist=lambda: [{"doc_id": doc_id} for doc_id in ids]
            )

        def delete_embedding_records(
            self, name: str, doc_id: str, **kwargs: Any
        ) -> int:
            self.deleted.append((name, doc_id, kwargs))
            return 3

    store = _Store()
    monkeypatch.setattr(mod, "get_vector_index_store", lambda: store)
    monkeypatch.setattr(mod, "embeddings_table", lambda model: "embeddings_new")

    deleted = mod.prune_orphan_documents("kb", target_model_id="new-model")

    assert deleted == 3
    assert store.deleted == [
        (
            "kb",
            "doc-gone",
            {"model_tag": "new-model", "user_id": None, "is_admin": True},
        )
    ]


def test_wait_until_idle_reports_what_is_left_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        mod.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds)
    )
    monkeypatch.setattr(
        mod,
        "live_leases",
        lambda name: [SimpleNamespace(operation="wipe", owner="h:2")],
    )

    left = mod._wait_until_idle(
        "kb", timeout=2.0, poll=1.0, watcher=_FakeWatcher(["job-1"])
    )

    assert "job:job-1" in left
    assert "lease:wipe@h:2" in left


def test_cli_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "switch_collection",
        lambda *args, **kwargs: mod.SwitchReport(
            name="kb", target_model_id="new-model", result=mod.RESULT_SWITCHED
        ),
    )
    assert mod.main(["--model", "new-model", "--collection", "kb"]) == 0

    monkeypatch.setattr(
        mod,
        "switch_collection",
        lambda *args, **kwargs: mod.SwitchReport(
            name="kb", target_model_id="new-model", result=mod.RESULT_REFUSED
        ),
    )
    assert mod.main(["--model", "new-model", "--collection", "kb"]) == 1


def test_cli_returns_2_when_the_run_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no model hub")

    monkeypatch.setattr(mod, "switch_collection", _raise)

    assert mod.main(["--model", "new-model", "--collection", "kb"]) == 2
