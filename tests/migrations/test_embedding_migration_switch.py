"""Tests for the per-collection embedding switch window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    FROZEN,
    IDLE,
    CollectionMigrationState,
    read_migration_state,
    write_migration_state,
)
from xagent.migrations.embedding import switch as mod
from xagent.migrations.embedding.acceptance import AcceptanceResult
from xagent.migrations.embedding.embed import CollectionReport, DocumentFailure


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
    engine: str = "lancedb",
    bound: Optional[str] = "old-model",
    info: Any = "default",
    embed: Any = None,
    acceptance: Any = None,
    jobs: Optional[List[str]] = None,
    leases: Optional[List[Any]] = None,
    create_at: Any = "same",
) -> Dict[str, List[Any]]:
    calls: Dict[str, List[Any]] = {"switch": [], "embed": [], "accept": []}
    if info == "default":
        info_obj: Any = SimpleNamespace(
            embedding_model_id=bound,
            created_at=_CREATED if create_at == "same" else create_at,
        )
    else:
        info_obj = info
    state = {"info": info_obj}

    def _collection_info(name: str) -> Any:
        if state["info"] is None:
            raise mod.MigrationCliError(f"collection '{name}' does not exist")
        return state["info"]

    monkeypatch.setattr(mod, "collection_info", _collection_info)
    monkeypatch.setattr(mod, "collection_engine", lambda item: engine)
    monkeypatch.setattr(
        mod,
        "source_model_id",
        lambda name, item: (item.embedding_model_id or "").strip() or None,
    )
    monkeypatch.setattr(
        mod,
        "resolve_target_model",
        lambda model_id: (
            EmbeddingModelConfig(id=model_id, model_name="stub", dimension=2),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(mod, "connection", lambda: object())
    monkeypatch.setattr(mod, "iter_source_chunks", lambda *args, **kwargs: iter([]))
    monkeypatch.setattr(
        mod,
        "embed_collection",
        embed
        or (
            lambda *args, **kwargs: calls["embed"].append(kwargs)
            or CollectionReport(
                name="kb", documents_embedded=2, chunks_embedded=5, documents_total=2
            )
        ),
    )
    monkeypatch.setattr(
        mod,
        "check_collection",
        acceptance
        or (
            lambda *args, **kwargs: calls["accept"].append(kwargs)
            or AcceptanceResult(ok=True, checks={"ownerless_excluded": 0})
        ),
    )
    monkeypatch.setattr(mod, "live_leases", lambda name: list(leases or []))
    monkeypatch.setattr(mod, "FINAL_LEASE_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(
        mod,
        "switch_collection_embedding_model",
        lambda name, model: calls["switch"].append((name, model)),
    )
    monkeypatch.setattr(
        mod,
        "job_watcher",
        None,
        raising=False,
    )
    return calls


_CREATED = datetime(2026, 1, 1)


def _watcher(jobs: Optional[List[str]] = None) -> _FakeWatcher:
    return _FakeWatcher(jobs)


def test_a_healthy_collection_is_frozen_switched_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED
    assert report.source_model_id == "old-model"
    assert report.chunks_embedded == 5
    assert calls["switch"] == [("kb", "new-model")]
    state = read_migration_state("kb")
    assert state.state == IDLE
    assert state.source_model_id == "old-model"
    assert state.target_model_id == "new-model"
    assert state.mode == mod.MODE_SWITCH
    assert state.owner is None


def test_a_refusal_restores_the_previous_row_without_claiming_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=IDLE,
            source_model_id="model-a",
            target_model_id="old-model",
            mode=mod.MODE_SWITCH,
        ),
    )

    def _failing_acceptance(*args: Any, **kwargs: Any) -> AcceptanceResult:
        return AcceptanceResult(ok=False, failures=["chunks differ"])

    calls = _wire(monkeypatch, acceptance=_failing_acceptance)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("acceptance-failed")
    assert calls["switch"] == []
    state = read_migration_state("kb")
    assert state.state == IDLE
    assert state.source_model_id == "model-a"
    # A refused window must not look like a completed switch.
    assert state.target_model_id is None
    assert state.mode is None


def test_a_window_owned_by_another_run_is_not_resumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch)
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="old-model",
            target_model_id="new-model",
            owner="other-host:1:abcd",
        ),
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("window-in-progress")
    assert read_migration_state("kb").state == FROZEN


def _age_row(name: str, seconds: float) -> None:
    """Rewrite the row's timestamp: the public writer always stamps now."""
    from xagent.core.tools.core.RAG_tools.management.migration_state import (
        MIGRATION_TABLE,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    table = get_metadata_store().get_raw_connection().open_table(MIGRATION_TABLE)
    table.update(
        f"name = '{name}'",
        {
            "updated_at": datetime.now(timezone.utc).replace(tzinfo=None)
            - timedelta(seconds=seconds)
        },
    )


def test_a_stale_window_is_resumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="old-model",
            target_model_id="new-model",
            owner="dead-host:9:beef",
        ),
    )
    _age_row("kb", mod.STALE_FREEZE_SECONDS + 60)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED
    assert calls["switch"] == [("kb", "new-model")]


def test_our_own_window_is_resumed(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch)
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="old-model",
            target_model_id="new-model",
            owner="me",
        ),
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), owner="me", wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED


def test_a_frozen_window_for_another_target_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=FROZEN,
            source_model_id="old-model",
            target_model_id="third-model",
            owner=None,
        ),
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("frozen-for-another-target")
    assert calls["switch"] == []
    assert read_migration_state("kb").state == FROZEN


def test_a_collection_already_on_the_target_needs_no_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch, bound="new-model")

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_ALREADY
    assert calls["switch"] == []
    assert read_migration_state("kb").state == IDLE


@pytest.mark.parametrize(
    ("engine", "bound", "info", "reason"),
    [
        ("milvus", "old-model", "default", mod.SKIP_OTHER_ENGINE),
        ("lancedb", "old-model", None, "unknown-collection"),
        ("lancedb", None, "default", "unbound"),
    ],
)
def test_the_window_refuses_collections_it_cannot_switch(
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    bound: Optional[str],
    info: Any,
    reason: str,
) -> None:
    calls = _wire(monkeypatch, engine=engine, bound=bound, info=info)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == reason
    assert calls["switch"] == []


def test_an_unresolvable_target_is_refused_before_freezing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch)

    def _raise(model_id: str) -> Any:
        raise mod.MigrationCliError(f"'{model_id}' is not configured")

    monkeypatch.setattr(mod, "resolve_target_model", _raise)

    report = mod.switch_collection(
        "kb", "alias", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert "not configured" in report.reason
    assert read_migration_state("kb").state == IDLE


def test_in_flight_writers_refuse_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(["job-1"]), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("in-flight-writers")
    assert calls["switch"] == []
    assert read_migration_state("kb").state == IDLE


def test_a_live_lease_also_holds_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, leases=[SimpleNamespace(operation="wipe", owner="h:2")])

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("in-flight-writers")


def test_a_rename_is_detected_by_identity_not_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch)
    seen = {"n": 0}
    original = SimpleNamespace(embedding_model_id="old-model", created_at=_CREATED)

    def _collection_info(name: str) -> Any:
        seen["n"] += 1
        if seen["n"] > 1:
            return SimpleNamespace(
                embedding_model_id="old-model",
                created_at=_CREATED + timedelta(seconds=1),
            )
        return original

    monkeypatch.setattr(mod, "collection_info", _collection_info)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == "renamed"


def test_ownerless_chunks_are_excluded_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skipped_docs: Dict[str, Any] = {}

    def _embed(*args: Any, **kwargs: Any) -> CollectionReport:
        return CollectionReport(
            name="kb",
            documents_embedded=1,
            skipped={"no-owner": 3},
            documents_total=1,
        )

    def _acceptance(*args: Any, **kwargs: Any) -> AcceptanceResult:
        skipped_docs.update(kwargs)
        return AcceptanceResult(ok=True, checks={"ownerless_excluded": 3})

    calls = _wire(monkeypatch, embed=_embed, acceptance=_acceptance)
    monkeypatch.setattr(
        mod,
        "iter_source_chunks",
        lambda *args, **kwargs: iter(
            [
                SimpleNamespace(
                    doc_id="legacy",
                    chunk_id="l1",
                    parse_hash="h",
                    text="t",
                    chunk_hash="c",
                    metadata=None,
                    user_id=None,
                )
            ]
        ),
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED
    assert report.ownerless_excluded == 3
    assert skipped_docs["skipped"] == {"legacy"}
    assert skipped_docs["source_model_id"] == "old-model"
    assert skipped_docs["target_model_id"] == "new-model"
    assert calls["switch"] == [("kb", "new-model")]


@pytest.mark.parametrize("reason", ["no-text", "no-doc-id"])
def test_uncopyable_chunks_are_excluded_and_reported_not_refused(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    def _embed(*args: Any, **kwargs: Any) -> CollectionReport:
        return CollectionReport(name="kb", skipped={reason: 2, "no-owner": 1})

    seen: Dict[str, Any] = {}

    def _acceptance(*args: Any, **kwargs: Any) -> AcceptanceResult:
        seen.update(kwargs)
        return AcceptanceResult(ok=True, checks={"ownerless_excluded": 1})

    calls = _wire(monkeypatch, embed=_embed, acceptance=_acceptance)
    monkeypatch.setattr(
        mod,
        "iter_source_chunks",
        lambda *args, **kwargs: iter(
            [
                SimpleNamespace(
                    doc_id="empty",
                    chunk_id="e1",
                    parse_hash="h",
                    text="",
                    chunk_hash="c",
                    metadata=None,
                    user_id=7,
                )
            ]
        ),
    )

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED
    assert report.skipped == {reason: 2, "no-owner": 1}
    assert report.ownerless_excluded == 1
    assert seen["skipped"] == {"empty"}
    assert calls["switch"] == [("kb", "new-model")]


def test_a_failed_copy_still_refuses_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _embed(*args: Any, **kwargs: Any) -> CollectionReport:
        return CollectionReport(
            name="kb",
            failures=[DocumentFailure("d", "1 chunks missing from the target model")],
        )

    calls = _wire(monkeypatch, embed=_embed)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("embed-failed")
    assert calls["switch"] == []


def test_document_failures_refuse_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _embed(*args: Any, **kwargs: Any) -> CollectionReport:
        return CollectionReport(
            name="kb", documents_total=1, failures=[DocumentFailure("d", "boom")]
        )

    calls = _wire(monkeypatch, embed=_embed)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("embed-failed")
    assert calls["switch"] == []


def test_writers_reappearing_before_the_flip_refuse_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    counts = {"n": 0}

    def _leases(name: str) -> List[Any]:
        counts["n"] += 1
        if counts["n"] > 1:
            return [SimpleNamespace(operation="late", owner="h:1")]
        return []

    monkeypatch.setattr(mod, "live_leases", _leases)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason.startswith("writers-restarted")
    assert calls["switch"] == []


def test_a_switch_error_reports_failure_and_unfreezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    def _explode(name: str, model: str) -> None:
        calls["switch"].append((name, model))
        raise RuntimeError("binding write exploded")

    monkeypatch.setattr(mod, "switch_collection_embedding_model", _explode)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_FAILED
    assert "RuntimeError" in report.reason
    assert read_migration_state("kb").state == IDLE


def test_a_failed_record_after_the_flip_is_reported_as_switched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)

    def _write(name: str, state: CollectionMigrationState) -> Any:
        if state.state == IDLE and state.target_model_id is not None:
            raise OSError("metadata store gone")
        write_migration_state(name, state)
        return state

    monkeypatch.setattr(mod, "write_migration_state", _write)

    report = mod.switch_collection(
        "kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0
    )

    assert report.result == mod.RESULT_SWITCHED
    assert report.reason.startswith("left-frozen")
    assert calls["switch"] == [("kb", "new-model")]


def test_the_run_config_and_rate_limit_reach_the_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch)
    cfg = IngestionConfig(
        embedding_model_id="new-model", embedding_batch_size=7, embedding_use_async=True
    )

    mod.switch_collection(
        "kb",
        "new-model",
        job_watcher=_watcher(),
        cfg=cfg,
        rate_limit=3.0,
        wait_seconds=0.0,
    )

    assert calls["embed"][0]["cfg"] is cfg
    assert calls["embed"][0]["dry_run"] is False


def test_rollback_needs_a_recorded_switch_back_to_its_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _wire(monkeypatch, bound="new-model")

    report = mod.switch_collection(
        "kb",
        "old-model",
        job_watcher=_watcher(),
        mode=mod.MODE_ROLLBACK,
        wait_seconds=0.0,
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == "no-recorded-switch"
    assert calls["switch"] == []


def test_rollback_refuses_when_the_record_is_for_another_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, bound="new-model")
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=IDLE,
            source_model_id="other-model",
            target_model_id="new-model",
            mode=mod.MODE_SWITCH,
        ),
    )

    report = mod.switch_collection(
        "kb",
        "old-model",
        job_watcher=_watcher(),
        mode=mod.MODE_ROLLBACK,
        wait_seconds=0.0,
    )

    assert report.result == mod.RESULT_REFUSED
    assert report.reason == "recorded-source-is-other-model"


def test_rollback_undoes_a_recorded_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _wire(monkeypatch, bound="new-model")
    write_migration_state(
        "kb",
        CollectionMigrationState(
            state=IDLE,
            source_model_id="old-model",
            target_model_id="new-model",
            mode=mod.MODE_SWITCH,
        ),
    )

    report = mod.switch_collection(
        "kb",
        "old-model",
        job_watcher=_watcher(),
        mode=mod.MODE_ROLLBACK,
        wait_seconds=0.0,
    )

    assert report.result == mod.RESULT_SWITCHED
    assert calls["switch"] == [("kb", "old-model")]
    assert read_migration_state("kb").mode == mod.MODE_ROLLBACK


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
        "kb", timeout=2.0, poll=1.0, watcher=_watcher(["job-1"])
    )

    assert "job:job-1" in left
    assert "lease:wipe@h:2" in left


def test_the_freeze_is_written_before_the_first_lease_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch)
    events: List[str] = []

    def _write(name: str, state: CollectionMigrationState) -> Any:
        events.append(f"state:{state.state}")
        write_migration_state(name, state)
        return state

    def _leases(name: str) -> List[Any]:
        events.append("leases")
        return []

    monkeypatch.setattr(mod, "write_migration_state", _write)
    monkeypatch.setattr(mod, "live_leases", _leases)

    mod.switch_collection("kb", "new-model", job_watcher=_watcher(), wait_seconds=0.0)

    assert events[:2] == [f"state:{FROZEN}", "leases"]


def test_the_job_watcher_reads_running_ingest_jobs_from_the_database(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.web.models.background_job import BackgroundJob, BackgroundJobStatus
    from xagent.web.models.database import get_session_local, init_db
    from xagent.web.models.user import User

    init_db(f"sqlite:///{tmp_path / 'watcher.db'}")
    sessions = get_session_local()
    session = sessions()
    try:
        session.add(User(id=1, username="watcher", password_hash="x"))
        session.commit()
        for job_id, status, collection in (
            ("running-kb", BackgroundJobStatus.RUNNING.value, "kb"),
            ("running-other", BackgroundJobStatus.RUNNING.value, "other"),
            ("pending-kb", BackgroundJobStatus.PENDING.value, "kb"),
        ):
            session.add(
                BackgroundJob(
                    id=job_id,
                    user_id=1,
                    job_type="kb.ingest.document",
                    status=status,
                    payload={"collection": collection},
                )
            )
        session.commit()
    finally:
        session.close()

    from xagent.web.models import database

    # The command configures the application's database; here that is the
    # sqlite file the test just filled.
    monkeypatch.setattr(database, "configure_db", lambda *a, **k: None)
    monkeypatch.setattr(database, "get_session_local", lambda: sessions)

    watcher = mod.InflightJobWatcher()

    assert watcher.running("kb") == ["running-kb"]
    assert watcher.running("other") == ["running-other"]
