"""Tests for the embeddings FTS index rebuild migration."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from xagent.migrations.lancedb import rebuild_fts_indexes as mod


class _FakeIndex:
    def __init__(self, name: str) -> None:
        self.name = name
        self.index_type = "FTS"
        self.columns = ["text"]


class _FakeStats:
    num_indexed_rows = 10
    num_unindexed_rows = 0


class _FakeTable:
    def __init__(self, version: int = 7) -> None:
        self.version = version
        self.closed = False

    def list_indices(self) -> List[_FakeIndex]:
        return [_FakeIndex("text_idx")]

    def index_stats(self, name: str) -> _FakeStats:
        return _FakeStats()

    def close(self) -> None:
        self.closed = True


class _FakeConn:
    """A rebuild bumps the table version, which is how the script detects it."""

    def __init__(self, names: List[str]) -> None:
        self._names = names
        self.versions = {name: 7 for name in names}

    def table_names(self) -> List[str]:
        return self._names

    def open_table(self, name: str) -> _FakeTable:
        if name not in self._names:
            raise ValueError(f"no table {name}")
        return _FakeTable(self.versions[name])


class _RecordingStore:
    """Stands in for LanceDBVectorIndexStore; ``fail`` names tables that raise."""

    def __init__(
        self, fail: Dict[str, Any] | None = None, conn: "_FakeConn | None" = None
    ) -> None:
        self.calls: List[str] = []
        self.fail = fail or {}
        self.conn = conn

    def trigger_reindex(self, table_name: str) -> bool:
        self.calls.append(table_name)
        outcome = self.fail.get(table_name)
        if isinstance(outcome, Exception):
            raise outcome
        if self.conn is not None and outcome is not False:
            self.conn.versions[table_name] += 2
        return True if outcome is None else outcome


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _RecordingStore:
    recorder = _RecordingStore()
    monkeypatch.setattr(mod, "LanceDBVectorIndexStore", lambda: recorder)
    return recorder


def _wire(monkeypatch: pytest.MonkeyPatch, conn: _FakeConn, **kw: Any):
    recorder = _RecordingStore(conn=conn, **kw)
    monkeypatch.setattr(mod, "LanceDBVectorIndexStore", lambda: recorder)
    return recorder


def test_lists_only_embeddings_tables() -> None:
    conn = _FakeConn(
        ["documents", "embeddings_b", "chunks", "embeddings_a", "parses"],
    )
    assert mod.list_embeddings_tables(conn) == ["embeddings_a", "embeddings_b"]


def test_dry_run_does_not_rebuild(store: _RecordingStore) -> None:
    conn = _FakeConn(["documents", "embeddings_a", "embeddings_b"])

    result = mod.rebuild_fts_indexes(dry_run=True, conn=conn)

    assert store.calls == []
    assert result["tables"] == ["embeddings_a", "embeddings_b"]
    assert result["succeeded"] == []
    assert result["failed"] == []
    assert result["dry_run"] is True


def test_failure_does_not_stop_remaining_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConn(["embeddings_a", "embeddings_b", "embeddings_c"])
    recorder = _wire(monkeypatch, conn, fail={"embeddings_b": RuntimeError("boom")})

    result = mod.rebuild_fts_indexes(conn=conn)

    assert recorder.calls == ["embeddings_a", "embeddings_b", "embeddings_c"]
    assert result["succeeded"] == ["embeddings_a", "embeddings_c"]
    assert result["failed"] == ["embeddings_b"]


def test_trigger_reindex_returning_false_counts_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConn(["embeddings_a", "embeddings_b"])
    _wire(monkeypatch, conn, fail={"embeddings_a": False})

    result = mod.rebuild_fts_indexes(conn=conn)

    assert result["failed"] == ["embeddings_a"]
    assert result["succeeded"] == ["embeddings_b"]


def test_table_option_rejects_non_embeddings_table(store: _RecordingStore) -> None:
    conn = _FakeConn(["documents", "embeddings_a"])

    with pytest.raises(ValueError, match="documents"):
        mod.rebuild_fts_indexes(table="documents", conn=conn)
    assert store.calls == []


def test_table_option_rebuilds_only_that_table(store: _RecordingStore) -> None:
    conn = _FakeConn(["embeddings_a", "embeddings_b"])

    result = mod.rebuild_fts_indexes(table="embeddings_b", conn=conn)

    assert store.calls == ["embeddings_b"]
    assert result["tables"] == ["embeddings_b"]


def test_main_exit_code_reflects_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(["embeddings_a", "embeddings_b"])
    _wire(monkeypatch, conn, fail={"embeddings_b": RuntimeError("boom")})
    monkeypatch.setattr(mod, "get_connection_from_env", lambda: conn)

    assert mod.main([]) == 1
    assert mod.main(["--dry-run"]) == 0
    assert mod.main(["--table", "embeddings_a"]) == 0
    assert mod.main(["--table", "documents"]) == 2


def test_describe_indexes_reports_version_and_row_counts() -> None:
    conn = _FakeConn(["embeddings_a"])

    summary = mod.describe_indexes(conn, "embeddings_a")

    assert summary["version"] == 7
    assert summary["indexes"]["text_idx"]["type"] == "FTS"
    assert summary["indexes"]["text_idx"]["indexed_rows"] == 10


def test_rebuilt_index_counts_even_when_compaction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trigger_reindex returns False when only optimize failed; the index is fine."""
    conn = _FakeConn(["embeddings_a"])
    recorder = _wire(monkeypatch, conn, fail={"embeddings_a": False})
    conn.versions["embeddings_a"] = 7

    original = recorder.trigger_reindex

    def rebuild_then_fail_optimize(table_name: str) -> bool:
        conn.versions[table_name] += 2
        return original(table_name)

    recorder.trigger_reindex = rebuild_then_fail_optimize  # type: ignore[method-assign]

    result = mod.rebuild_fts_indexes(conn=conn)

    assert result["succeeded"] == ["embeddings_a"]
    assert result["failed"] == []
