"""Tests for the switch's per-document acceptance check."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pyarrow as pa
import pytest

from xagent.migrations.embedding import acceptance as mod

Row = Tuple[str, str, Optional[int]]


class _Table:
    def __init__(self, rows: List[Dict[str, Any]], *, nulls: int = 0) -> None:
        self._rows = rows
        self._nulls = nulls

    def to_batches(self, *, filter: str, batch_size: int = 1000) -> Iterator[Any]:
        if not self._rows:
            return
        rows = [dict(row) for row in self._rows]
        for index in range(self._nulls):
            rows[index]["vector"] = None
        yield pa.RecordBatch.from_pylist(
            rows,
            schema=pa.schema(
                [
                    ("doc_id", pa.string()),
                    ("chunk_id", pa.string()),
                    ("user_id", pa.int64()),
                    ("vector", pa.list_(pa.float32())),
                ]
            ),
        )

    def close(self) -> None:  # pragma: no cover
        pass


class _Conn:
    def __init__(self, tables: Dict[str, _Table]) -> None:
        self.tables = tables

    def open_table(self, name: str) -> _Table:
        return self.tables[name]

    def list_tables(self) -> List[str]:
        return list(self.tables)


def _rows(items: List[Row]) -> List[Dict[str, Any]]:
    return [
        {
            "doc_id": doc,
            "chunk_id": chunk,
            "user_id": owner,
            "vector": [0.5, 0.5],
        }
        for doc, chunk, owner in items
    ]


def _check(
    monkeypatch: pytest.MonkeyPatch,
    source: Optional[List[Row]],
    target: Optional[List[Row]],
    *,
    skipped: Optional[Set[str]] = None,
    nulls: int = 0,
) -> mod.AcceptanceResult:
    tables: Dict[str, _Table] = {}
    if source is not None:
        tables["embeddings_old"] = _Table(_rows(source))
    if target is not None:
        tables["embeddings_new"] = _Table(_rows(target), nulls=nulls)
    monkeypatch.setattr(mod, "connection", lambda: _Conn(tables))
    return mod.check_collection(
        "kb",
        source_model_id="old",
        target_model_id="new",
        skipped=skipped,
    )


def test_matching_chunks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: List[Row] = [("a", "a1", 1), ("a", "a2", 1), ("b", "b1", 1)]

    result = _check(monkeypatch, rows, list(rows))

    assert result.ok is True
    assert result.checks["documents"] == 2
    assert result.checks["chunks"] == 3


def test_an_empty_collection_switches_without_a_target_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _check(monkeypatch, [], None)

    assert result.ok is True


def test_nothing_copied_for_a_non_empty_source_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _check(monkeypatch, [("a", "a1", 1)], None)

    assert result.ok is False
    assert result.checks["document_mismatch_count"] == 1


def test_a_missing_source_table_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _check(monkeypatch, None, [("a", "a1", 1)])

    assert result.ok is False
    assert "does not exist" in result.failures[0]


def test_ownerless_documents_are_left_out_of_both_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source: List[Row] = [("a", "a1", 1), ("legacy", "l1", None), ("legacy", "l2", None)]

    result = _check(monkeypatch, source, [("a", "a1", 1)])

    assert result.ok is True
    assert result.checks["ownerless_excluded"] == 1


def test_a_skipped_document_is_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _check(
        monkeypatch,
        [("a", "a1", 1), ("broken", "x1", 1)],
        [("a", "a1", 1)],
        skipped={"broken"},
    )

    assert result.ok is True


def test_a_chunk_missing_from_the_target_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _check(monkeypatch, [("a", "a1", 1), ("a", "a2", 1)], [("a", "a1", 1)])

    assert result.ok is False
    assert result.checks["document_mismatches"] == ["a"]


def test_an_extra_target_chunk_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _check(monkeypatch, [("a", "a1", 1)], [("a", "a1", 1), ("a", "a2", 1)])

    assert result.ok is False


def test_a_chunk_with_another_owner_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _check(monkeypatch, [("a", "a1", 1)], [("a", "a1", 2)])

    assert result.ok is False
    assert result.checks["document_mismatches"] == ["a"]


def test_a_null_vector_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _check(monkeypatch, [("a", "a1", 1)], [("a", "a1", 1)], nulls=1)

    assert result.ok is False
    assert result.checks["null_vectors"] == 1


def test_a_failed_read_raises_instead_of_passing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Broken(_Table):
        def to_batches(self, **kwargs: Any) -> Any:
            raise RuntimeError("corrupt table")

    monkeypatch.setattr(
        mod, "connection", lambda: _Conn({"embeddings_old": _Broken([])})
    )

    with pytest.raises(RuntimeError, match="corrupt table"):
        mod.check_collection("kb", source_model_id="old", target_model_id="new")


def test_a_null_vector_on_an_excluded_document_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source: List[Row] = [("a", "a1", 1), ("legacy", "l1", None)]
    tables = {
        "embeddings_old": _Table(_rows(source)),
        "embeddings_new": _Table(
            _rows([("a", "a1", 1), ("legacy", "l1", None)]), nulls=0
        ),
    }
    # Give the ownerless document a null vector under the target model.
    tables["embeddings_new"]._rows[1]["vector"] = None
    monkeypatch.setattr(mod, "connection", lambda: _Conn(tables))

    result = mod.check_collection("kb", source_model_id="old", target_model_id="new")

    assert result.ok is True
    assert result.checks["null_vectors"] == 0
