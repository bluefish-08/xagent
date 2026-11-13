"""Tests for the minimal acceptance check of the embedding switch."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pytest

from xagent.migrations.embedding import acceptance as mod
from xagent.migrations.embedding.scope import (
    SKIP_NO_OWNER,
    SKIP_NO_PARSE_HASH,
    SKIP_NO_SUCCESSFUL_INGEST,
)

Row = Tuple[str, Optional[int]]


class _Store:
    def __init__(self, tables: Dict[str, List[Row]], nulls: int = 0) -> None:
        self._tables = tables
        self._nulls = nulls

    def iter_batches(self, **kwargs: Any) -> Any:
        rows = self._tables[kwargs["table_name"]]
        assert kwargs["is_admin"] is True and kwargs["user_id"] is None
        if kwargs["columns"] == ["vector"]:
            vectors = [None] * self._nulls + [[0.0]] * (len(rows) - self._nulls)
            yield pa.RecordBatch.from_pydict(
                {"vector": pa.array(vectors, type=pa.list_(pa.float32()))}
            )
            return
        yield pa.RecordBatch.from_pylist(
            [{"doc_id": doc_id, "user_id": user_id} for doc_id, user_id in rows]
        )


def _check(
    monkeypatch: pytest.MonkeyPatch,
    source: List[Row],
    target: List[Row],
    *,
    skipped: Optional[Dict[str, str]] = None,
    nulls: int = 0,
) -> mod.AcceptanceResult:
    store = _Store({"old": source, "new": target}, nulls=nulls)
    monkeypatch.setattr(mod, "get_vector_index_store", lambda: store)
    monkeypatch.setattr(
        mod, "embeddings_table", lambda model: {"m-old": "old", "m-new": "new"}[model]
    )
    return mod.check_collection(
        "kb", source_model_id="m-old", target_model_id="m-new", skipped=skipped
    )


def test_matching_documents_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: List[Row] = [("a", 1), ("a", 1), ("b", 2)]

    result = _check(monkeypatch, rows, list(rows))

    assert result.ok is True
    assert result.checks["source_count"] == 3
    assert result.checks["target_count"] == 3


def test_ownerless_documents_are_left_out_of_both_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _check(
        monkeypatch,
        [("a", 1), ("legacy", None), ("legacy", None)],
        [("a", 1)],
        skipped={"legacy": SKIP_NO_OWNER},
    )

    assert result.ok is True
    assert result.uncovered == []
    assert result.checks["ownerless_excluded"] == 1
    assert result.checks["source_count"] == 1


@pytest.mark.parametrize("reason", [SKIP_NO_SUCCESSFUL_INGEST, SKIP_NO_PARSE_HASH])
def test_a_skipped_document_without_old_rows_is_left_out(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    result = _check(
        monkeypatch, [("a", 1)], [("a", 1), ("failed", 1)], skipped={"failed": reason}
    )

    assert result.ok is True
    assert result.uncovered == []
    assert result.checks["excluded"] == 1


@pytest.mark.parametrize("reason", [SKIP_NO_SUCCESSFUL_INGEST, SKIP_NO_PARSE_HASH])
def test_a_skipped_document_the_old_model_still_answers_for_is_uncovered(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    result = _check(
        monkeypatch, [("a", 1), ("failed", 1)], [("a", 1)], skipped={"failed": reason}
    )

    assert result.ok is False
    assert result.uncovered == ["failed"]


def test_a_document_missing_from_the_new_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _check(monkeypatch, [("a", 1), ("b", 1)], [("a", 1)])

    assert result.ok is False
    assert result.checks["document_mismatches"] == ["b"]
    assert result.uncovered == []


def test_a_document_with_another_owner_fails_even_when_totals_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _check(monkeypatch, [("a", 1), ("b", 2)], [("a", 1), ("b", 3)])

    assert result.ok is False
    assert result.checks["source_count"] == result.checks["target_count"]
    assert result.checks["document_mismatches"] == ["b"]


def test_rows_moved_between_documents_fail_even_when_totals_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _check(monkeypatch, [("a", 1), ("b", 1)], [("a", 1), ("a", 1)])

    assert result.ok is False
    assert result.checks["document_mismatch_count"] == 2


def test_null_vectors_in_the_new_model_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _check(monkeypatch, [("a", 1)], [("a", 1)], nulls=1)

    assert result.ok is False
    assert result.checks["null_vectors"] == 1


def test_a_missing_table_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "embeddings_table", lambda model: None)

    result = mod.check_collection("kb", source_model_id="a", target_model_id="b")

    assert result.ok is False
    assert "missing embeddings table" in result.failures[0]
