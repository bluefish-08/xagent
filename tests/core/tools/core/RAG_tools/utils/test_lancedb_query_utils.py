"""Tests for the LanceDB FTS query builder."""

from __future__ import annotations

import pytest

from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import build_fts_query


def _terms(query_text: str, text_column: str = "text"):
    built = build_fts_query(query_text, text_column)
    assert built is not None
    return [(match.query, match.column) for _, match in built.queries]


def test_build_fts_query_splits_on_whitespace_and_cjk_punctuation():
    assert _terms("点击 Save 按钮，然后审批") == [
        ("点击", "text"),
        ("Save", "text"),
        ("按钮", "text"),
        ("然后审批", "text"),
    ]


@pytest.mark.parametrize("query_text", ["COVID-19", "GPT-4o", "C++", "3.5", "it's"])
def test_build_fts_query_keeps_ascii_punctuation_inside_a_term(query_text: str):
    """These are single tokens in the index; splitting them makes the query miss."""
    assert _terms(query_text) == [(query_text, "text")]


def test_build_fts_query_keeps_single_term_and_column():
    assert _terms("incident", "body") == [("incident", "body")]


@pytest.mark.parametrize("query_text", ["", " ", "   ", " ,. ", "，。", "😀"])
def test_build_fts_query_returns_none_without_terms(query_text: str):
    """No terms means no FTS query: the space token matches every chunk holding one."""
    assert build_fts_query(query_text) is None
