"""Real-LanceDB checks that the FTS query builder is index-agnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import lancedb
import pytest

from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import build_fts_query

DOCS = [
    {"id": 0, "text": "To print an incident report, open Incidents then Tickets."},
    {"id": 1, "text": "Timesheets are approved by the manager every Friday."},
    {"id": 2, "text": "Ticks and mites are common in rural areas."},
    {"id": 3, "text": "事故报告可以在工单页面下载 PDF 文件并打印。"},
]
# Single words, so every tokenizer -- including the legacy ngram one -- agrees
# on them; multi-word queries are where the builder deliberately differs.
QUERIES = ["incident", "Tickets", "manager", "Timesheets", "nonexistent"]


def _table(tmp_path: Path, name: str, **index_params: Any) -> Any:
    table = lancedb.connect(str(tmp_path / name)).create_table(name, DOCS)
    table.create_fts_index("text", replace=True, with_position=True, **index_params)
    return table


def _hits(table: Any, query: Any) -> List[int]:
    return sorted(
        x["id"] for x in table.search(query, query_type="fts").limit(9).to_list()
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "index_params",
    [
        {"base_tokenizer": "simple"},
        {"base_tokenizer": "jieba/default"},
        # The parameters every index built before this change still carries.
        {
            "base_tokenizer": "ngram",
            "ngram_min_length": 2,
            "prefix_only": True,
        },
    ],
    ids=["simple", "jieba", "legacy-ngram"],
)
def test_built_query_matches_the_raw_string_on_any_index(
    tmp_path: Path, index_params: dict
):
    """Word queries return the same rows whichever tokenizer the index was built with."""
    table = _table(tmp_path, "t", **index_params)

    for query_text in QUERIES:
        built = build_fts_query(query_text)
        assert built is not None
        assert _hits(table, built) == _hits(table, query_text), query_text
