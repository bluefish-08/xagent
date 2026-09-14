"""#671 - the two filter evaluators must agree on the same condition.

Operator semantics live twice: ``lancedb_filter_utils.translate_condition``
renders SQL for the indexed paths, and ``collection_handle._single_condition_mask``
evaluates pandas for the scan fallbacks. Nothing forced them to agree, and every
divergence this PR fixed was a symptom of that. This runs both against one real
LanceDB table and compares the rows they select.
"""

from __future__ import annotations

from typing import Any

import lancedb
import pandas as pd
import pytest

from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    _single_condition_mask,
)
from xagent.core.tools.core.RAG_tools.storage.contracts import (
    FilterCondition,
    FilterOperator,
)
from xagent.core.tools.core.RAG_tools.storage.lancedb_filter_utils import (
    translate_filter_expression,
)
from xagent.core.tools.core.RAG_tools.utils.filter_utils import (
    normalize_filter_conditions,
)

ROWS = [
    {"doc_id": "d1", "n": 1, "label": "a"},
    {"doc_id": "d2", "n": 2, "label": None},
    {"doc_id": "d3", "n": 3, "label": "b%c"},
    # A "b%c" needle reaches these two only as a wildcard; "b_c" reaches d4
    # only as one, and d5 literally.
    {"doc_id": "d4", "n": 4, "label": "bXc"},
    {"doc_id": "d5", "n": 5, "label": "b_c"},
]

# Filter shapes a caller can actually write, one per operator the dict form
# reaches. The CONTAINS needles carry LIKE metacharacters, which the backend
# would otherwise read as wildcards while a scan matches them literally.
PARITY_CASES = [
    ("eq_scalar", {"doc_id": "d1"}),
    ("shorthand_list", {"doc_id": ["d1", "d3"]}),
    ("eq_with_list", {"doc_id": {"operator": "eq", "value": ["d1", "d3"]}}),
    ("explicit_in", {"doc_id": {"operator": "in", "value": ["d2"]}}),
    ("gt", {"n": {"operator": "gt", "value": 1}}),
    ("gte", {"n": {"operator": "gte", "value": 2}}),
    ("lt", {"n": {"operator": "lt", "value": 3}}),
    ("lte", {"n": {"operator": "lte", "value": 2}}),
    ("ne_scalar", {"label": {"operator": "ne", "value": "a"}}),
    ("contains", {"label": {"operator": "contains", "value": "b"}}),
    ("contains_percent", {"label": {"operator": "contains", "value": "b%c"}}),
    ("contains_underscore", {"label": {"operator": "contains", "value": "b_c"}}),
    (
        "is_null",
        FilterCondition(field="label", operator=FilterOperator.IS_NULL, value=None),
    ),
    (
        "is_not_null",
        FilterCondition(field="label", operator=FilterOperator.IS_NOT_NULL, value=None),
    ),
    (
        "ne_null",
        FilterCondition(field="label", operator=FilterOperator.NE, value=None),
    ),
]


@pytest.fixture(scope="module")
def real_table(tmp_path_factory: pytest.TempPathFactory) -> Any:
    connection = lancedb.connect(str(tmp_path_factory.mktemp("parity")))
    return connection.create_table("rows", data=ROWS)


@pytest.mark.parametrize(
    ("case", "filters"), PARITY_CASES, ids=[row[0] for row in PARITY_CASES]
)
def test_both_evaluators_select_the_same_rows(
    real_table: Any, case: str, filters: Any
) -> None:
    (condition,) = normalize_filter_conditions(filters)
    frame = pd.DataFrame(ROWS)

    backend_rows = sorted(
        real_table.search()
        .where(translate_filter_expression(condition))
        .limit(100)
        .to_pandas()["doc_id"]
    )
    scan_rows = sorted(frame[_single_condition_mask(frame, condition)]["doc_id"])

    assert backend_rows == scan_rows, (
        f"{case}: backend selected {backend_rows}, scan selected {scan_rows}"
    )


def test_an_operator_that_cannot_carry_a_collection_is_rejected() -> None:
    """There is no NOT IN, so `ne` with a list cannot agree on both paths."""
    with pytest.raises(ValueError, match="does not accept a collection"):
        normalize_filter_conditions({"doc_id": {"operator": "ne", "value": ["d1"]}})
