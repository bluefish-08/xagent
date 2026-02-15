"""LanceDB query utilities for RAG tools.

This module provides unified query functions for LanceDB operations,
implementing a three-tier fallback pattern for maximum compatibility.
"""

import logging
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa  # type: ignore

logger = logging.getLogger(__name__)


def query_to_list(
    query: Any,
    normalize_nan: bool = True,
) -> List[Dict[str, Any]]:
    """Convert LanceDB query result to List[Dict] with three-tier fallback.

    This function implements a performance-optimized fallback chain:
    1. to_arrow() (fastest, native format)
    2. to_list() (fast, direct List[Dict])
    3. to_pandas() (compatibility fallback)

    Args:
        query: LanceDB query object that supports to_arrow/to_list/to_pandas methods.
            Can be:
            - table.search().where(filter_expr)
            - search_query (pre-built query object)
            - table.head(n)
            - table.search().where().select()
        normalize_nan: Whether to convert pandas NaN values to None for consistent handling.

    Returns:
        List[Dict[str, Any]]: Query results as a list of dictionaries.
            Empty list indicates no results found.

    Example:
        >>> from ..utils.lancedb_query_utils import query_to_list
        >>> results = query_to_list(table.search().where(filter_expr))
        >>> if results:
        ...     record = results[0]
    """
    results: List[Dict[str, Any]] = []

    # Check if query is already a PyArrow Table (e.g., from table.head())
    if pa is not None and isinstance(query, pa.Table):
        # PyArrow Table can directly convert to list
        results = query.to_pylist()
        return results

    try:
        # First choice: Arrow (fastest, native format)
        arrow_table = query.to_arrow()
        results = arrow_table.to_pylist()
    except Exception as e:  # noqa: BLE001 - Catch all exceptions to ensure fallback works
        logger.debug("to_arrow() failed (will try fallback): %s", e)
        try:
            # Second choice: to_list() (fast, direct List[Dict])
            results = query.to_list()
        except Exception as e:  # noqa: BLE001 - Catch all exceptions to ensure fallback works
            # Last resort: pandas (compatibility fallback)
            logger.debug(
                "to_list() failed (will try pandas fallback): %s. Falling back to to_pandas()",
                e,
            )
            df = query.to_pandas()
            results = df.to_dict("records")

            # Normalize NaN to None for consistent handling (pandas may return NaN)
            # Only check scalar values to avoid errors with array/list fields (e.g., vector)
            if normalize_nan:
                for row in results:
                    for key, value in row.items():
                        # Only check NaN for scalar values, skip arrays/lists
                        if pd.api.types.is_scalar(value) and pd.isna(value):
                            row[key] = None

    return results
