"""Main pointer management for version control.

This module provides functionality for managing main version pointers
across different processing stages (parse, chunk, embed).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ......providers.vector_store.lancedb import get_connection_from_env
from ..core.exceptions import MainPointerError
from ..LanceDB.schema_manager import ensure_main_pointers_table
from ..utils.string_utils import build_lancedb_filter_expression

logger = logging.getLogger(__name__)


def get_main_pointer(
    collection: str, doc_id: str, step_type: str, model_tag: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get the main pointer for a specific document and stage.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type (parse, chunk, embed)
        model_tag: Model tag for embed stage (optional)

    Returns:
        Main pointer data or None if not found

    Raises:
        MainPointerError: If there's an error retrieving the pointer
    """
    try:
        conn = get_connection_from_env()
        ensure_main_pointers_table(conn)

        table = conn.open_table("main_pointers")

        # Build safe filter conditions
        base_filters = {
            "collection": collection,
            "doc_id": doc_id,
            "step_type": step_type,
        }

        # Handle model_tag (supports both value and NULL)
        if model_tag is not None:
            base_filters["model_tag"] = model_tag
            filter_expr = build_lancedb_filter_expression(base_filters)
        else:
            filter_expr = build_lancedb_filter_expression(base_filters)
            filter_expr += " AND model_tag IS NULL"

        # Query the table
        result = table.search().where(filter_expr).to_pandas()

        if result.empty:
            return None

        # Return the first (and should be only) result
        row = result.iloc[0]
        return {
            "collection": row["collection"],
            "doc_id": row["doc_id"],
            "step_type": row["step_type"],
            "model_tag": row["model_tag"],
            "semantic_id": row["semantic_id"],
            "technical_id": row["technical_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "operator": row["operator"],
        }

    except Exception as e:
        raise MainPointerError(f"Failed to get main pointer: {e}")


def set_main_pointer(
    lancedb_dir: str,
    collection: str,
    doc_id: str,
    step_type: str,
    semantic_id: str,
    technical_id: str,
    model_tag: Optional[str] = None,
    operator: Optional[str] = None,
) -> None:
    """Set or update the main pointer for a specific document and stage.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type (parse, chunk, embed)
        semantic_id: Semantic identifier for the version
        technical_id: Technical identifier (hash) for the version
        model_tag: Model tag for embed stage (optional)
        operator: Operator who made the change (optional)

    Raises:
        MainPointerError: If there's an error setting the pointer
    """
    try:
        conn = get_connection_from_env()
        ensure_main_pointers_table(conn)

        table = conn.open_table("main_pointers")

        # Check if pointer already exists
        existing = get_main_pointer(collection, doc_id, step_type, model_tag)

        now = pd.Timestamp.now(tz="UTC")

        if existing:
            # Update existing pointer
            update_data = {
                "collection": [collection],
                "doc_id": [doc_id],
                "step_type": [step_type],
                "model_tag": [model_tag],
                "semantic_id": [semantic_id],
                "technical_id": [technical_id],
                "created_at": [existing["created_at"]],  # Keep original creation time
                "updated_at": [now],
                "operator": [operator or "unknown"],
            }

            # Delete old record and insert new one
            base_filters = {
                "collection": collection,
                "doc_id": doc_id,
                "step_type": step_type,
            }

            if model_tag is not None:
                base_filters["model_tag"] = model_tag
                delete_expr = build_lancedb_filter_expression(base_filters)
            else:
                delete_expr = build_lancedb_filter_expression(base_filters)
                delete_expr += " AND model_tag IS NULL"

            table.delete(delete_expr)

        else:
            # Create new pointer
            update_data = {
                "collection": [collection],
                "doc_id": [doc_id],
                "step_type": [step_type],
                "model_tag": [model_tag],
                "semantic_id": [semantic_id],
                "technical_id": [technical_id],
                "created_at": [now],
                "updated_at": [now],
                "operator": [operator or "unknown"],
            }

        # Insert the new/updated record
        df = pd.DataFrame(update_data)
        table.add(df)

        logger.info(
            f"Set main pointer for {collection}/{doc_id}/{step_type} to {technical_id} (semantic: {semantic_id})"
        )

    except Exception as e:
        raise MainPointerError(f"Failed to set main pointer: {e}")


def list_main_pointers(
    collection: str, doc_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List main pointers for a collection and optionally a specific document.

    Args:
        collection: Collection name
        doc_id: Document ID (optional, if None lists all documents in collection)

    Returns:
        List of main pointer data

    Raises:
        MainPointerError: If there's an error listing pointers
    """
    try:
        conn = get_connection_from_env()
        ensure_main_pointers_table(conn)

        table = conn.open_table("main_pointers")

        # Build safe filter conditions
        filters_dict = {"collection": collection}
        if doc_id is not None:
            filters_dict["doc_id"] = doc_id

        filter_expr = build_lancedb_filter_expression(filters_dict)

        # First check if any pointers exist using efficient count_rows
        if table.search().where(filter_expr).count_rows() == 0:
            return []

        # Only load data if pointers exist
        result = table.search().where(filter_expr).to_pandas()

        # Convert to list of dictionaries
        pointers = []
        for _, row in result.iterrows():
            pointers.append(
                {
                    "collection": row["collection"],
                    "doc_id": row["doc_id"],
                    "step_type": row["step_type"],
                    "model_tag": row["model_tag"],
                    "semantic_id": row["semantic_id"],
                    "technical_id": row["technical_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "operator": row["operator"],
                }
            )

        return pointers

    except Exception as e:
        raise MainPointerError(f"Failed to list main pointers: {e}")


def delete_main_pointer(
    collection: str, doc_id: str, step_type: str, model_tag: Optional[str] = None
) -> bool:
    """Delete a main pointer.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type
        model_tag: Model tag for embed stage (optional)

    Returns:
        True if pointer was deleted, False if not found

    Raises:
        MainPointerError: If there's an error deleting the pointer
    """
    try:
        conn = get_connection_from_env()
        ensure_main_pointers_table(conn)

        table = conn.open_table("main_pointers")

        # Build safe filter conditions
        base_filters = {
            "collection": collection,
            "doc_id": doc_id,
            "step_type": step_type,
        }

        # Handle model_tag (supports both value and NULL)
        if model_tag is not None:
            base_filters["model_tag"] = model_tag
            filter_expr = build_lancedb_filter_expression(base_filters)
        else:
            filter_expr = build_lancedb_filter_expression(base_filters)
            filter_expr += " AND model_tag IS NULL"

        # Check if pointer exists using count_rows for efficiency
        count = table.search().where(filter_expr).count_rows()
        if count == 0:
            return False

        # Delete the pointer
        table.delete(filter_expr)
        logger.info(f"Deleted main pointer for {collection}/{doc_id}/{step_type}")
        return True

    except Exception as e:
        raise MainPointerError(f"Failed to delete main pointer: {e}")
