"""
Index management for vector storage.

This module provides centralized index management functionality for embeddings tables,
including index creation, status checking, and basic maintenance operations.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ..core.config import IndexPolicy
from ..core.schemas import IndexType

# Import LanceDB index types
try:
    from lancedb.index import IVF_HNSW_SQ, IVF_PQ  # type: ignore
except ImportError:
    # Fallback if import fails
    IVF_HNSW_SQ = "IVF_HNSW_SQ"
    IVF_PQ = "IVF_PQ"

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Centralized index manager for embeddings tables.

    This class handles index lifecycle management including creation,
    status checking, and basic maintenance operations.
    """

    def __init__(self, policy: Optional[IndexPolicy] = None):
        """
        Initialize index manager with policy configuration.

        Args:
            policy: Index policy configuration, uses default if None
        """
        self.policy = policy or IndexPolicy()

    def check_and_create_index(
        self,
        table: Any,
        table_name: str,
        readonly: bool = False,
    ) -> Tuple[str, Optional[str]]:
        """
        Check table index status and create index if needed.

        Automatically selects index type based on row count:
        - < 50k rows: No index
        - 50k-10M rows: HNSW index
        - >= 10M rows: IVFPQ index

        Args:
            table: LanceDB table instance
            table_name: Table name for logging
            readonly: If True, don't create indexes

        Returns:
            Tuple of (index_status, index_advice)
        """
        if readonly:
            return "readonly", f"Readonly mode - no index operations for {table_name}"

        vector_index_status: str = "no_index"
        vector_index_advice: Optional[str] = None

        try:
            # Get row count efficiently without loading all data into memory
            row_count = table.count_rows()

            if row_count < self.policy.enable_threshold_rows:
                vector_index_status = "below_threshold"
                vector_index_advice = (
                    f"Table {table_name} has {row_count} rows - below threshold "
                    f"({self.policy.enable_threshold_rows}) for index creation"
                )
            else:
                # Auto-select index type based on scale
                if row_count >= self.policy.ivfpq_threshold_rows:
                    recommended_type = IndexType.IVFPQ
                else:
                    recommended_type = IndexType.HNSW

                # Check existing indexes
                indexes = table.list_indices()
                has_vector_index = any(idx.name == "vector" for idx in indexes)

                if not has_vector_index:
                    # Create index with recommended type and parameters
                    if recommended_type == IndexType.IVFPQ:
                        index_type = IVF_PQ
                        create_params = self.policy.ivfpq_params or {}
                    else:  # HNSW
                        index_type = IVF_HNSW_SQ
                        create_params = self.policy.hnsw_params or {}

                    # Merge metric with create_params, avoiding duplicates
                    all_params = {
                        "metric": self.policy.metric.value,
                        "index_type": index_type,
                        **create_params,
                    }

                    table.create_index(**all_params)
                    vector_index_status = "index_building"
                    logger.info(
                        "Successfully created vector index for %s (type=%s, metric=%s)",
                        table_name,
                        index_type,
                        self.policy.metric.value,
                    )
                    if recommended_type == IndexType.IVFPQ:
                        vector_index_advice = (
                            f"IVFPQ index created for {table_name} "
                            f"({row_count} rows, using IVFPQ strategy for large-scale data), metric: {self.policy.metric.value}"
                        )
                    else:  # HNSW
                        vector_index_advice = (
                            f"HNSW index created for {table_name} "
                            f"({row_count} rows, using HNSW strategy for medium-scale data), metric: {self.policy.metric.value}"
                        )
                else:
                    vector_index_status = "index_ready"
                    vector_index_advice = f"Index ready for {table_name} ({row_count} rows), metric: {self.policy.metric.value}"

        except Exception as e:
            logger.error(f"Vector index operation failed for {table_name}: {str(e)}")
            vector_index_status = "index_corrupted"
            vector_index_advice = (
                f"Vector index check failed for {table_name}: {str(e)}"
            )

        # FTS Index Management
        if self.policy.fts_enabled:
            fts_success, fts_message = self.create_fts_index(
                table, table_name, self.policy.fts_params
            )
            if not fts_success:
                logger.warning(
                    f"FTS index creation/check failed for {table_name}: {fts_message}"
                )
                # If FTS index fails, it does not necessarily corrupt the vector index
                # but we should reflect the partial failure or warning.
                # For now, we will log and return vector index status primarily.

        return vector_index_status, vector_index_advice

    def get_index_status(self, table: Any) -> str:
        """
        Get current index status for a table.

        Args:
            table: LanceDB table instance

        Returns:
            Index status string
        """
        try:
            indexes = table.list_indices()
            has_vector_index = any(idx.name == "vector" for idx in indexes)

            if has_vector_index:
                return "index_ready"
            else:
                row_count = table.count_rows()
                if row_count >= self.policy.enable_threshold_rows:
                    return "no_index"
                else:
                    return "below_threshold"
        except Exception as e:
            logger.error(f"Failed to get index status: {str(e)}")
            return "index_corrupted"

    def get_fts_index_status(self, table: Any) -> bool:
        """
        Check if a Full-Text Search (FTS) index exists on the 'text' column of the table.

        Args:
            table: LanceDB table instance.

        Returns:
            True if an FTS index exists on the 'text' column, False otherwise.
        """
        try:
            indexes = table.list_indices()
            # New lancedb versions return IndexConfig objects, not dicts.
            # Access properties via attributes.
            return any(
                idx.index_type == "FTS" and "text" in idx.columns for idx in indexes
            )
        except Exception as e:
            logger.error(f"Failed to check FTS index status for {table.name}: {str(e)}")
            return False

    def create_fts_index(
        self,
        table: Any,
        table_name: str,
        fts_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Create a Full-Text Search (FTS) index on the 'text' column.

        Args:
            table: LanceDB table instance.
            table_name: Name of the table for logging.
            fts_params: Optional dictionary of FTS parameters (e.g., language, stem, ascii_folding, with_position).

        Returns:
            Tuple of (success: bool, message: Optional[str]).
        """
        if self.get_fts_index_status(table):
            return True, f"FTS index already exists on 'text' column for {table_name}"

        try:
            # Default FTS parameters, can be overridden by fts_params
            _fts_params = {"with_position": True, **(fts_params or {})}
            # Add replace=True to make the operation idempotent
            table.create_fts_index("text", replace=True, **_fts_params)
            logger.info(
                "Successfully created FTS index on 'text' column for %s", table_name
            )
            return (
                True,
                f"FTS index created on 'text' column for {table_name} with params: {_fts_params}",
            )
        except Exception as e:
            logger.error(f"Failed to create FTS index for {table_name}: {str(e)}")
            return False, f"Failed to create FTS index: {str(e)}"


# Global index manager instance
_default_index_manager: Optional[IndexManager] = None


def get_index_manager(policy: Optional[IndexPolicy] = None) -> IndexManager:
    """
    Get the global index manager instance.

    Args:
        policy: Optional policy to configure the manager

    Returns:
        IndexManager instance
    """
    global _default_index_manager

    if _default_index_manager is None or (policy is not None):
        _default_index_manager = IndexManager(policy)

    return _default_index_manager
