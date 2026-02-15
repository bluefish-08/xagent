"""Tests for main_pointer_manager functions.

These tests mock the LanceDB connection returned by get_connection_from_env
to validate basic CRUD behaviors without touching real storage.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager import (
    delete_main_pointer,
    get_main_pointer,
    list_main_pointers,
    set_main_pointer,
)


class TestMainPointerManager:
    """Test cases for main_pointer_manager functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures."""
        # Restore original environment
        if self.original_env is not None:
            os.environ["LANCEDB_DIR"] = self.original_env
        elif "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_get_main_pointer_not_found(self, mock_get_conn: MagicMock) -> None:
        conn = MagicMock()
        table = MagicMock()
        table.search.return_value.where.return_value.to_pandas.return_value = (
            pd.DataFrame()
        )
        conn.open_table.return_value = table
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        assert get_main_pointer("c", "d", "parse") is None

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_set_and_get_main_pointer_roundtrip(self, mock_get_conn: MagicMock) -> None:
        conn = MagicMock()
        table = MagicMock()

        # First get returns empty, then returns a single row after set
        empty_df = pd.DataFrame()
        row_df = pd.DataFrame(
            [
                {
                    "collection": "c",
                    "doc_id": "d",
                    "step_type": "parse",
                    "model_tag": None,
                    "semantic_id": "parse_x",
                    "technical_id": "abc",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "operator": "tester",
                }
            ]
        )

        # Configure search.where.to_pandas to return empty then row
        table.search.return_value.where.return_value.to_pandas.side_effect = [
            empty_df,
            row_df,
            row_df,
        ]
        conn.open_table.return_value = table
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        # set should add once
        set_main_pointer(
            self.temp_dir,  # Pass lancedb_dir
            "c",
            "d",
            "parse",
            semantic_id="parse_x",
            technical_id="abc",
            operator="tester",
        )
        table.add.assert_called_once()

        # get should return the row
        result = get_main_pointer("c", "d", "parse")
        assert result is not None and result["technical_id"] == "abc"

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_list_and_delete_main_pointers(self, mock_get_conn: MagicMock) -> None:
        conn = MagicMock()
        table = MagicMock()
        df = pd.DataFrame(
            [
                {
                    "collection": "c",
                    "doc_id": "d",
                    "step_type": "parse",
                    "model_tag": None,
                    "semantic_id": "parse_x",
                    "technical_id": "abc",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "operator": "tester",
                }
            ]
        )
        table.search.return_value.where.return_value.to_pandas.return_value = df
        table.search.return_value.where.return_value.count_rows.return_value = 1
        conn.open_table.return_value = table
        conn.table_names.return_value = ["main_pointers"]
        mock_get_conn.return_value = conn

        rows = list_main_pointers("c", doc_id="d")
        assert len(rows) == 1
        row = rows[0]
        assert row["collection"] == "c"
        assert row["doc_id"] == "d"
        assert row["step_type"] == "parse"
        assert row["technical_id"] == "abc"
        assert row["semantic_id"] == "parse_x"

        deleted = delete_main_pointer("c", "d", "parse")
        assert deleted is True
        table.delete.assert_called_once()

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_get_main_pointer_injection_attack_prevention(self, mock_get_conn):
        """Test that SQL injection attacks are properly prevented in get_main_pointer.

        Verifies that special characters in collection, doc_id, step_type, and model_tag
        are properly escaped to prevent SQL injection attacks.
        """
        from xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager import (
            get_main_pointer,
        )

        # Malicious inputs attempting SQL injection
        malicious_collection = "coll'; DROP TABLE main_pointers; --"
        malicious_doc_id = "doc' OR '1'='1"
        malicious_step_type = "parse' OR 'a'='a"
        malicious_model_tag = "model'; DELETE FROM main_pointers; --"

        conn = MagicMock()
        conn.table_names.return_value = ["main_pointers"]

        # Mock table that records the filter expression used
        table = MagicMock()
        captured_filter = []

        def capture_where(filter_expr: str):
            captured_filter.append(filter_expr)
            mock_result = MagicMock()
            # Return empty result to avoid processing
            mock_result.to_pandas.return_value = pd.DataFrame()
            return mock_result

        table.search.return_value.where.side_effect = capture_where
        conn.open_table.return_value = table
        mock_get_conn.return_value = conn

        # Execute with malicious inputs
        get_main_pointer(
            malicious_collection,
            malicious_doc_id,
            malicious_step_type,
            model_tag=malicious_model_tag,
        )

        # Assert: The filter expression should have properly escaped all inputs
        assert len(captured_filter) > 0
        filter_expr = captured_filter[0]

        # Check that single quotes are escaped (doubled)
        assert "coll''; DROP TABLE main_pointers; --'" in filter_expr
        assert "doc'' OR ''1''=''1'" in filter_expr
        assert "parse'' OR ''a''=''a'" in filter_expr
        assert "model''; DELETE FROM main_pointers; --'" in filter_expr

        # The filter should NOT contain unescaped malicious SQL
        assert "'; DROP TABLE" not in filter_expr.replace("'';", "")
        assert "' OR '1'='1" not in filter_expr.replace("''", "")

    @patch(
        "xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager.get_connection_from_env"
    )
    def test_list_main_pointers_injection_attack_prevention(self, mock_get_conn):
        """Test that SQL injection attacks are properly prevented in list_main_pointers.

        Verifies that special characters in collection and doc_id are properly escaped.
        """
        from xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager import (
            list_main_pointers,
        )

        malicious_collection = "test'; DROP TABLE main_pointers; --"
        malicious_doc_id = "doc' OR 1=1; --"

        conn = MagicMock()
        conn.table_names.return_value = ["main_pointers"]

        # Mock table that records the filter expression
        table = MagicMock()
        captured_filter = []

        def capture_where(filter_expr: str):
            captured_filter.append(filter_expr)
            mock_result = MagicMock()
            mock_result.to_pandas.return_value = pd.DataFrame()
            return mock_result

        table.search.return_value.where.side_effect = capture_where
        conn.open_table.return_value = table
        mock_get_conn.return_value = conn

        # Execute with malicious inputs
        list_main_pointers(malicious_collection, doc_id=malicious_doc_id)

        # Assert: Filter expression should have escaped inputs
        assert len(captured_filter) > 0
        filter_expr = captured_filter[0]

        # Check proper escaping
        assert "test''; DROP TABLE main_pointers; --'" in filter_expr
        assert "doc'' OR 1=1; --'" in filter_expr

        # Should NOT contain unescaped malicious SQL
        assert "'; DROP TABLE" not in filter_expr.replace("'';", "")
        assert "' OR 1=1" not in filter_expr.replace("''", "")
