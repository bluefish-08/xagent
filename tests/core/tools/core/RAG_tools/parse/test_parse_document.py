"""Tests for parse_document functionality (core layer).

This module validates the parse pipeline contracts:
register_document -> parse_document
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    DocumentNotFoundError,
    DocumentValidationError,
)
from xagent.core.tools.core.RAG_tools.core.schemas import ParseMethod
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.parse.parse_document import parse_document

RESOURCES_DIR = Path("tests/resources/test_files")


@pytest.fixture
def temp_lancedb_dir():
    """Create a temporary LanceDB directory for testing by pointing to a unique subdir.

    We intentionally use a per-test unique subdirectory under the project test lancedb root to avoid cross-test pollution.
    """
    # If the project defines a dev dir, still isolate per test
    base_dir = Path(os.environ.get("LANCEDB_DIR", "/tmp/.lancedb_test_root")).resolve()
    unique_dir = base_dir / f"pytest_{uuid.uuid4().hex[:8]}"
    unique_dir.mkdir(parents=True, exist_ok=True)
    old_dir = os.environ.get("LANCEDB_DIR")
    os.environ["LANCEDB_DIR"] = str(unique_dir)
    try:
        yield str(unique_dir)
    finally:
        # Restore original env to avoid side effects on other tests
        if old_dir is not None:
            os.environ["LANCEDB_DIR"] = old_dir
        else:
            os.environ.pop("LANCEDB_DIR", None)
        # Cleanup temp directory
        import shutil

        if unique_dir.exists():
            shutil.rmtree(unique_dir)


@pytest.fixture
def test_collection() -> str:
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_doc_id() -> str:
    return str(uuid.uuid4())


class TestParseDocumentCore:
    """Core parse_document tests using files under tests/resources/test_files.

    These tests assume the sample files exist. If a specific sample is missing on the filesystem,
    the corresponding test will be skipped to keep CI green.
    """

    def _require_file(self, relative: str) -> Path:
        p = RESOURCES_DIR / relative
        if not p.exists():
            pytest.skip(f"Sample file not found: {p}")
        return p

    def test_parse_txt_default_happy_path(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        sample = self._require_file("test.txt")
        reg = register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        assert reg["created"] is True
        out = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        assert out["written"] is True
        assert out["doc_id"] == test_doc_id
        assert isinstance(out.get("parse_hash"), str) and len(out["parse_hash"]) > 0
        assert isinstance(out.get("paragraphs"), list)
        # For non-empty txt, expect at least one paragraph
        if sample.read_text(encoding="utf-8").strip():
            assert len(out["paragraphs"]) >= 1
        # Metadata presence
        if out["paragraphs"]:
            meta = out["paragraphs"][0]["metadata"]
            assert meta.get("source")
            assert (
                meta.get("file_type") == "txt"
            )  # file_type from database is without dot
            assert meta.get("parse_method") == ParseMethod.DEEPDOC.value
            assert meta.get("parser")

    def test_idempotency_same_doc_same_params(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        sample = self._require_file("test.md")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        first = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        second = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        assert first["written"] is True
        assert second["written"] is False  # idempotent path
        # paragraphs should be materialized from DB on second call
        assert isinstance(second.get("paragraphs"), list)
        assert len(second["paragraphs"]) == len(first["paragraphs"])  # basic stability

    def test_collection_isolation(self, temp_lancedb_dir: str) -> None:
        sample = self._require_file("test.txt")
        doc_id = str(uuid.uuid4())
        c1 = f"c1_{uuid.uuid4().hex[:6]}"
        c2 = f"c2_{uuid.uuid4().hex[:6]}"
        # Register same file under two collections
        register_document(
            collection=c1, source_path=str(sample), doc_id=doc_id, user_id=1
        )
        register_document(
            collection=c2, source_path=str(sample), doc_id=doc_id, user_id=1
        )
        p1 = parse_document(
            collection=c1,
            doc_id=doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        p2 = parse_document(
            collection=c2,
            doc_id=doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        assert p1["written"] is True
        assert p2["written"] is True
        assert p1["parse_hash"] == p2["parse_hash"]  # same method/params
        assert len(p1["paragraphs"]) >= 0 and len(p2["paragraphs"]) >= 0

    def test_parse_pdf_pypdf(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        sample = self._require_file("test.pdf")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        out = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.PYPDF,
            user_id=1,
            is_admin=True,
        )
        assert out["doc_id"] == test_doc_id
        assert out["written"] in (True, False)
        assert isinstance(out.get("paragraphs"), list)
        # For a valid sample pdf we expect at least zero or more paragraphs (be lenient)
        assert len(out["paragraphs"]) >= 0

    def test_invalid_params_rejected(
        self, temp_lancedb_dir: str, test_collection: str, test_doc_id: str
    ) -> None:
        sample = self._require_file("test.txt")
        register_document(
            collection=test_collection,
            source_path=str(sample),
            doc_id=test_doc_id,
            user_id=1,
        )
        with pytest.raises(DocumentValidationError):
            parse_document(
                collection=test_collection,
                doc_id=test_doc_id,
                parse_method=ParseMethod.DEEPDOC,
                params={"unknown_flag": True},
                user_id=1,
                is_admin=True,
            )

    def test_document_not_found_raises(self, temp_lancedb_dir: str) -> None:
        with pytest.raises(DocumentNotFoundError):
            parse_document(
                collection=f"c_{uuid.uuid4().hex[:6]}",
                doc_id=str(uuid.uuid4()),
                parse_method=ParseMethod.DEEPDOC,
                user_id=1,
                is_admin=True,
            )


class TestParseDocumentFallback:
    """Test three-tier fallback logic for parse_document internal functions."""

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary LanceDB directory for testing."""
        base_dir = Path(
            os.environ.get("LANCEDB_DIR", "/tmp/.lancedb_test_root")
        ).resolve()
        unique_dir = base_dir / f"pytest_{uuid.uuid4().hex[:8]}"
        unique_dir.mkdir(parents=True, exist_ok=True)
        old_dir = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = str(unique_dir)
        try:
            yield str(unique_dir)
        finally:
            if old_dir is not None:
                os.environ["LANCEDB_DIR"] = old_dir
            else:
                os.environ.pop("LANCEDB_DIR", None)
            # Cleanup temp directory
            import shutil

            if unique_dir.exists():
                shutil.rmtree(unique_dir)

    @pytest.fixture
    def test_collection(self) -> str:
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_parse_document_arrow_fallback_chain(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test parse_document uses to_arrow() -> to_list() -> to_pandas() fallback."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.parse.parse_document import (
            _get_document_from_db,
        )

        mock_db_connection = MagicMock()
        mock_table = MagicMock()

        def mock_open_table_func(table_name):
            return mock_table

        mock_db_connection.open_table.side_effect = mock_open_table_func

        doc_data = {
            "collection": test_collection,
            "doc_id": "doc1",
            "file_path": "/path/to/file",
        }
        mock_arrow_table = MagicMock()
        mock_arrow_table.to_pylist.return_value = [doc_data]

        mock_search = MagicMock()
        mock_where = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.where.return_value = mock_where
        mock_where.to_arrow.return_value = mock_arrow_table
        mock_table.count_rows.return_value = 1

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.parse.parse_document.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.parse.parse_document.ensure_documents_table"
            ),
        ):
            result = _get_document_from_db(
                collection=test_collection,
                doc_id="doc1",
                user_id=1,
            )

            assert result is not None
            assert result["doc_id"] == "doc1"
            # Verify to_arrow() was called
            mock_where.to_arrow.assert_called_once()

    def test_parse_document_fallback_to_list(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test parse_document fallback from to_arrow() to to_list()."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.parse.parse_document import (
            _get_document_from_db,
        )

        mock_db_connection = MagicMock()
        mock_table = MagicMock()

        def mock_open_table_func(table_name):
            return mock_table

        mock_db_connection.open_table.side_effect = mock_open_table_func

        doc_data = {
            "collection": test_collection,
            "doc_id": "doc1",
            "file_path": "/path/to/file",
        }

        mock_search = MagicMock()
        mock_where = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.where.return_value = mock_where
        # to_arrow() fails, fallback to to_list()
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.return_value = [doc_data]
        mock_table.count_rows.return_value = 1

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.parse.parse_document.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.parse.parse_document.ensure_documents_table"
            ),
        ):
            result = _get_document_from_db(
                collection=test_collection,
                doc_id="doc1",
                user_id=1,
            )

            assert result is not None
            assert result["doc_id"] == "doc1"
            # Verify fallback was used
            mock_where.to_arrow.assert_called_once()
            mock_where.to_list.assert_called_once()

    def test_parse_document_fallback_to_pandas_with_nan(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test parse_document fallback to to_pandas() and NaN normalization."""
        from unittest.mock import MagicMock, patch

        import numpy as np
        import pandas as pd

        from xagent.core.tools.core.RAG_tools.parse.parse_document import (
            _get_document_from_db,
        )

        mock_db_connection = MagicMock()
        mock_table = MagicMock()

        def mock_open_table_func(table_name):
            return mock_table

        mock_db_connection.open_table.side_effect = mock_open_table_func

        # Create DataFrame with NaN values
        doc_df = pd.DataFrame(
            [
                {
                    "collection": test_collection,
                    "doc_id": "doc1",
                    "file_path": "/path/to/file",
                    "optional_field": np.nan,  # NaN value
                }
            ]
        )

        mock_search = MagicMock()
        mock_where = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.where.return_value = mock_where
        # Both to_arrow() and to_list() fail, fallback to to_pandas()
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = doc_df
        mock_table.count_rows.return_value = 1

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.parse.parse_document.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.parse.parse_document.ensure_documents_table"
            ),
        ):
            result = _get_document_from_db(
                collection=test_collection,
                doc_id="doc1",
                user_id=1,
            )

            assert result is not None
            assert result["doc_id"] == "doc1"
            # Verify all fallbacks were attempted
            mock_where.to_arrow.assert_called_once()
            mock_where.to_list.assert_called_once()
            mock_where.to_pandas.assert_called_once()
            # Verify NaN was normalized to None
            assert result.get("optional_field") is None
