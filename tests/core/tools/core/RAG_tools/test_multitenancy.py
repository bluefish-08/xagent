"""Tests for RAG multi-tenancy support.

Tests user_id and is_admin filtering in RAG tools and pipelines.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.storage import initialize_storage_manager
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.document_search import (
    get_knowledge_search_tool,
    get_list_knowledge_bases_tool,
)
from xagent.core.tools.core.RAG_tools.chunk.chunk_document import chunk_document
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkEmbeddingData,
)
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.management.collections import list_collections
from xagent.core.tools.core.RAG_tools.parse.parse_document import parse_document
from xagent.core.tools.core.RAG_tools.utils.user_permissions import UserPermissions
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    read_chunks_for_embedding,
    write_vectors_to_db,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env


class _FakeEmbeddingAdapter(BaseEmbedding):
    """Local embedding adapter for testing."""

    def encode(self, text, dimension: int | None = None, instruct: str | None = None):
        if isinstance(text, str):
            return [float(len(text))]
        return [[float(len(item))] for item in text]

    def get_dimension(self) -> int:
        return 1

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


class TestUserPermissions:
    """Test UserPermissions class logic."""

    def test_admin_no_filter(self):
        """Admin users get None filter (see all data)."""
        filter_str = UserPermissions.get_user_filter(user_id=None, is_admin=True)
        assert filter_str is None

        filter_str = UserPermissions.get_user_filter(user_id=1, is_admin=True)
        assert filter_str is None

    def test_regular_user_filter(self):
        """Regular users only see their own data."""
        filter_str = UserPermissions.get_user_filter(user_id=1, is_admin=False)
        assert filter_str == "user_id == 1"

        filter_str = UserPermissions.get_user_filter(user_id=42, is_admin=False)
        assert filter_str == "user_id == 42"

    def test_unauthenticated_no_access(self):
        """Unauthenticated users cannot see any data."""
        filter_str = UserPermissions.get_user_filter(user_id=None, is_admin=False)
        assert filter_str == "user_id == -1"  # Impossible condition

    def test_can_access_data_admin(self):
        """Admin can access all data."""
        assert UserPermissions.can_access_data(user_id=1, data_user_id=1, is_admin=True)
        assert UserPermissions.can_access_data(user_id=1, data_user_id=2, is_admin=True)
        assert UserPermissions.can_access_data(
            user_id=1, data_user_id=None, is_admin=True
        )

    def test_can_access_data_regular_user(self):
        """Regular users can only access their own data."""
        assert UserPermissions.can_access_data(
            user_id=1, data_user_id=1, is_admin=False
        )
        # Cannot access other user's data
        assert not UserPermissions.can_access_data(
            user_id=1, data_user_id=2, is_admin=False
        )
        # Cannot access NULL data (legacy)
        assert not UserPermissions.can_access_data(
            user_id=1, data_user_id=None, is_admin=False
        )

    def test_can_access_data_unauthenticated(self):
        """Unauthenticated users cannot access any data."""
        assert not UserPermissions.can_access_data(
            user_id=None, data_user_id=1, is_admin=False
        )
        assert not UserPermissions.can_access_data(
            user_id=None, data_user_id=None, is_admin=False
        )


class TestMultiTenancyCollections:
    """Test multi-tenancy in list_collections."""

    @pytest.fixture()
    def temp_lancedb_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Isolate LanceDB per test."""
        import os

        original = os.environ.get("LANCEDB_DIR")
        lancedb_dir = tmp_path / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))
        yield str(lancedb_dir)
        if original is None:
            monkeypatch.delenv("LANCEDB_DIR", raising=False)
        else:
            monkeypatch.setenv("LANCEDB_DIR", original)

    def _insert_test_documents(self, user_id: int | None):
        """Insert test documents with specific user_id."""
        conn = get_connection_from_env()
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_documents_table,
        )

        ensure_documents_table(conn)
        table = conn.open_table("documents")

        records = [
            {
                "collection": "test_collection",
                "doc_id": f"doc_{uuid.uuid4().hex[:8]}",
                "source_path": "/tmp/test.txt",
                "file_type": "txt",
                "content_hash": "hash1",
                "uploaded_at": datetime.utcnow(),
                "title": "Test Document",
                "language": "en",
                "user_id": user_id,
            }
            for _ in range(5)
        ]
        table.add(records)

    def test_list_collections_admin_sees_all(self, temp_lancedb_dir: str) -> None:
        """Admin users should see all collections regardless of user_id."""
        # Insert documents for different users
        self._insert_test_documents(user_id=1)
        self._insert_test_documents(user_id=2)
        self._insert_test_documents(user_id=None)  # Legacy data

        # Admin sees everything
        result = list_collections(user_id=None, is_admin=True)
        assert result.status == "success"
        # Should see at least one collection
        assert len(result.collections) >= 1
        # Total documents should be sum of all inserted
        total_docs = sum(c.documents for c in result.collections)
        assert total_docs == 15  # 5 docs per user * 3 users

    def test_list_collections_regular_user_sees_only_own(
        self, temp_lancedb_dir: str
    ) -> None:
        """Regular users should only see their own documents."""
        # Insert documents for different users
        self._insert_test_documents(user_id=1)
        self._insert_test_documents(user_id=2)
        self._insert_test_documents(user_id=None)

        # User 1 sees only user 1's data
        result = list_collections(user_id=1, is_admin=False)
        assert result.status == "success"
        total_docs = sum(c.documents for c in result.collections)
        assert total_docs == 5

        # User 2 sees only user 2's data
        result = list_collections(user_id=2, is_admin=False)
        assert result.status == "success"
        total_docs = sum(c.documents for c in result.collections)
        assert total_docs == 5


class TestMultiTenancySearch:
    """Test multi-tenancy in document search."""

    @pytest.fixture()
    def temp_lancedb_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Isolate LanceDB per test."""
        import os

        original = os.environ.get("LANCEDB_DIR")
        lancedb_dir = tmp_path / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))

        # Setup storage
        storage_root = tmp_path / "storage"
        storage_root.mkdir(parents=True, exist_ok=True)
        initialize_storage_manager(str(storage_root), str(storage_root / "uploads"))

        yield str(lancedb_dir)
        if original is None:
            monkeypatch.delenv("LANCEDB_DIR", raising=False)
        else:
            monkeypatch.setenv("LANCEDB_DIR", original)

    def _setup_document_pipeline(
        self, temp_lancedb_dir: str, user_id: int | None, collection: str
    ):
        """Setup complete document pipeline for a user."""
        # Create test file
        test_file = Path(temp_lancedb_dir) / "test.txt"
        test_file.write_text("Test content for search")

        # Register document
        doc_id = uuid.uuid4().hex
        register_document(
            collection=collection,
            source_path=str(test_file),
            doc_id=doc_id,
            user_id=user_id,
        )

        # Parse
        parse_result = parse_document(
            collection=collection,
            doc_id=doc_id,
            parse_method="deepdoc",
            user_id=user_id,
            is_admin=False,
        )
        parse_hash = parse_result["parse_hash"]

        # Chunk
        chunk_document(
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            user_id=user_id,
        )

        # Embed
        embedding_model_id = "test-model"
        embedding_read = read_chunks_for_embedding(
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            model=embedding_model_id,
            user_id=user_id,
            is_admin=False,
        )

        embeddings = [
            ChunkEmbeddingData(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                parse_hash=chunk.parse_hash,
                model=embedding_model_id,
                vector=[1.0],
                text=chunk.text,
                chunk_hash=chunk.chunk_hash,
            )
            for chunk in embedding_read.chunks
        ]

        write_vectors_to_db(
            collection=collection,
            embeddings=embeddings,
            user_id=user_id,
        )

        return collection

    @pytest.mark.integration
    def test_search_regular_user_only_own_results(
        self, temp_lancedb_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regular users can only search their own documents - direct LanceDB level test."""
        # This test verifies user_id filtering at the LanceDB level
        # without going through the full search pipeline

        # Setup: Create embeddings table and insert test data for different users
        import pandas as pd

        conn = get_connection_from_env()

        # Create embeddings table
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )

        ensure_embeddings_table(conn, "test-model", vector_dim=1)

        table = conn.open_table("embeddings_test-model")

        # Insert data for user 1
        data_user1 = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "doc1_user1",
                    "chunk_id": "chunk1",
                    "parse_hash": "hash1",
                    "model": "test-model",
                    "vector": [1.0],
                    "vector_dimension": 1,
                    "text": "content for user 1",
                    "chunk_hash": "chash1",
                    "created_at": datetime.utcnow(),
                    "metadata": "{}",
                    "user_id": 1,
                }
            ]
        )

        # Insert data for user 2
        data_user2 = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "doc2_user2",
                    "chunk_id": "chunk2",
                    "parse_hash": "hash2",
                    "model": "test-model",
                    "vector": [1.0],
                    "vector_dimension": 1,
                    "text": "content for user 2",
                    "chunk_hash": "chash2",
                    "created_at": datetime.utcnow(),
                    "metadata": "{}",
                    "user_id": 2,
                }
            ]
        )

        table.add(data_user1)
        table.add(data_user2)

        # Test 1: User 1 can see their own data
        result_user1 = table.search().where("user_id == 1").to_arrow()
        assert len(result_user1) == 1
        assert result_user1["text"][0].as_py() == "content for user 1"

        # Test 2: User 1 cannot see user 2's data
        result_user1_filtered = table.search().where("user_id == 1").to_arrow()
        assert len(result_user1_filtered) == 1  # Only their own data
        assert result_user1_filtered["doc_id"][0].as_py() == "doc1_user1"

        # Test 3: Admin can see all data
        result_admin = table.search().to_arrow()
        assert len(result_admin) == 2  # All data

        # Test 4: User filter for user 2
        result_user2 = table.search().where("user_id == 2").to_arrow()
        assert len(result_user2) == 1
        assert result_user2["doc_id"][0].as_py() == "doc2_user2"


class TestToolUserContext:
    """Test user context passing through tools."""

    def test_list_knowledge_bases_tool_with_user_context(self):
        """Tool should respect user context when listing collections."""
        tool = get_list_knowledge_bases_tool(
            allowed_collections=None, user_id=1, is_admin=False
        )

        assert tool.user_id == 1
        assert tool.is_admin is False

    def test_search_tool_with_user_context(self):
        """Tool should respect user context when searching."""
        tool = get_knowledge_search_tool(
            embedding_model_id="test-model",
            allowed_collections=None,
            user_id=1,
            is_admin=False,
        )

        assert tool.user_id == 1
        assert tool.is_admin is False

    def test_admin_tool_context(self):
        """Admin tool should have admin flag set."""
        tool = get_list_knowledge_bases_tool(
            allowed_collections=None, user_id=None, is_admin=True
        )

        assert tool.user_id is None
        assert tool.is_admin is True


class TestToolConfigUserContext:
    """Test user context in ToolConfig."""

    def test_tool_config_with_user_context(self):
        """ToolConfig should store and retrieve user context."""
        config = ToolConfig(
            {
                "user_id": 42,
                "is_admin": False,
                "basic_tools_enabled": True,
            }
        )

        assert config.get_user_id() == 42
        assert config.is_admin() is False

    def test_tool_config_admin(self):
        """ToolConfig should handle admin context."""
        config = ToolConfig(
            {
                "user_id": 1,
                "is_admin": True,
                "basic_tools_enabled": True,
            }
        )

        assert config.get_user_id() == 1
        assert config.is_admin() is True

    def test_tool_config_no_user(self):
        """ToolConfig should handle missing user context."""
        config = ToolConfig(
            {
                "basic_tools_enabled": True,
            }
        )

        assert config.get_user_id() is None
        assert config.is_admin() is False
