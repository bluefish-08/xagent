"""Test model service functionality"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User
from xagent.web.services.model_service import (
    get_compact_model,
    get_default_model,
    get_default_vision_model,
    get_embedding_model,
    get_fast_model,
)

# Test database setup - use in-memory database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create database session"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def admin_user(db_session):
    """Create admin user"""
    user = User(username="admin", password_hash="hashed_admin_pass", is_admin=True)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def regular_user(db_session):
    """Create regular user"""
    user = User(username="regularuser", password_hash="hashed_pass", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def sample_model(db_session):
    """Create sample model"""
    model = Model(
        model_id="test-openai-model",
        category="llm",
        model_provider="openai",
        model_name="gpt-4",
        api_key="test-api-key",
        base_url="https://api.openai.com/v1",
        temperature=0.7,
        abilities=["chat", "tool_calling"],
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


class TestModelService:
    """Test model service functionality"""

    def test_get_default_model_user_specific(self):
        """Test getting user-specific default model"""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
        ):
            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Create mock objects
            mock_user_default = MagicMock()
            mock_user_default.user_id = 1
            mock_user_default.config_type = "general"
            mock_user_default.model = MagicMock()
            mock_user_default.model.model_id = "test-model"

            # Setup query result
            mock_db.query.return_value.join.return_value.filter.return_value.first.return_value = mock_user_default

            # Setup mock LLM creation
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_default_model(1)

            assert result == mock_llm
            mock_create.assert_called_once_with(mock_user_default.model)

    def test_get_default_model_admin_shared(self):
        """Test getting admin shared default model"""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
        ):
            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Setup query to return None for user-specific query and shared default for shared query
            mock_query_result = MagicMock()
            mock_query_result.first.return_value = None
            mock_query_result.all.return_value = [MagicMock()]
            mock_db.query.return_value.join.return_value.filter.return_value = (
                mock_query_result
            )

            # Setup mock LLM creation
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_default_model(2)  # regular user

            assert result == mock_llm
            mock_create.assert_called_once()

    def test_get_default_model_no_user_id(self):
        """Test getting default model without user ID - should return None"""
        with patch("xagent.web.models.database.get_db") as mock_get_db:
            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Setup query to return no shared defaults
            mock_db.query.return_value.join.return_value.filter.return_value.all.return_value = []

            result = get_default_model()
            assert result is None

    def test_get_default_vision_model(self):
        """Test getting vision model"""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
        ):
            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Create mock vision default
            mock_vision_default = MagicMock()
            mock_vision_default.user_id = 1
            mock_vision_default.config_type = "visual"
            mock_vision_default.model = MagicMock()
            mock_vision_default.model.model_id = "vision-model"

            # Setup query result
            mock_db.query.return_value.join.return_value.filter.return_value.first.return_value = mock_vision_default

            # Setup mock LLM creation
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_default_vision_model(1)

            assert result == mock_llm
            mock_create.assert_called_once_with(mock_vision_default.model)

    def test_get_fast_model(self):
        """Test getting fast model"""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
        ):
            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Create mock fast default
            mock_fast_default = MagicMock()
            mock_fast_default.user_id = 1
            mock_fast_default.config_type = "small_fast"
            mock_fast_default.model = MagicMock()
            mock_fast_default.model.model_id = "fast-model"

            # Setup query result
            mock_db.query.return_value.join.return_value.filter.return_value.first.return_value = mock_fast_default

            # Setup mock LLM creation
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_fast_model(1)

            assert result == mock_llm
            mock_create.assert_called_once_with(mock_fast_default.model)

    def test_get_compact_model(self):
        """Test getting compact model"""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
        ):
            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Create mock compact default
            mock_compact_default = MagicMock()
            mock_compact_default.user_id = 1
            mock_compact_default.config_type = "compact"
            mock_compact_default.model = MagicMock()
            mock_compact_default.model.model_id = "compact-model"

            # Setup query result
            mock_db.query.return_value.join.return_value.filter.return_value.first.return_value = mock_compact_default

            # Setup mock LLM creation
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_compact_model(1)

            assert result == mock_llm
            mock_create.assert_called_once_with(mock_compact_default.model)

    def test_get_embedding_model(self):
        """Test getting embedding model"""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
        ):
            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Create mock embedding default
            mock_embedding_default = MagicMock()
            mock_embedding_default.user_id = 1
            mock_embedding_default.config_type = "embedding"
            mock_embedding_default.model = MagicMock()
            mock_embedding_default.model.model_id = "embedding-model"

            # Setup query result
            mock_db.query.return_value.join.return_value.filter.return_value.first.return_value = mock_embedding_default

            # Setup mock LLM creation
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_embedding_model(1)

            assert result == mock_llm
            mock_create.assert_called_once_with(mock_embedding_default.model)

    def test_get_default_model_no_configuration(self):
        """Test getting default model when no configuration exists - should return None"""
        with patch("xagent.web.models.database.get_db") as mock_get_db:
            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Setup query to return no results
            mock_db.query.return_value.join.return_value.filter.return_value.first.return_value = None
            mock_db.query.return_value.join.return_value.filter.return_value.all.return_value = []

            result = get_default_model(1)
            assert result is None

    def test_model_service_multiple_users(self):
        """Test model service with multiple users"""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
        ):
            # Setup mock database session - create a new session for each call
            mock_db1 = MagicMock()
            mock_db2 = MagicMock()
            mock_get_db.side_effect = [iter([mock_db1]), iter([mock_db2])]

            # Setup query to return None for individual user defaults and shared default for shared query
            mock_query_result1 = MagicMock()
            mock_query_result1.first.return_value = None
            mock_query_result1.all.return_value = [MagicMock()]
            mock_db1.query.return_value.join.return_value.filter.return_value = (
                mock_query_result1
            )

            mock_query_result2 = MagicMock()
            mock_query_result2.first.return_value = None
            mock_query_result2.all.return_value = [MagicMock()]
            mock_db2.query.return_value.join.return_value.filter.return_value = (
                mock_query_result2
            )

            # Setup mock LLM creation
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            # Both users should get the same shared model
            admin_result = get_default_model(1)  # admin user
            regular_result = get_default_model(2)  # regular user

            assert admin_result == mock_llm
            assert regular_result == mock_llm

            # Should have been called twice (once for each user)
            assert mock_create.call_count == 2
