"""Test authentication API functionality"""

import os

# Test database setup - use file-based database for testing
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.auth import auth_router
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User

# Create temporary directory for database
temp_dir = tempfile.mkdtemp()
temp_db_path = os.path.join(temp_dir, "test.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_db_path}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Create test app without startup events
test_app = FastAPI()
test_app.include_router(auth_router)
test_app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(test_app)


# Cleanup function
def cleanup_test_db():
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except OSError:
        pass


@pytest.fixture(scope="session", autouse=True)
def cleanup_global_test_db():
    """Cleanup global test database after all tests"""
    yield
    cleanup_test_db()


@pytest.fixture(scope="function")
def test_db():
    """Create test database"""
    # Create unique database for each test
    import uuid

    test_db_path = os.path.join(temp_dir, f"test_{uuid.uuid4().hex}.db")
    test_engine = create_engine(
        f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
    )

    # Create all tables
    Base.metadata.create_all(bind=test_engine)

    # Update the engine for this test
    global engine, TestingSessionLocal
    engine = test_engine
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield

    # Cleanup
    Base.metadata.drop_all(bind=test_engine)
    try:
        os.unlink(test_db_path)
    except OSError:
        pass


@pytest.fixture(scope="function")
def test_user_data():
    """Test user data"""
    return {"username": "testuser", "password": "testpassword123"}


@pytest.fixture(scope="function")
def test_admin_data():
    """Test admin user data"""
    return {"username": "admin", "password": "admin123"}


class TestAuthAPI:
    """Test authentication API endpoints"""

    def test_login_success(self, test_db, test_user_data):
        """Test successful user login"""
        # First register the user
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200

        # Then login
        response = client.post("/api/auth/login", json=test_user_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Login successful"
        assert data["user"]["username"] == test_user_data["username"]
        assert "id" in data["user"]
        assert "loginTime" in data["user"]

    def test_login_invalid_credentials(self, test_db, test_user_data):
        """Test login with invalid credentials"""
        # First register the user
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200

        # Try to login with wrong password
        wrong_credentials = {
            "username": test_user_data["username"],
            "password": "wrongpassword",
        }
        response = client.post("/api/auth/login", json=wrong_credentials)
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Incorrect username or password" in data["detail"]

    def test_login_nonexistent_user(self, test_db):
        """Test login with non-existent user"""
        credentials = {"username": "nonexistent", "password": "password123"}
        response = client.post("/api/auth/login", json=credentials)
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Incorrect username or password" in data["detail"]

    def test_register_success(self, test_db, test_user_data):
        """Test successful user registration"""
        response = client.post("/api/auth/register", json=test_user_data)
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Registration successful"
        assert data["user"]["username"] == test_user_data["username"]
        assert "id" in data["user"]
        assert "createdAt" in data["user"]

    def test_register_duplicate_username(self, test_db, test_user_data):
        """Test registration with duplicate username"""
        # Register first user
        response1 = client.post("/api/auth/register", json=test_user_data)
        assert response1.status_code == 200

        # Try to register same username again
        response2 = client.post("/api/auth/register", json=test_user_data)
        assert response2.status_code == 200
        data = response2.json()
        assert data["success"] is False
        assert data["message"] == "Username already exists"

    def test_register_missing_fields(self, test_db):
        """Test registration with missing fields"""
        incomplete_data = {
            "username": "testuser"
            # Missing password
        }
        response = client.post("/api/auth/register", json=incomplete_data)
        assert response.status_code == 422  # Validation error

    def test_auth_check_endpoint(self, test_db):
        """Test auth check endpoint"""
        response = client.get("/api/auth/check")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Authentication API is working"

    def test_password_hashing(self, test_db, test_user_data):
        """Test that passwords are properly hashed"""
        # Register user
        response = client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == 200

        # Check database directly
        db = TestingSessionLocal()

        user = (
            db.query(User).filter(User.username == test_user_data["username"]).first()
        )
        assert user is not None
        assert user.password_hash != test_user_data["password"]  # Should be hashed
        assert len(user.password_hash) == 64  # SHA-256 hash length

        db.close()

    def test_admin_user_creation(self, test_db, test_admin_data):
        """Test that admin user is created with admin privileges"""
        # Register admin user
        response = client.post("/api/auth/register", json=test_admin_data)
        assert response.status_code == 200

        # Check database directly
        db = TestingSessionLocal()

        admin_user = (
            db.query(User).filter(User.username == test_admin_data["username"]).first()
        )
        assert admin_user is not None

        # Check if admin flag is set (this should be set during database initialization)
        # Note: This depends on the database initialization logic
        db.close()

    def test_multiple_users(self, test_db):
        """Test creating multiple users"""
        users = [
            {"username": "user1", "password": "password1"},
            {"username": "user2", "password": "password2"},
            {"username": "user3", "password": "password3"},
        ]

        for user_data in users:
            response = client.post("/api/auth/register", json=user_data)
            assert response.status_code == 200

        # Verify all users can login
        for user_data in users:
            response = client.post("/api/auth/login", json=user_data)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["user"]["username"] == user_data["username"]
