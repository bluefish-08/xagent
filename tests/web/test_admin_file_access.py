import os
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.auth import hash_password
from xagent.web.api.files import file_router
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.database import Base, get_db
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


@pytest.fixture(scope="function")
def test_db():
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_db_fd)

    test_engine = create_engine(
        f"sqlite:///{temp_db_path}", connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    def override_get_db():
        db = None
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            if db is not None:
                db.close()

    test_app = FastAPI()
    test_app.include_router(file_router)
    test_app.dependency_overrides[get_db] = override_get_db

    Base.metadata.create_all(bind=test_engine)

    session = TestingSessionLocal()
    try:
        admin_user = User(
            username="admin", password_hash=hash_password("admin"), is_admin=True
        )
        regular_user = User(
            username="regular", password_hash=hash_password("regular"), is_admin=False
        )
        session.add(admin_user)
        session.add(regular_user)
        session.commit()
        session.refresh(admin_user)
        session.refresh(regular_user)
        yield admin_user, regular_user, test_app, session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()
        try:
            os.unlink(temp_db_path)
        except OSError:
            pass


@pytest.fixture(scope="function")
def temp_uploads_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        import xagent.web.api.files

        monkeypatch.setattr(xagent.web.api.files, "UPLOADS_DIR", temp_path)
        yield temp_path


def create_auth_headers(user):
    from datetime import datetime, timedelta

    import jwt

    payload = {
        "sub": user.username,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "user_id": user.id,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


def create_uploaded_file(
    session,
    uploads_dir: Path,
    user_id: int,
    task_id: int,
    filename: str,
    content: str,
) -> UploadedFile:
    user_dir = uploads_dir / f"user_{user_id}" / f"web_task_{task_id}" / "output"
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / filename
    file_path.write_text(content)

    uploaded_file = UploadedFile(
        user_id=user_id,
        task_id=task_id,
        filename=filename,
        storage_path=str(file_path),
        mime_type="text/html",
        file_size=len(content.encode("utf-8")),
    )
    session.add(uploaded_file)
    session.commit()
    session.refresh(uploaded_file)
    return uploaded_file


class TestAdminFileAccess:
    def test_admin_access_other_user_file(self, test_db, temp_uploads_dir):
        admin_user, regular_user, test_app, session = test_db
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=78,
            user_id=regular_user_id,
            title="Test Task",
            description="Test task for file access",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "report.html",
            "test content",
        )

        client = TestClient(test_app)
        admin_headers = create_auth_headers(admin_user)
        response = client.get(
            f"/api/files/download/{uploaded_file.file_id}", headers=admin_headers
        )

        assert response.status_code == 200
        assert response.content == b"test content"

    def test_regular_user_access_own_file(self, test_db, temp_uploads_dir):
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=79,
            user_id=regular_user_id,
            title="Test Task",
            description="Test task for file access",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "my_report.html",
            "my content",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/download/{uploaded_file.file_id}", headers=user_headers
        )

        assert response.status_code == 200
        assert response.content == b"my content"

    def test_regular_user_access_other_user_file_denied(
        self, test_db, temp_uploads_dir
    ):
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        another_user = User(
            username="another", password_hash=hash_password("another"), is_admin=False
        )
        session.add(another_user)
        session.commit()

        another_user_id = int(cast(Any, another_user.id))
        task = Task(
            id=80,
            user_id=another_user_id,
            title="Another User Task",
            description="Task belonging to another user",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            another_user_id,
            int(cast(Any, task.id)),
            "secret_report.html",
            "secret content",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/download/{uploaded_file.file_id}", headers=user_headers
        )

        assert response.status_code == 403

    def test_missing_file_id_returns_404(self, test_db, temp_uploads_dir):
        admin_user, regular_user, test_app, session = test_db
        del regular_user, session, temp_uploads_dir
        client = TestClient(test_app)
        admin_headers = create_auth_headers(admin_user)
        response = client.get(
            "/api/files/download/00000000-0000-0000-0000-000000000000",
            headers=admin_headers,
        )
        assert response.status_code == 404

    def test_delete_file_by_file_id(self, test_db, temp_uploads_dir):
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=81,
            user_id=regular_user_id,
            title="Delete Task",
            description="delete test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "delete_me.txt",
            "to delete",
        )
        file_path = Path(str(uploaded_file.storage_path))
        assert file_path.exists()

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.delete(
            f"/api/files/{uploaded_file.file_id}", headers=user_headers
        )

        assert response.status_code == 200
        assert not file_path.exists()
        assert (
            session.query(UploadedFile)
            .filter(UploadedFile.file_id == uploaded_file.file_id)
            .first()
            is None
        )
