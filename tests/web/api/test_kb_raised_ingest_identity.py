"""A raised ingest is rolled back under its real document identity (#2662)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from tests.web.api import test_kb_cloud_rollback_contract as cloud
from tests.web.api import test_kb_dir as kb_dir
from tests.web.api import test_kb_local_rollback_contract as local
from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.web.api import kb as kb_module
from xagent.web.api.kb import CollectionConfigSaveError
from xagent.web.models.uploaded_file import UploadedFile

test_env = kb_dir.test_env
temp_uploads = kb_dir.temp_uploads

_list_refs = kb_module._list_document_refs_for_uploaded_file


def _register(**kwargs: Any) -> dict[str, Any]:
    return register_document(
        collection=kwargs["collection"],
        source_path=kwargs["source_path"],
        user_id=kwargs["user_id"],
        file_id=kwargs["file_id"],
    )


def _register_then_raise(seen: dict[str, Any]) -> Any:
    def _ingest(**kwargs: Any) -> IngestionResult:
        seen.update(file_id=kwargs["file_id"], **_register(**kwargs))
        raise RuntimeError("binding write failed")

    return _ingest


def _register_and_succeed(seen: dict[str, Any]) -> Any:
    def _ingest(**kwargs: Any) -> IngestionResult:
        registered = _register(**kwargs)
        seen.update(file_id=kwargs["file_id"], **registered)
        return IngestionResult(
            status="success",
            doc_id=registered["doc_id"],
            completed_steps=[
                {"name": "register_document", "metadata": registered},
            ],
            message="ok",
        )

    return _ingest


def _file_ids(test_env: Any) -> list[str]:
    session = test_env[3]()
    try:
        return [str(row.file_id) for row in session.query(UploadedFile).all()]
    finally:
        session.close()


def _refs_down(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("refs down")


def test_ingest_raise_after_registration_removes_the_new_document(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise(seen),
        ),
    )

    assert response.status_code == 500
    assert seen["created"] is True
    assert _list_refs(seen["file_id"]) == []
    assert _file_ids(test_env) == []


def test_ingest_raise_keeps_a_document_that_existed_before(
    test_env, temp_uploads
) -> None:
    first: dict[str, Any] = {}
    local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_and_succeed(first),
        ),
    )
    second: dict[str, Any] = {}

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise(second),
        ),
    )

    assert response.status_code == 500
    assert second["file_id"] == first["file_id"]
    assert _list_refs(first["file_id"]) == [("coll", first["doc_id"])]
    assert _file_ids(test_env) == [first["file_id"]]


def test_ingest_raise_keeps_the_document_when_the_pre_check_fails(
    test_env, temp_uploads
) -> None:
    first: dict[str, Any] = {}
    local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_and_succeed(first),
        ),
    )

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise({}),
        ),
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            side_effect=_refs_down,
        ),
    )

    assert response.status_code == 500
    assert _list_refs(first["file_id"]) == [("coll", first["doc_id"])]
    assert _file_ids(test_env) == [first["file_id"]]


def test_ingest_raise_removes_the_document_when_the_post_check_fails(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise(seen),
        ),
        patch(
            "xagent.web.api.kb._list_document_refs_for_uploaded_file",
            side_effect=_refs_down,
        ),
    )

    assert response.status_code == 500
    assert _list_refs(seen["file_id"]) == []
    assert _file_ids(test_env) == []


def test_ingest_raise_into_a_new_collection_compares_real_doc_ids(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}
    may_delete = AsyncMock(return_value=False)

    response = local._post_ingest(
        test_env,
        "x.txt",
        "fresh",
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("new")),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise(seen),
        ),
        patch("xagent.web.api.kb._rollback_may_delete_collection", may_delete),
    )

    assert response.status_code == 500
    assert may_delete.await_args.kwargs["other_document_present"] is False
    assert _list_refs(seen["file_id"]) == []


def test_ingest_raise_keeps_the_only_document_of_a_configless_collection(
    test_env, temp_uploads
) -> None:
    first: dict[str, Any] = {}
    local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        local._existing_collection(),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_and_succeed(first),
        ),
        patch(
            "xagent.web.api.kb._save_collection_config_after_ingest",
            AsyncMock(side_effect=CollectionConfigSaveError("config write failed")),
        ),
    )

    response = local._post_ingest(
        test_env,
        "x.txt",
        "coll",
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("read")),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=_register_then_raise({}),
        ),
    )

    assert response.status_code == 500
    assert _list_refs(first["file_id"]) == [("coll", first["doc_id"])]
    assert _file_ids(test_env) == [first["file_id"]]
    stored = temp_uploads / f"user_{test_env[2].id}" / "coll" / "x.txt"
    assert stored.read_bytes() == b"new content"


def test_ingest_raise_before_registration_may_still_delete_a_new_collection(
    test_env, temp_uploads
) -> None:
    may_delete = AsyncMock(return_value=False)

    response = local._post_ingest(
        test_env,
        "x.txt",
        "fresh",
        patch("xagent.web.api.kb.get_collection_sync", side_effect=ValueError("new")),
        patch(
            "xagent.web.api.kb.run_document_ingestion",
            side_effect=RuntimeError("before registration"),
        ),
        patch("xagent.web.api.kb._rollback_may_delete_collection", may_delete),
    )

    assert response.status_code == 500
    assert may_delete.await_args.kwargs["other_document_present"] is False


def test_ingest_cloud_raise_after_registration_removes_the_new_document(
    test_env, temp_uploads
) -> None:
    seen: dict[str, Any] = {}

    response = cloud._post_cloud(
        test_env,
        _register_then_raise(seen),
        patch("xagent.web.api.kb.get_collection_sync", return_value=object()),
    )

    assert response.status_code == 200
    entry = response.json()[0]
    assert entry["status"] == "error"
    assert entry["doc_id"] == "cloud.csv"
    assert entry["message"] == "Ingestion failed: binding write failed"
    assert _list_refs(seen["file_id"]) == []
    assert _file_ids(test_env) == []


def test_ingest_cloud_raise_keeps_a_document_that_existed_before(
    test_env, temp_uploads
) -> None:
    existing = patch("xagent.web.api.kb.get_collection_sync", return_value=object())
    first: dict[str, Any] = {}
    cloud._post_cloud(test_env, _register_and_succeed(first), existing)

    response = cloud._post_cloud(test_env, _register_then_raise({}), existing)

    assert response.json()[0]["status"] == "error"
    assert _list_refs(first["file_id"]) == [("cloud_coll", first["doc_id"])]
    assert _file_ids(test_env) == [first["file_id"]]
