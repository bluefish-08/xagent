from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    IngestionResult,
    WebIngestionResult,
)
from xagent.core.tools.core.RAG_tools.kb import KBApiOperationResult
from xagent.web.api import kb as kb_module
from xagent.web.models.user import User


class _Facade:
    def __init__(self) -> None:
        self.rollback_complete_inputs: list[tuple[KBApiOperationResult[Any], bool]] = []
        self.single_cleanup_inputs: list[
            tuple[KBApiOperationResult[Any], int | None]
        ] = []
        self.batch_cleanup_inputs: list[
            tuple[list[KBApiOperationResult[Any]], int | None]
        ] = []

    def with_rollback_complete(
        self,
        api_result: KBApiOperationResult[Any],
        rollback_complete: bool,
    ) -> KBApiOperationResult[Any]:
        self.rollback_complete_inputs.append((api_result, rollback_complete))
        return KBApiOperationResult(
            result=api_result.result,
            operation_outcome=api_result.operation_outcome,
            rollback_complete=rollback_complete,
        )

    def failed_ingest_cleanup_decision(
        self,
        api_result: KBApiOperationResult[Any],
        *,
        successful_documents: int | None = None,
    ) -> Any:
        self.single_cleanup_inputs.append((api_result, successful_documents))
        return type(
            "Decision",
            (),
            {
                "successful_documents": 3,
                "side_effects_may_remain": api_result.rollback_complete is False,
            },
        )()

    def failed_batch_ingest_cleanup_decision(
        self,
        api_results: list[KBApiOperationResult[Any]],
        *,
        successful_documents: int | None = None,
    ) -> Any:
        self.batch_cleanup_inputs.append((api_results, successful_documents))
        return type(
            "Decision",
            (),
            {
                "successful_documents": 5,
                "side_effects_may_remain": True,
            },
        )()


@pytest.mark.asyncio
async def test_api_failed_ingest_config_cleanup_uses_api_outcome_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = _Facade()
    restore_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(kb_module, "_get_api_compatibility_facade", lambda: facade)

    async def fake_restore(**kwargs: Any) -> None:
        restore_calls.append(kwargs)

    monkeypatch.setattr(
        kb_module,
        "_cleanup_collection_metadata_after_failed_ingest",
        fake_restore,
    )
    api_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="failed")
    )
    user = User()
    user.id = 7

    updated = await kb_module._cleanup_collection_metadata_after_failed_api_ingest(
        api_result=api_result,
        collection_existed_before=True,
        collection_name="demo",
        user=user,
        context="ingest",
        successful_documents=1,
        rollback_complete=False,
    )

    assert updated.rollback_complete is False
    assert facade.rollback_complete_inputs == [(api_result, False)]
    assert facade.single_cleanup_inputs == [(updated, 1)]
    assert restore_calls == [
        {
            "collection_existed_before": True,
            "collection_name": "demo",
            "user": user,
            "context": "ingest",
            "successful_documents": 3,
            "side_effects_may_remain": True,
        }
    ]


@pytest.mark.asyncio
async def test_api_failed_batch_ingest_config_cleanup_uses_api_outcome_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = _Facade()
    restore_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(kb_module, "_get_api_compatibility_facade", lambda: facade)

    async def fake_restore(**kwargs: Any) -> None:
        restore_calls.append(kwargs)

    monkeypatch.setattr(
        kb_module,
        "_cleanup_collection_metadata_after_failed_ingest",
        fake_restore,
    )
    api_results = [
        KBApiOperationResult(result=IngestionResult(status="success", message="ok")),
        KBApiOperationResult(result=IngestionResult(status="error", message="failed")),
    ]
    user = User()
    user.id = 9

    await kb_module._cleanup_collection_metadata_after_failed_batch_api_ingest(
        api_results=api_results,
        collection_existed_before=False,
        collection_name="demo",
        user=user,
        context="ingest_cloud",
        successful_documents=1,
    )

    assert facade.batch_cleanup_inputs == [(api_results, 1)]
    assert restore_calls == [
        {
            "collection_existed_before": False,
            "collection_name": "demo",
            "user": user,
            "context": "ingest_cloud",
            "successful_documents": 5,
            "side_effects_may_remain": True,
        }
    ]


def test_background_failed_ingest_config_cleanup_reuses_api_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.jobs import kb_tasks

    api_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="failed")
    )
    user = User()
    user.id = 11
    db = object()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(kb_tasks, "_get_job_user", lambda *args, **kwargs: user)

    async def fake_api_helper(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        kb_module,
        "_cleanup_collection_metadata_after_failed_api_ingest",
        fake_api_helper,
    )

    kb_tasks._cleanup_failed_job_collection_metadata_after_api_ingest(
        db,  # type: ignore[arg-type]
        {
            "collection": "job-kb",
            "collection_existed_before": False,
        },
        api_result=api_result,
        context="background document ingest",
        successful_documents=2,
        rollback_complete=True,
    )

    assert calls == [
        {
            "api_result": api_result,
            "collection_existed_before": False,
            "collection_name": "job-kb",
            "user": user,
            "context": "background document ingest",
            "successful_documents": 2,
            "rollback_complete": True,
        }
    ]


def test_background_web_cleanup_keeps_early_exception_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.jobs import kb_tasks

    fallback = MagicMock()
    api_helper = MagicMock()
    db = object()
    payload = {"collection": "job-kb"}
    monkeypatch.setattr(
        kb_tasks,
        "_cleanup_failed_job_collection_metadata",
        fallback,
    )
    monkeypatch.setattr(
        kb_tasks,
        "_cleanup_failed_job_collection_metadata_after_api_ingest",
        api_helper,
    )

    kb_tasks._cleanup_failed_web_collection_metadata_if_new(
        db,  # type: ignore[arg-type]
        payload,
    )

    fallback.assert_called_once_with(
        db,
        payload,
        context="background web ingest",
        successful_documents=0,
    )
    api_helper.assert_not_called()


def _vector_store_returning(records: Any) -> MagicMock:
    """Vector store stub whose document listing drives failure-time scoping."""
    store = MagicMock()
    if isinstance(records, Exception):
        store.list_document_records.side_effect = records
    else:
        store.list_document_records.return_value = records
    return store


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([{"file_id": "sibling"}], True),
        ([], False),
        (RuntimeError("vector store down"), True),
    ],
)
def test_collection_holds_other_documents_rechecks_at_failure_time(
    monkeypatch: pytest.MonkeyPatch,
    records: Any,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        kb_module, "get_vector_index_store", lambda: _vector_store_returning(records)
    )

    assert (
        kb_module._collection_holds_other_documents(
            collection_name="shared-kb",
            user_id=3,
            is_admin=False,
            context="test",
        )
        is expected
    )


@pytest.mark.asyncio
async def test_rollback_keeps_collection_holding_a_sibling_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale collection_existed_before must not delete a sibling job's work."""
    from pathlib import Path

    delete_collection = MagicMock()
    monkeypatch.setattr(
        kb_module,
        "get_vector_index_store",
        lambda: _vector_store_returning([{"file_id": "sibling"}]),
    )
    monkeypatch.setattr(kb_module, "delete_collection", delete_collection)
    monkeypatch.setattr(kb_module, "_restore_ingest_file_backup", MagicMock())
    monkeypatch.setattr(kb_module, "clear_ingestion_status", MagicMock())

    user = User()
    user.id = 5
    user.is_admin = False
    file_record = MagicMock()
    file_record.file_id = "file-1"

    await kb_module._rollback_failed_ingestion(
        db=MagicMock(),
        user=user,
        collection_name="shared-kb",
        result=IngestionResult(status="error", message="failed", doc_id="doc-1"),
        file_path=Path("/tmp/does-not-matter.pdf"),
        file_record=file_record,
        collection_existed_before=False,
        uploaded_file_existed_before=True,
        file_backup_path=None,
        had_existing_file=False,
    )

    delete_collection.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_keeps_metadata_when_a_sibling_ingest_published_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stamped flag is stale; a config-only ghost holds no documents."""
    cleanup = MagicMock()
    monkeypatch.setattr(
        kb_module,
        "get_vector_index_store",
        lambda: _vector_store_returning([{"file_id": "sibling"}]),
    )
    monkeypatch.setattr(kb_module, "_cleanup_failed_new_collection_metadata", cleanup)

    user = User()
    user.id = 5
    user.is_admin = False

    await kb_module._cleanup_collection_metadata_after_failed_ingest(
        collection_existed_before=False,
        collection_name="shared-kb",
        user=user,
        context="ingest",
    )

    cleanup.assert_not_called()


def _web_result(documents_created: int) -> WebIngestionResult:
    return WebIngestionResult(
        status="success",
        collection="web-kb",
        total_urls_found=documents_created,
        pages_crawled=documents_created,
        pages_failed=0,
        documents_created=documents_created,
        chunks_created=documents_created,
        embeddings_created=documents_created,
        message="crawl completed",
        elapsed_time_ms=0,
    )


@pytest.mark.parametrize(
    ("collection_existed_before", "expected_outcome"),
    [(False, "the knowledge base was not created"), (True, "nothing was added")],
)
def test_empty_ingest_is_demoted_to_error(
    monkeypatch: pytest.MonkeyPatch,
    collection_existed_before: bool,
    expected_outcome: str,
) -> None:
    monkeypatch.setattr(
        kb_module,
        "_get_api_compatibility_facade",
        lambda: type(
            "F",
            (),
            {
                "with_result": staticmethod(
                    lambda api_result, result: KBApiOperationResult(result=result)
                )
            },
        )(),
    )
    api_result = KBApiOperationResult(result=_web_result(0))

    demoted = kb_module._demote_empty_ingest_to_error(
        api_result,
        collection_existed_before=collection_existed_before,
    ).result

    assert demoted.status == "error"
    assert expected_outcome in demoted.message


def test_successful_ingest_is_not_demoted() -> None:
    api_result = KBApiOperationResult(result=_web_result(2))

    assert (
        kb_module._demote_empty_ingest_to_error(
            api_result, collection_existed_before=False
        )
        is api_result
    )


def test_job_config_save_failure_is_not_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying after the documents landed would delete the collection."""
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
    from xagent.web.jobs import kb_tasks
    from xagent.web.jobs.exceptions import BackgroundJobHandlerError

    user = User()
    user.id = 13
    monkeypatch.setattr(kb_tasks, "_get_job_user", lambda *args, **kwargs: user)

    async def fail_save(**kwargs: Any) -> None:
        raise kb_module.CollectionConfigSaveError("config store down")

    monkeypatch.setattr(kb_module, "_save_collection_config_after_ingest", fail_save)

    with pytest.raises(BackgroundJobHandlerError) as excinfo:
        kb_tasks._save_job_collection_config_after_ingest(
            MagicMock(),
            {"collection": "job-kb", "user_id": 13},
            IngestionConfig(),
            context="background document ingest",
            documents_created=1,
        )

    assert excinfo.value.retryable is False
