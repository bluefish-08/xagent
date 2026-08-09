from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
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


class _ConfigLookupFacade:
    """Facade stub whose collection config lookup drives rollback scoping."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self.lookups: list[tuple[str, int]] = []

    async def get_collection_config(
        self,
        *,
        collection: str,
        user_id: int,
        is_admin: bool = False,
    ) -> Any:
        self.lookups.append((collection, user_id))
        if isinstance(self._config, Exception):
            raise self._config
        return self._config


@pytest.mark.parametrize(
    ("stored_config", "expected"),
    [
        ('{"chunk_size": 512}', True),
        (None, False),
        (RuntimeError("config store down"), True),
    ],
)
@pytest.mark.asyncio
async def test_collection_or_config_exists_now_rechecks_at_failure_time(
    monkeypatch: pytest.MonkeyPatch,
    stored_config: Any,
    expected: bool,
) -> None:
    facade = _ConfigLookupFacade(stored_config)
    monkeypatch.setattr(kb_module, "_get_api_compatibility_facade", lambda: facade)

    assert (
        await kb_module._collection_or_config_exists_now(
            False,
            collection_name="shared-kb",
            user_id=3,
            context="test",
        )
        is expected
    )
    assert facade.lookups == [("shared-kb", 3)]


@pytest.mark.asyncio
async def test_rollback_keeps_collection_published_by_a_sibling_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale collection_existed_before must not delete a sibling job's work."""
    from pathlib import Path

    facade = _ConfigLookupFacade('{"chunk_size": 512}')
    delete_collection = MagicMock()
    monkeypatch.setattr(kb_module, "_get_api_compatibility_facade", lambda: facade)
    monkeypatch.setattr(kb_module, "delete_collection", delete_collection)
    monkeypatch.setattr(kb_module, "get_vector_index_store", MagicMock())
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
