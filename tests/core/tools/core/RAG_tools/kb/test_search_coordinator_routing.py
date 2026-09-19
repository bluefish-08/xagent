"""#796 - search resolves through the coordinator and never touches a backend.

The coordinator owns scope/access/backend resolution for search exactly as it
does for the other data-plane families; the legacy facade and the public
``retrieval.search_*`` functions are thin adapters over it.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.kb.coordinator import KBCoordinator
from xagent.core.tools.core.RAG_tools.kb.legacy_step_compatibility import (
    KBLegacyStepCompatibilityFacade,
)
from xagent.core.tools.core.RAG_tools.kb.models import KBAccessMode


async def _async_return(value):
    return value


def _coordinator_with_handle(handle: MagicMock) -> KBCoordinator:
    # Bare instance: the search entry points only open a handle and delegate,
    # so a coordinator with no store wiring proves nothing else is reached.
    coordinator = KBCoordinator.__new__(KBCoordinator)
    coordinator.open_collection_sync = MagicMock(return_value=handle)
    coordinator.open_collection = MagicMock(
        side_effect=lambda _request: _async_return(handle)
    )
    return coordinator


SYNC_CASES = [
    ("search_dense", ("col1", "model-x", [0.1]), {"top_k": 4}),
    ("search_sparse", ("col2", "model-y", "query text"), {"top_k": 5}),
    ("search_hybrid", ("col3", "model-z", "query", [0.3, 0.4]), {"top_k": 6}),
]

ASYNC_CASES = [
    ("search_dense_async", ("col1", "model-x", [0.1]), {"top_k": 8}),
    ("search_sparse_async", ("col2", "model-y", "hello"), {"top_k": 9}),
]


@pytest.mark.parametrize("method, args, kwargs", SYNC_CASES)
def test_sync_search_opens_a_read_handle_and_delegates(method, args, kwargs):
    handle = MagicMock()
    coordinator = _coordinator_with_handle(handle)

    getattr(coordinator, method)(*args, user_id=5, is_admin=True, **kwargs)

    request = coordinator.open_collection_sync.call_args.args[0]
    assert request.collection == args[0]
    assert request.access_mode == KBAccessMode.READ
    assert request.user_id == 5
    assert request.is_admin is True
    assert request.hide_missing is True

    delegate = getattr(handle, method)
    delegate.assert_called_once()
    assert delegate.call_args.kwargs["top_k"] == kwargs["top_k"]


@pytest.mark.parametrize("method, args, kwargs", ASYNC_CASES)
def test_async_search_opens_a_read_handle_and_delegates(method, args, kwargs):
    handle = MagicMock()
    getattr(handle, method).side_effect = lambda *a, **k: _async_return(MagicMock())
    coordinator = _coordinator_with_handle(handle)

    asyncio.run(getattr(coordinator, method)(*args, **kwargs))

    request = coordinator.open_collection.call_args.args[0]
    assert request.access_mode == KBAccessMode.READ
    assert request.hide_missing is True

    delegate = getattr(handle, method)
    delegate.assert_called_once()
    assert delegate.call_args.kwargs["top_k"] == kwargs["top_k"]


@pytest.mark.parametrize("method, args, kwargs", SYNC_CASES)
def test_sync_search_stays_synchronous(method, args, kwargs):
    """A sync caller must not be pushed onto the async handle-opening path."""
    handle = MagicMock()
    coordinator = _coordinator_with_handle(handle)

    getattr(coordinator, method)(*args, **kwargs)

    coordinator.open_collection.assert_not_called()


@pytest.mark.parametrize("method, args, kwargs", SYNC_CASES)
def test_legacy_facade_forwards_to_the_coordinator(method, args, kwargs):
    facade = KBLegacyStepCompatibilityFacade()
    coordinator = MagicMock()

    with patch.object(facade, "_active_coordinator", return_value=coordinator):
        getattr(facade, method)(*args, **kwargs)

    getattr(coordinator, method).assert_called_once()


@pytest.mark.parametrize(
    "module_name, method",
    [
        ("search_dense", "search_dense"),
        ("search_sparse", "search_sparse"),
        ("search_hybrid", "search_hybrid"),
    ],
)
def test_public_retrieval_functions_route_through_the_coordinator(module_name, method):
    import importlib

    module = importlib.import_module(
        f"xagent.core.tools.core.RAG_tools.retrieval.{module_name}"
    )
    coordinator = MagicMock()

    with patch.object(module, "_get_coordinator", return_value=coordinator):
        if method == "search_dense":
            module.search_dense("col", "model", [0.1], top_k=3)
        elif method == "search_sparse":
            module.search_sparse("col", "model", "q", top_k=3)
        else:
            module.search_hybrid("col", "model", "q", [0.1], top_k=3)

    getattr(coordinator, method).assert_called_once()
