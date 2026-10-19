"""API tests for agent prompt_context_fields declarations."""

import pytest

from .conftest import _admin_headers, client

pytestmark = pytest.mark.usefixtures("_test_db")

_FIELDS = [
    {"name": "account_id", "max_length": 64},
    {"name": "plan.tier", "max_length": 16},
]


def _create(headers: dict[str, str], name: str, fields=None) -> dict:
    body = {"name": name, "execution_mode": "balanced"}
    if fields is not None:
        body["prompt_context_fields"] = fields
    response = client.post("/api/agents", headers=headers, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_create_echoes_declarations() -> None:
    headers = _admin_headers()
    created = _create(headers, "declared", _FIELDS)

    assert created["prompt_context_fields"] == _FIELDS


def test_create_without_declarations_defaults_to_empty() -> None:
    headers = _admin_headers()
    assert _create(headers, "undeclared")["prompt_context_fields"] == []


def test_update_replaces_declarations_and_omission_keeps_them() -> None:
    headers = _admin_headers()
    agent_id = _create(headers, "updatable", _FIELDS)["id"]

    kept = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"description": "changed"},
    )
    assert kept.status_code == 200, kept.text
    assert kept.json()["prompt_context_fields"] == _FIELDS

    replaced = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"prompt_context_fields": [{"name": "only", "max_length": 1}]},
    )
    assert replaced.status_code == 200, replaced.text
    assert replaced.json()["prompt_context_fields"] == [
        {"name": "only", "max_length": 1}
    ]


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param([{"name": "1bad", "max_length": 8}], id="bad-name-start"),
        pytest.param([{"name": "has space", "max_length": 8}], id="bad-name-space"),
        pytest.param([{"name": "a" * 65, "max_length": 8}], id="bad-name-length"),
        pytest.param(
            [
                {"name": "dup", "max_length": 8},
                {"name": "dup", "max_length": 4},
            ],
            id="duplicate-name",
        ),
        pytest.param(
            [{"name": f"f{i}", "max_length": 8} for i in range(33)],
            id="too-many",
        ),
        pytest.param([{"name": "zero", "max_length": 0}], id="max-length-zero"),
        pytest.param([{"name": "huge", "max_length": 1025}], id="max-length-1025"),
    ],
)
def test_invalid_declarations_are_422(fields) -> None:
    headers = _admin_headers()
    response = client.post(
        "/api/agents",
        headers=headers,
        json={"name": "invalid", "prompt_context_fields": fields},
    )

    assert response.status_code == 422, response.text


def test_update_invalid_declarations_are_422() -> None:
    headers = _admin_headers()
    agent_id = _create(headers, "invalid-update")["id"]

    response = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"prompt_context_fields": [{"name": "dup", "max_length": 1}] * 2},
    )

    assert response.status_code == 422, response.text


def test_boundary_declarations_are_accepted() -> None:
    headers = _admin_headers()
    fields = [
        {"name": "A" + "b" * 63, "max_length": 1024},
        {"name": "_", "max_length": 1},
    ]
    created = _create(headers, "boundary", fields)

    assert created["prompt_context_fields"] == fields
