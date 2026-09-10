"""History replay does not turn a past pause into a clarification form.

Every suspending ReAct turn now publishes at least one answerable field, and
the field is persisted on the assistant history row so the *current* waiting
turn still renders a form after a reload. Replay of the *earlier* turns reads
the same rows: without this filter, every past ``send_message`` pause -- which
carried no interactions before and rendered as ordinary prose -- would come
back as a collapsed clarification form. The frontend is unchanged; what it is
handed is what has to stay the same.
"""

from types import SimpleNamespace

import pytest

from tests.web.api.conftest import _direct_db_session
from xagent.core.tools.adapters.vibe.interaction_types import (
    default_waiting_interaction,
)
from xagent.web.api import websocket as websocket_api
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User

REAL_FIELD = {
    "type": "select_one",
    "field": "city",
    "label": "City",
    "options": [{"label": "Paris", "value": "paris"}],
}


def _waiting_history_row(*, username: str, interactions: list[dict]) -> tuple[int, int]:
    db = _direct_db_session()
    try:
        user = User(username=username, password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(
            user_id=int(user.id),
            title="Replayed pause",
            description="Replayed pause",
            status=TaskStatus.WAITING_FOR_USER,
        )
        db.add(task)
        db.flush()
        db.add(
            TaskChatMessage(
                task_id=int(task.id),
                user_id=int(user.id),
                role="assistant",
                content="Shall I proceed?",
                message_type="question",
                interactions=interactions,
                turn_id=None,
                attachments=None,
            )
        )
        db.commit()
        return int(task.id), int(user.id)
    finally:
        db.close()


async def _replayed_interactions(
    monkeypatch: pytest.MonkeyPatch, *, task_id: int, user_id: int
) -> list[list[dict]]:
    sent_events: list[dict] = []

    async def send_personal_message(event: dict, _websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr(websocket_api, "cache_get", lambda _key: None)
    monkeypatch.setattr(websocket_api, "cache_set", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        websocket_api.manager, "send_personal_message", send_personal_message
    )

    await websocket_api.send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )
    return [
        event["data"]["metadata"]["interactions"]
        for event in sent_events
        if event.get("event_type") == "agent_message"
        and "metadata" in event.get("data", {})
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("persisted", "expected"),
    [
        ([default_waiting_interaction()], []),
        ([default_waiting_interaction("Simplified Chinese")], []),
        ([{**default_waiting_interaction(), "field": "response_2"}], []),
        ([REAL_FIELD, default_waiting_interaction()], [REAL_FIELD]),
        ([REAL_FIELD], [REAL_FIELD]),
    ],
    ids=["substituted", "localized", "renamed", "beside_a_real_field", "real_only"],
)
async def test_replay_drops_only_the_substituted_field(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
    persisted: list[dict],
    expected: list[dict],
) -> None:
    task_id, user_id = _waiting_history_row(
        username=f"replay-{len(persisted)}-{persisted[0]['field']}",
        interactions=persisted,
    )

    replayed = await _replayed_interactions(
        monkeypatch, task_id=task_id, user_id=user_id
    )

    assert replayed == [expected]
