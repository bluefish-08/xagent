"""Tests for the agents.models general-slot backfill migration."""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User, UserDefaultModel

MIGRATION = importlib.import_module(
    "xagent.migrations.versions.20260908_backfill_agent_general_model"
)


# Tables this test touches, in dependency order.
_TABLE_NAMES = ("users", "models", "user_default_models", "agents")
_METADATA_TABLES = sa.MetaData()
for _name in _TABLE_NAMES:
    Base.metadata.tables[_name].to_metadata(_METADATA_TABLES)


def _drop_tables(engine) -> None:
    with engine.begin() as conn:
        for name in reversed(_TABLE_NAMES):
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {name} CASCADE"))


def _postgres_url() -> str | None:
    return os.getenv("XAGENT_TEST_POSTGRES_URL") or os.getenv(
        "POSTGRES_TEST_DATABASE_URL"
    )


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param("postgresql", marks=pytest.mark.postgresql),
    ]
)
def db(request: pytest.FixtureRequest) -> Session:
    """Both backends, because the payload arrives in different shapes.

    SQLite stores `models` as text, so the migration receives a JSON string
    and decodes it. PostgreSQL's json column is decoded by the driver, so the
    migration receives an object -- a branch no SQLite run can reach.
    """
    if request.param == "postgresql":
        url = _postgres_url()
        if not url:
            pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
        engine = create_engine(url)
        # Only this test's own tables, dropped by name: the Postgres URL is a
        # shared test database, so create_all/drop_all over the full metadata
        # would tear down whatever else is using it.
        _drop_tables(engine)
        _METADATA_TABLES.create_all(bind=engine)
    else:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        if request.param == "postgresql":
            _drop_tables(engine)
        engine.dispose()


def _model(db: Session, model_id: str, *, is_active: bool = True) -> DBModel:
    model = DBModel(
        model_id=model_id,
        category="llm",
        model_provider="openai",
        model_name=model_id,
        api_key="test-api-key",
        base_url="https://api.openai.com/v1",
        is_active=is_active,
    )
    db.add(model)
    db.flush()
    return model


def _user(db: Session, username: str) -> User:
    user = User(username=username, password_hash="hash")
    db.add(user)
    db.flush()
    return user


def _agent(db: Session, user: User, name: str, models: dict | None) -> Agent:
    agent = Agent(
        user_id=user.id,
        name=name,
        instructions="Be useful.",
        execution_mode="balanced",
        models=models,
    )
    db.add(agent)
    db.flush()
    return agent


def _run_upgrade(db: Session) -> None:
    db.commit()
    with patch.object(MIGRATION.op, "get_bind", return_value=db.connection()):
        MIGRATION.upgrade()
    db.commit()


def _models_of(db: Session, agent_id: int) -> dict | None:
    return db.execute(sa.select(Agent.models).where(Agent.id == agent_id)).scalar_one()


def test_backfills_null_config_from_owner_default(db: Session) -> None:
    user = _user(db, "owner")
    model = _model(db, "gpt-4o")
    db.add(UserDefaultModel(user_id=user.id, model_id=model.id, config_type="general"))
    agent = _agent(db, user, "Template agent", None)
    agent_id, model_id = int(agent.id), int(model.id)

    _run_upgrade(db)

    assert _models_of(db, agent_id) == {"general": model_id}


def test_preserves_other_slots_and_respects_an_explicit_general(db: Session) -> None:
    """An omitted `general` is filled while other slots survive; a value the
    owner already chose, including an explicit null, is left alone."""
    user = _user(db, "owner")
    default_model = _model(db, "gpt-4o")
    chosen = _model(db, "gpt-4o-mini")
    db.add(
        UserDefaultModel(
            user_id=user.id, model_id=default_model.id, config_type="general"
        )
    )
    partial = _agent(db, user, "Partial", {"small_fast": 7})
    already_set = _agent(db, user, "Already set", {"general": chosen.id})
    explicit_null = _agent(db, user, "Explicit null", {"general": None})
    partial_id, already_id = int(partial.id), int(already_set.id)
    null_id = int(explicit_null.id)
    default_id, chosen_id = int(default_model.id), int(chosen.id)

    _run_upgrade(db)

    assert _models_of(db, partial_id) == {"small_fast": 7, "general": default_id}
    assert _models_of(db, already_id) == {"general": chosen_id}
    assert _models_of(db, null_id) == {"general": None}


def test_malformed_payload_does_not_abort_the_run(db: Session) -> None:
    """Nothing deserializes a payload up front, so a row holding invalid JSON
    cannot take the whole backfill down with it (rogercloud review, finding 2).

    SQLite-only by construction: `models` is TEXT there and accepts anything,
    while PostgreSQL's json column rejects invalid JSON at write time, so the
    bad row cannot exist to begin with.
    """
    if db.bind.dialect.name != "sqlite":
        pytest.skip("a json column rejects invalid JSON at write time")
    user = _user(db, "owner")
    model = _model(db, "gpt-4o")
    db.add(UserDefaultModel(user_id=user.id, model_id=model.id, config_type="general"))
    target = _agent(db, user, "Needs backfill", None)
    broken = _agent(db, user, "Broken payload", None)
    target_id, broken_id, model_id = int(target.id), int(broken.id), int(model.id)
    db.commit()
    db.execute(
        sa.text("UPDATE agents SET models = :junk WHERE id = :id"),
        {"junk": "{not json", "id": broken_id},
    )

    _run_upgrade(db)

    assert _models_of(db, target_id) == {"general": model_id}
    assert (
        db.execute(
            sa.text("SELECT models FROM agents WHERE id = :id"), {"id": broken_id}
        ).scalar_one()
        == "{not json"
    )


def test_leaves_agent_untouched_without_usable_owner_default(db: Session) -> None:
    """No default at all, and an inactive one, both leave the slot empty
    rather than writing an id the app would reject as unusable."""
    no_default = _user(db, "no-default")
    inactive_default = _user(db, "inactive-default")
    inactive = _model(db, "retired-model", is_active=False)
    db.add(
        UserDefaultModel(
            user_id=inactive_default.id, model_id=inactive.id, config_type="general"
        )
    )
    a = _agent(db, no_default, "No default", None)
    b = _agent(db, inactive_default, "Inactive default", None)
    a_id, b_id = int(a.id), int(b.id)

    _run_upgrade(db)

    assert _models_of(db, a_id) is None
    assert _models_of(db, b_id) is None


def test_ignores_non_general_defaults(db: Session) -> None:
    user = _user(db, "owner")
    model = _model(db, "gpt-4o-mini")
    db.add(
        UserDefaultModel(user_id=user.id, model_id=model.id, config_type="small_fast")
    )
    agent = _agent(db, user, "Fast only", None)
    agent_id = int(agent.id)

    _run_upgrade(db)

    assert _models_of(db, agent_id) is None


def test_processes_agents_past_the_first_chunk(db: Session) -> None:
    """Payloads are read a chunk at a time, addressed by id range rather than
    an IN list (SQLite caps bound parameters at 999 before 3.32.0), so a row
    in a later chunk must be backfilled too."""
    user = _user(db, "owner")
    model = _model(db, "gpt-4o")
    db.add(UserDefaultModel(user_id=user.id, model_id=model.id, config_type="general"))
    agents = [_agent(db, user, f"Agent {i}", None) for i in range(5)]
    agent_ids = [int(a.id) for a in agents]
    model_id = int(model.id)

    with patch.object(MIGRATION, "_CHUNK_SIZE", 2):
        _run_upgrade(db)

    assert [_models_of(db, i) for i in agent_ids] == [{"general": model_id}] * 5


def test_is_idempotent(db: Session) -> None:
    user = _user(db, "owner")
    model = _model(db, "gpt-4o")
    db.add(UserDefaultModel(user_id=user.id, model_id=model.id, config_type="general"))
    agent = _agent(db, user, "Template agent", None)
    agent_id, model_id = int(agent.id), int(model.id)

    _run_upgrade(db)
    _run_upgrade(db)

    assert _models_of(db, agent_id) == {"general": model_id}
