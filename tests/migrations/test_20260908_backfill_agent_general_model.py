"""Tests for the agents.models general-slot backfill migration.

Run against SQLite and PostgreSQL both: the eligibility predicate compares
`models` cast to text (`Column(JSON)` with `none_as_null=False` stores an
unset config as the JSON text `null`, so `IS NULL` matches nothing), and that
cast is the one part of the statement whose behaviour differs per backend.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.models.agent import Agent, AgentOrigin
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


def _agent(db: Session, user: User, name: str, models: dict | None, **kwargs) -> Agent:
    agent = Agent(
        user_id=user.id,
        name=name,
        instructions="Be useful.",
        execution_mode="balanced",
        models=models,
        **kwargs,
    )
    db.add(agent)
    db.flush()
    return agent


def _owner_with_default(db: Session, username: str) -> tuple[User, int]:
    user = _user(db, username)
    model = _model(db, f"{username}-default")
    db.add(UserDefaultModel(user_id=user.id, model_id=model.id, config_type="general"))
    return user, int(model.id)


def _run_upgrade(db: Session) -> None:
    db.commit()
    with patch.object(MIGRATION.op, "get_bind", return_value=db.connection()):
        MIGRATION.upgrade()
    db.commit()


def _models_of(db: Session, agent_id: int) -> dict | None:
    return db.execute(sa.select(Agent.models).where(Agent.id == agent_id)).scalar_one()


def test_backfills_an_unset_config_from_the_owner_default(db: Session) -> None:
    """`None` reaches the column as the JSON text `null`, which is the shape
    every server-side creation path left behind."""
    user, model_id = _owner_with_default(db, "owner")
    agent_id = int(_agent(db, user, "Template agent", None).id)

    _run_upgrade(db)

    assert _models_of(db, agent_id) == {"general": model_id}


def test_leaves_a_config_that_already_has_a_value_alone(db: Session) -> None:
    user, _ = _owner_with_default(db, "owner")
    chosen = _model(db, "chosen")
    already_set = int(_agent(db, user, "Already set", {"general": chosen.id}).id)
    explicit_null = int(_agent(db, user, "Explicit null", {"general": None}).id)
    partial = int(_agent(db, user, "Partial", {"small_fast": 7}).id)
    chosen_id = int(chosen.id)

    _run_upgrade(db)

    assert _models_of(db, already_set) == {"general": chosen_id}
    assert _models_of(db, explicit_null) == {"general": None}
    # Out of scope by design: preserving sibling slots would need a
    # read-modify-write, and production holds no row of this shape.
    assert _models_of(db, partial) == {"small_fast": 7}


def test_skips_shared_and_team_agents(db: Session) -> None:
    """An unset config is resolved per runner against that user's own
    default, so filling it on a shared agent would switch everyone else onto
    the owner's model."""
    user, model_id = _owner_with_default(db, "owner")
    private = int(_agent(db, user, "Private", None).id)
    shared = int(_agent(db, user, "Shared", None, share_enabled=True).id)
    team = int(_agent(db, user, "Team", None, team_id=4242).id)

    _run_upgrade(db)

    assert _models_of(db, private) == {"general": model_id}
    assert _models_of(db, shared) is None
    assert _models_of(db, team) is None


def test_skips_the_hidden_workforce_manager_agent(db: Session) -> None:
    """Filtered out of every user-facing read path, so no UI ever showed
    "--" for one."""
    user, model_id = _owner_with_default(db, "owner")
    visible = int(_agent(db, user, "Visible", None).id)
    hidden = int(
        _agent(
            db,
            user,
            "Generated manager",
            None,
            origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        ).id
    )

    _run_upgrade(db)

    assert _models_of(db, visible) == {"general": model_id}
    assert _models_of(db, hidden) is None


def test_leaves_agents_whose_owner_has_no_usable_default(db: Session) -> None:
    """No default at all, and an inactive one, both leave the slot unset
    rather than writing an id the app would reject."""
    no_default = _user(db, "no-default")
    inactive_owner = _user(db, "inactive-default")
    retired = _model(db, "retired", is_active=False)
    db.add(
        UserDefaultModel(
            user_id=inactive_owner.id, model_id=retired.id, config_type="general"
        )
    )
    a = int(_agent(db, no_default, "No default", None).id)
    b = int(_agent(db, inactive_owner, "Inactive default", None).id)

    _run_upgrade(db)

    assert _models_of(db, a) is None
    assert _models_of(db, b) is None


def test_ignores_non_general_defaults(db: Session) -> None:
    user = _user(db, "owner")
    model = _model(db, "fast-only")
    db.add(
        UserDefaultModel(user_id=user.id, model_id=model.id, config_type="small_fast")
    )
    agent_id = int(_agent(db, user, "Fast only", None).id)

    _run_upgrade(db)

    assert _models_of(db, agent_id) is None


def test_fills_each_owner_from_their_own_default(db: Session) -> None:
    """One UPDATE per owner: nobody inherits another user's model."""
    first, first_model = _owner_with_default(db, "first")
    second, second_model = _owner_with_default(db, "second")
    a = int(_agent(db, first, "First agent", None).id)
    b = int(_agent(db, second, "Second agent", None).id)

    _run_upgrade(db)

    assert _models_of(db, a) == {"general": first_model}
    assert _models_of(db, b) == {"general": second_model}


def test_is_idempotent(db: Session) -> None:
    user, model_id = _owner_with_default(db, "owner")
    agent_id = int(_agent(db, user, "Template agent", None).id)

    _run_upgrade(db)
    _run_upgrade(db)

    assert _models_of(db, agent_id) == {"general": model_id}
