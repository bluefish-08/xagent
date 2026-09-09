"""The `general` model slot is filled where the creation paths converge
(rogercloud review on #2229, finding 1).

`AgentStore.add_agent` is reached by `create_agent` (plain `POST /api/agents`,
the vibe agent tool), by `AgentManagementService.create_agent_with_optional_key`
(template creates, `/v1/agents`), and directly by `workforce_creator`. Filling
the slot in any one caller leaves the others persisting an unset config, which
the builder renders as "--". `migration/loaders.py` builds its `Agent` outside
the store and calls the helper itself; `test_platform_migration` covers it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User, UserDefaultModel, UserModel
from xagent.web.services.agent_store import AgentStore


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield session
    finally:
        session.close()
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


@pytest.fixture()
def owner(db: Session) -> User:
    user = User(username="owner", password_hash="x", is_admin=False)
    db.add(user)
    db.flush()
    return user


@pytest.fixture()
def default_model(db: Session, owner: User) -> DBModel:
    model = _model(db, "gpt-4o")
    db.add(UserModel(user_id=owner.id, model_id=model.id, is_owner=True))
    db.add(UserDefaultModel(user_id=owner.id, model_id=model.id, config_type="general"))
    db.commit()
    return model


def _add(db: Session, owner: User, name: str, models: object) -> object:
    agent = AgentStore(db).add_agent(
        user_id=int(owner.id),
        name=name,
        description=None,
        instructions="Be useful.",
        models=models,
    )
    db.commit()
    return agent.models


def test_omitted_config_is_filled_from_the_owner_default(
    db: Session, owner: User, default_model: DBModel
) -> None:
    assert _add(db, owner, "No config", None) == {"general": int(default_model.id)}


def test_empty_config_is_filled_from_the_owner_default(
    db: Session, owner: User, default_model: DBModel
) -> None:
    assert _add(db, owner, "Empty config", {}) == {"general": int(default_model.id)}


def test_an_explicit_choice_survives_unmodified(
    db: Session, owner: User, default_model: DBModel
) -> None:
    """Blocking criterion #3 of the PR: a caller-supplied payload is not
    rewritten by the fallback."""
    chosen = _model(db, "gpt-4o-mini")
    db.add(UserModel(user_id=owner.id, model_id=chosen.id, is_owner=True))
    db.commit()
    payload = {"general": int(chosen.id), "small_fast": int(default_model.id)}

    assert _add(db, owner, "Explicit", dict(payload)) == payload


def test_an_explicit_null_general_is_left_alone(
    db: Session, owner: User, default_model: DBModel
) -> None:
    """`_validate_models` honours an explicit null, so the fallback must not
    overwrite it into a model id."""
    assert _add(db, owner, "Explicit null", {"general": None}) == {"general": None}


def test_other_slots_are_preserved_while_general_is_filled(
    db: Session, owner: User, default_model: DBModel
) -> None:
    assert _add(db, owner, "Partial", {"small_fast": 7}) == {
        "small_fast": 7,
        "general": int(default_model.id),
    }


def test_a_non_dict_payload_passes_through_untouched(
    db: Session, owner: User, default_model: DBModel
) -> None:
    """`workforce_creator` forwards `agent_config["models"]` straight from
    unvalidated template YAML. Rejecting the shape here would turn authoring
    typos into a 500; it is stored as-is, exactly as before this fallback
    existed."""
    assert _add(db, owner, "Malformed", ["not", "a", "dict"]) == ["not", "a", "dict"]


def test_no_usable_owner_default_leaves_the_config_untouched(db: Session) -> None:
    """No default at all, and an inactive one, both leave the slot unset
    rather than writing an id the app would reject."""
    no_default = User(username="no-default", password_hash="x", is_admin=False)
    inactive_owner = User(username="inactive", password_hash="x", is_admin=False)
    db.add_all([no_default, inactive_owner])
    db.flush()
    retired = _model(db, "retired", is_active=False)
    db.add(UserModel(user_id=inactive_owner.id, model_id=retired.id, is_owner=True))
    db.add(
        UserDefaultModel(
            user_id=inactive_owner.id, model_id=retired.id, config_type="general"
        )
    )
    db.commit()

    assert _add(db, no_default, "No default", None) is None
    assert _add(db, inactive_owner, "Inactive default", None) is None


def test_a_shared_default_is_not_consulted(db: Session, owner: User) -> None:
    """Deliberately unlike ModelStore.get_user_default_models: the backfill
    migration cannot replicate that shared/admin-default layer without
    freezing its visibility rules into a historical revision, so neither
    side uses it and an agent's slot never depends on when it was created."""
    other = User(username="sharer", password_hash="x", is_admin=True)
    db.add(other)
    db.flush()
    shared = _model(db, "shared-model")
    db.add(
        UserModel(user_id=other.id, model_id=shared.id, is_owner=True, is_shared=True)
    )
    db.add(
        UserDefaultModel(user_id=other.id, model_id=shared.id, config_type="general")
    )
    db.commit()

    assert _add(db, owner, "No personal default", None) is None
