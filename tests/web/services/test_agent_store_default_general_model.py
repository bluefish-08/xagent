"""The ``general`` model slot is filled on both agent write paths.

Server-side creation paths pass no model config (``workforce_creator``
hardcodes ``models=None``), which persisted as an unset slot: the builder
rendered "--" and its required-field guard refused to save, and vibe
delegation failed outright. Every create path reaches
``AgentStore.add_agent``; ``migration/loaders.py`` builds its ``Agent``
outside the store and calls the same helper itself.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.migration.bundle import MigrationBundle, PersonaItem
from xagent.migration.loaders import MigrationLoader
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User, UserDefaultModel, UserModel
from xagent.web.services.agent_store import AgentStore, with_default_general_model


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _user_row(db: Session, username: str) -> User:
    user = User(username=username, password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _user(db: Session, username: str) -> int:
    return int(_user_row(db, username).id)


def _model(db: Session, model_id: str, *, is_active: bool = True) -> int:
    model = DBModel(
        model_id=model_id,
        category="llm",
        model_provider="p",
        model_name=model_id,
        api_key="k",
        is_active=is_active,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return int(model.id)


def _own(db: Session, user_id: int, model_pk: int, *, is_shared: bool = False) -> None:
    db.add(
        UserModel(
            user_id=user_id, model_id=model_pk, is_owner=True, is_shared=is_shared
        )
    )
    db.commit()


def _default(db: Session, user_id: int, model_pk: int) -> None:
    db.add(UserDefaultModel(user_id=user_id, model_id=model_pk, config_type="general"))
    db.commit()


@pytest.fixture()
def owner(db: Session) -> tuple[int, int]:
    """A user whose visible, active model is their ``general`` default."""
    user_id = _user(db, "owner")
    model_pk = _model(db, "gpt-x")
    _own(db, user_id, model_pk)
    _default(db, user_id, model_pk)
    return user_id, model_pk


def test_omitted_config_is_filled(db: Session, owner: tuple[int, int]) -> None:
    user_id, model_pk = owner
    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": model_pk
    }


def test_empty_dict_is_filled(db: Session, owner: tuple[int, int]) -> None:
    user_id, model_pk = owner
    assert with_default_general_model(db, {}, user_id=user_id) == {"general": model_pk}


def test_other_slots_are_preserved(db: Session, owner: tuple[int, int]) -> None:
    user_id, model_pk = owner
    filled = with_default_general_model(db, {"compact": 7}, user_id=user_id)
    assert filled == {"compact": 7, "general": model_pk}


def test_explicit_none_means_no_main_model(db: Session, owner: tuple[int, int]) -> None:
    user_id, _ = owner
    assert with_default_general_model(db, {"general": None}, user_id=user_id) == {
        "general": None
    }


def test_stated_choice_is_untouched(db: Session, owner: tuple[int, int]) -> None:
    user_id, _ = owner
    assert with_default_general_model(db, {"general": 42}, user_id=user_id) == {
        "general": 42
    }


def test_non_dict_payload_passes_through(db: Session, owner: tuple[int, int]) -> None:
    """Template YAML reaches this layer unvalidated; a typo must not 500."""
    user_id, _ = owner
    assert with_default_general_model(db, "gpt-4", user_id=user_id) == "gpt-4"


def test_no_default_leaves_config_unset(db: Session) -> None:
    user_id = _user(db, "no_default")
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_inactive_default_is_not_used(db: Session) -> None:
    user_id = _user(db, "inactive_default")
    model_pk = _model(db, "retired", is_active=False)
    _own(db, user_id, model_pk)
    _default(db, user_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_invisible_default_is_not_used(db: Session) -> None:
    """A default row survives the model becoming invisible (a demotion, a
    team move); injecting it would slip past ``_validate_models`` and leave
    the builder's Main Model blank while its save guard sees a value."""
    user_id = _user(db, "orphaned_default")
    stranger_id = _user(db, "stranger")
    model_pk = _model(db, "someone-elses")
    _own(db, stranger_id, model_pk)
    _default(db, user_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_shared_flag_alone_is_not_visibility(db: Session) -> None:
    """A team-membership change drops the sharer out of the owner's visible
    set without touching ``is_shared``, so the flag alone must not qualify."""
    user_id = _user(db, "borrower")
    sharer_id = _user(db, "sharer")
    model_pk = _model(db, "shared-llm")
    _own(db, sharer_id, model_pk, is_shared=True)
    _default(db, user_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_add_agent_fills_the_slot(db: Session, owner: tuple[int, int]) -> None:
    user_id, model_pk = owner
    agent = AgentStore(db).add_agent(
        user_id=user_id, name="server-made", description=None, instructions=None
    )
    db.commit()
    db.refresh(agent)
    assert agent.models == {"general": model_pk}


def _imported_agent(db: Session, user: User) -> Agent:
    bundle = MigrationBundle(source="hermes", source_root="x")
    bundle.persona = PersonaItem(instructions="You are helpful.")
    MigrationLoader(db, user=user).load(bundle)
    return db.query(Agent).filter(Agent.user_id == user.id).one()


def test_imported_agent_inherits_the_owners_default(db: Session) -> None:
    """The loader builds its Agent outside ``add_agent``, so a fallback wired
    into the store alone would miss this path."""
    user = _user_row(db, "importer")
    model_pk = _model(db, "gpt-x")
    _own(db, int(user.id), model_pk)
    _default(db, int(user.id), model_pk)
    assert _imported_agent(db, user).models == {"general": model_pk}


def test_imported_agent_without_a_default_keeps_its_empty_config(db: Session) -> None:
    user = _user_row(db, "importer_no_default")
    assert _imported_agent(db, user).models == {}
