"""The ``general`` model slot is filled on both agent write paths.

Server-side creation paths pass no model config (``workforce_creator``
hardcodes ``models=None``), which persisted as an unset slot: the builder
rendered "--" and its required-field guard refused to save, and vibe
delegation failed outright. Every create path reaches
``AgentStore.add_agent``; ``migration/loaders.py`` builds its ``Agent``
outside the store and calls the same helper itself.

The resolution order is the one the LLM-returning resolvers use: the user's
own default, then a visible user's shared default.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import UniqueConstraint, create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.migration.bundle import MigrationBundle, PersonaItem
from xagent.migration.loaders import MigrationLoader
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User, UserDefaultModel, UserModel
from xagent.web.services.agent_store import AgentStore
from xagent.web.services.model_service import with_default_general_model


@pytest.fixture(autouse=True)
def _no_visibility_hook() -> Iterator[None]:
    """``set_visible_user_ids_hook`` is module-global and deployments install
    their own at import time, so pin the admin-default behaviour these cases
    assert on."""
    from xagent.web.services.model_service import set_visible_user_ids_hook

    set_visible_user_ids_hook(None)
    yield
    set_visible_user_ids_hook(None)


@pytest.fixture()
def legacy_db() -> Iterator[Session]:
    """A schema without ``uq_user_default_model``.

    The constraint is declared in the ORM metadata but appears in none of the
    alembic revisions, so a database created before it can hold several rows
    for one ``(user_id, config_type)`` pair. Dropping it from the metadata for
    one engine is the only way to build that shape.
    """
    table = UserDefaultModel.__table__
    dropped = [c for c in list(table.constraints) if isinstance(c, UniqueConstraint)]
    for constraint in dropped:
        table.constraints.discard(constraint)
    try:
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
        try:
            yield session
        finally:
            session.close()
    finally:
        for constraint in dropped:
            table.append_constraint(constraint)


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


def _user_row(db: Session, username: str, *, is_admin: bool = False) -> User:
    user = User(username=username, password_hash="x", is_admin=is_admin)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _user(db: Session, username: str, *, is_admin: bool = False) -> int:
    return int(_user_row(db, username, is_admin=is_admin).id)


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


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(None, {"general": "PK"}, id="omitted_is_filled"),
        pytest.param({}, {"general": "PK"}, id="empty_dict_is_filled"),
        pytest.param(
            {"compact": 7}, {"compact": 7, "general": "PK"}, id="other_slots_kept"
        ),
        # An explicit None means "no main model"; a stated id is a choice.
        pytest.param({"general": None}, {"general": None}, id="explicit_none_kept"),
        pytest.param({"general": 42}, {"general": 42}, id="stated_choice_kept"),
        # Template YAML reaches this layer unvalidated; a typo must not 500.
        pytest.param("gpt-4", "gpt-4", id="non_dict_passes_through"),
    ],
)
def test_fills_only_an_omitted_slot(
    db: Session, owner: tuple[int, int], payload: Any, expected: Any
) -> None:
    _, model_pk = owner
    if isinstance(expected, dict):
        expected = {k: (model_pk if v == "PK" else v) for k, v in expected.items()}
    assert with_default_general_model(db, payload, user_id=owner[0]) == expected


def test_a_default_for_another_slot_is_not_used(db: Session) -> None:
    user_id = _user(db, "visual_only")
    model_pk = _model(db, "vision-llm")
    _own(db, user_id, model_pk)
    db.add(UserDefaultModel(user_id=user_id, model_id=model_pk, config_type="visual"))
    db.commit()
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_an_older_own_default_is_used_when_the_newest_is_invisible(
    legacy_db: Session,
) -> None:
    """The newest row becoming invisible must not skip the remaining ones."""
    db = legacy_db
    user_id = _user(db, "two_rows")
    stranger_id = _user(db, "ex_teammate")
    visible_pk = _model(db, "kept-llm")
    gone_pk = _model(db, "lost-llm")
    _own(db, user_id, visible_pk)
    _own(db, stranger_id, gone_pk)
    _default(db, user_id, visible_pk)
    # The later row wins the ordering but points at a model this user lost.
    _default(db, user_id, gone_pk)
    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": visible_pk
    }


def test_no_default_anywhere_leaves_config_unset(db: Session) -> None:
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


def test_a_non_visible_sharer_does_not_qualify(db: Session) -> None:
    """``is_shared`` alone is not visibility: the sharer also has to be in the
    user's visible set, which ``_get_visible_user_ids`` resolves (admins by
    default, whatever the deployment's hook says otherwise)."""
    user_id = _user(db, "borrower")
    sharer_id = _user(db, "sharer")
    model_pk = _model(db, "shared-llm")
    _own(db, sharer_id, model_pk, is_shared=True)
    _default(db, user_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_a_visible_sharers_shared_default_is_inherited(db: Session) -> None:
    """A user whose only usable default is an admin-shared one must not be
    treated as having none -- that is the bug this PR fixes, on another path
    (``agent_tool.py`` already resolves through this fallback)."""
    user_id = _user(db, "no_own_default")
    admin_id = _user(db, "sharing_admin", is_admin=True)
    model_pk = _model(db, "admin-llm")
    _own(db, admin_id, model_pk, is_shared=True)
    _default(db, admin_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": model_pk
    }


def test_an_unshared_admin_default_is_not_inherited(db: Session) -> None:
    user_id = _user(db, "outsider")
    admin_id = _user(db, "private_admin", is_admin=True)
    model_pk = _model(db, "admin-private-llm")
    _own(db, admin_id, model_pk, is_shared=False)
    _default(db, admin_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_an_inactive_shared_default_is_not_inherited(db: Session) -> None:
    user_id = _user(db, "late_comer")
    admin_id = _user(db, "retiring_admin", is_admin=True)
    model_pk = _model(db, "admin-retired-llm", is_active=False)
    _own(db, admin_id, model_pk, is_shared=True)
    _default(db, admin_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_add_agent_fills_the_slot(db: Session, owner: tuple[int, int]) -> None:
    user_id, model_pk = owner
    agent = AgentStore(db).add_agent(
        user_id=user_id, name="server-made", description=None, instructions=None
    )
    db.commit()
    db.refresh(agent)
    assert agent.models == {"general": model_pk}


def test_a_stranger_pointing_at_a_shared_model_does_not_qualify(
    db: Session,
) -> None:
    """The shared layer keys on who OWNS the default row, not on who shared
    the model -- otherwise any unrelated user adopting an admin-shared model
    as their own default would hand it to everyone."""
    user_id = _user(db, "bystander")
    admin_id = _user(db, "silent_admin", is_admin=True)
    stranger_id = _user(db, "stranger_with_taste")
    model_pk = _model(db, "adopted-llm")
    _own(db, admin_id, model_pk, is_shared=True)
    # The admin shares the model but never made it their own default.
    _default(db, stranger_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_an_invisible_own_default_falls_through_to_the_shared_layer(
    db: Session,
) -> None:
    """An own default pointing at a model the user can no longer see must not
    short-circuit the shared layer (``agent_tool`` skips such a slot and keeps
    filling from the shared defaults)."""
    user_id = _user(db, "demoted")
    stranger_id = _user(db, "former_teammate")
    admin_id = _user(db, "sharing_admin_2", is_admin=True)
    gone_pk = _model(db, "no-longer-visible")
    shared_pk = _model(db, "still-visible")
    _own(db, stranger_id, gone_pk)
    _own(db, admin_id, shared_pk, is_shared=True)
    _default(db, user_id, gone_pk)
    _default(db, admin_id, shared_pk)
    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": shared_pk
    }


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
