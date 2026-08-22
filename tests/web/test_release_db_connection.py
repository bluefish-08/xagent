"""Tests for release_db_connection_if_clean (issue #889) and session_scope."""

import pytest
from sqlalchemy import Column, Integer, String, create_engine, insert, text, update
from sqlalchemy.orm import declarative_base, sessionmaker

from xagent.web.models.database import release_db_connection_if_clean

Base = declarative_base()


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)


def _make_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


def test_releases_read_only_transaction():
    db = _make_session()
    db.query(Item).all()
    assert db.in_transaction()

    assert release_db_connection_if_clean(db) is True
    assert not db.in_transaction()

    # Session stays usable and re-acquires a connection on the next query.
    assert db.query(Item).all() == []


def test_keeps_pending_writes():
    db = _make_session()
    db.add(Item(name="pending"))

    assert release_db_connection_if_clean(db) is False
    assert Item in {type(obj) for obj in db.new} or len(db.new) == 1

    db.commit()
    assert db.query(Item).count() == 1


def test_keeps_flushed_but_uncommitted_changes():
    """flush() empties new/dirty/deleted while the transaction still holds
    unpersisted DML; the helper must not roll that back."""
    db = _make_session()
    db.add(Item(name="one"))
    db.flush()
    assert not (db.new or db.dirty or db.deleted)

    assert release_db_connection_if_clean(db) is False

    db.commit()
    assert db.query(Item).count() == 1

    # After the commit the flush flag is cleared: a fresh read-only
    # transaction is releasable again.
    db.query(Item).all()
    assert release_db_connection_if_clean(db) is True


def test_keeps_core_dml_insert():
    """Core DML via Session.execute() never touches new/dirty/deleted and
    emits no after_flush; the do_orm_execute listener must catch it."""
    db = _make_session()
    db.execute(insert(Item).values(name="core"))
    assert not (db.new or db.dirty or db.deleted)

    assert release_db_connection_if_clean(db) is False

    db.commit()
    assert db.query(Item).count() == 1


def test_keeps_core_dml_update():
    db = _make_session()
    db.add(Item(name="one"))
    db.commit()

    db.execute(update(Item).values(name="core-updated"))
    assert release_db_connection_if_clean(db) is False

    db.commit()
    assert db.query(Item).first().name == "core-updated"


def test_keeps_textual_statements_conservatively():
    """text() statements can't be proven read-only; the helper must keep the
    connection even for a textual SELECT."""
    db = _make_session()
    db.execute(text("UPDATE items SET name = 'via-text'"))
    assert release_db_connection_if_clean(db) is False
    db.commit()

    db.execute(text("SELECT 1"))
    assert release_db_connection_if_clean(db) is False
    db.rollback()

    # A provable ORM SELECT after the rollback is releasable again.
    db.query(Item).all()
    assert release_db_connection_if_clean(db) is True


def test_savepoint_commit_preserves_outer_write_flag():
    """Savepoint completion must not clear the write flag: after_commit-style
    events fire for begin_nested() too while the outer transaction (and its
    flushed writes) is still open."""
    db = _make_session()
    db.add(Item(name="outer"))
    db.flush()

    nested = db.begin_nested()
    nested.commit()

    assert release_db_connection_if_clean(db) is False
    db.commit()
    assert db.query(Item).count() == 1


def test_savepoint_rollback_preserves_outer_write_flag():
    db = _make_session()
    db.add(Item(name="outer"))
    db.flush()

    nested = db.begin_nested()
    nested.rollback()

    assert release_db_connection_if_clean(db) is False
    db.commit()
    assert db.query(Item).count() == 1


def test_root_rollback_clears_write_flag():
    db = _make_session()
    db.add(Item(name="discarded"))
    db.flush()
    db.rollback()

    db.query(Item).all()
    assert release_db_connection_if_clean(db) is True


def test_keeps_flushed_dirty_changes():
    db = _make_session()
    db.add(Item(name="one"))
    db.commit()

    item = db.query(Item).first()
    item.name = "changed"

    assert release_db_connection_if_clean(db) is False
    db.commit()
    assert db.query(Item).first().name == "changed"


def test_none_session_is_noop():
    assert release_db_connection_if_clean(None) is False


def test_no_transaction_returns_true():
    db = _make_session()
    assert release_db_connection_if_clean(db) is True


def test_session_scope_commits_a_write_and_leaves_no_transaction(tmp_path):
    """The unit of work is committed and the connection released on exit."""
    from xagent.web.models.background_job import BackgroundJob
    from xagent.web.models.database import init_db, session_scope
    from xagent.web.models.user import User

    init_db(f"sqlite:///{tmp_path / 'scope-commit.db'}")

    with session_scope() as db:
        user = User(username="scope-commit", password_hash="x")
        db.add(user)
        db.flush()
        user_id = int(user.id)

    with session_scope() as db:
        assert db.query(User).filter(User.id == user_id).first() is not None
        assert db.query(BackgroundJob).count() == 0


def test_session_scope_does_not_commit_a_read_only_scope(tmp_path):
    """A scope that only read has nothing to commit.

    Committing anyway bypasses the release helper every other call site uses
    for this exact problem class, and issues a COMMIT per read.
    """
    from sqlalchemy import event

    from xagent.web.models.database import get_engine, init_db, session_scope
    from xagent.web.models.user import User

    init_db(f"sqlite:///{tmp_path / 'scope-readonly.db'}")

    commits: list[int] = []
    event.listen(get_engine(), "commit", lambda _conn: commits.append(1))

    with session_scope() as db:
        db.query(User).all()

    assert commits == []


def test_session_scope_reraises_the_original_when_rollback_fails(tmp_path):
    """A rollback that fails on a dead connection must not mask the failure.

    That is precisely this path's target scenario: the connection the server
    terminated is the one the rollback runs on.
    """
    from xagent.web.models.database import init_db, session_scope

    init_db(f"sqlite:///{tmp_path / 'scope-rollback-fail.db'}")

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with session_scope() as db:
            db.rollback = lambda: (_ for _ in ()).throw(RuntimeError("connection gone"))
            raise _Boom("the original failure")
