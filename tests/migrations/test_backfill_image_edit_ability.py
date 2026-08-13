"""Tests for the 20260813_backfill_image_edit_ability migration."""

import json
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

REVISION = "20260813_backfill_image_edit_ability"
PREVIOUS_REVISION = "20260812_seed_intercom_mcp_app"

ROWS = [
    ("img-qwen-edit", "image", "qwen-image-edit", ["generate"]),
    ("img-gemini-3pro", "image", "gemini-3-pro-image-preview", ["generate"]),
    ("img-gpt", "image", "gpt-image-1", ["generate"]),
    ("img-plain", "image", "wanx-v1", ["generate"]),
    ("img-already", "image", "qwen-image-edit-plus", ["generate", "edit"]),
    ("img-null", "image", "some-edit-model", None),
    ("llm-edit", "llm", "gpt-edit-sounding-name", ["chat"]),
]


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(db_url)

    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE models ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "model_id VARCHAR(100), category VARCHAR(20), "
                "model_name VARCHAR(100), abilities JSON)"
            )
        )
        conn.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
            {"rev": PREVIOUS_REVISION},
        )
        for model_id, category, model_name, abilities in ROWS:
            conn.execute(
                text(
                    "INSERT INTO models (model_id, category, model_name, abilities) "
                    "VALUES (:model_id, :category, :model_name, :abilities)"
                ),
                {
                    "model_id": model_id,
                    "category": category,
                    "model_name": model_name,
                    "abilities": None if abilities is None else json.dumps(abilities),
                },
            )

    config = Config()
    config.set_main_option("sqlalchemy.url", db_url)
    config.set_main_option("script_location", "src/xagent/migrations")
    command.upgrade(config, REVISION)
    return engine


def _abilities(engine: Any) -> dict[str, list[str]]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT model_id, abilities FROM models")).fetchall()
    result = {}
    for row in rows:
        value = row.abilities
        result[row.model_id] = json.loads(value) if isinstance(value, str) else value
    return result


def test_edit_added_only_to_image_models_whose_name_declares_it(engine: Any) -> None:
    abilities = _abilities(engine)

    assert abilities["img-qwen-edit"] == ["generate", "edit"]
    assert abilities["img-gemini-3pro"] == ["generate", "edit"]
    assert abilities["img-gpt"] == ["generate", "edit"]
    # A name with no edit marker keeps generate-only: guessing here would put back
    # the doomed edit_image the capability gating exists to withhold.
    assert abilities["img-plain"] == ["generate"]
    assert abilities["img-already"] == ["generate", "edit"]
    assert abilities["img-null"] == ["generate", "edit"]
    assert abilities["llm-edit"] == ["chat"]


def test_upgrade_skips_a_database_without_the_models_table(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'empty.db'}"
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
            {"rev": PREVIOUS_REVISION},
        )

    config = Config()
    config.set_main_option("sqlalchemy.url", db_url)
    config.set_main_option("script_location", "src/xagent/migrations")
    command.upgrade(config, REVISION)


def test_update_casts_the_json_parameter() -> None:
    """Postgres rejects a text parameter into a json column; SQLite accepts it.

    Every other test here runs on SQLite, so nothing else fails if the cast goes.
    """
    import importlib.util  # noqa: PLC0415

    from sqlalchemy.dialects.postgresql import psycopg2  # noqa: PLC0415

    path = Path("src/xagent/migrations/versions") / f"{REVISION}.py"
    spec = importlib.util.spec_from_file_location(REVISION, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    compiled = str(module.UPDATE_ABILITIES.compile(dialect=psycopg2.dialect()))
    assert "::JSON" in compiled
