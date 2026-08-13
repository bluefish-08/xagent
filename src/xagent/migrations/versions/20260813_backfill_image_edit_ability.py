"""backfill the edit ability on image models whose name declares it

Revision ID: 20260813_backfill_image_edit_ability
Revises: 20260812_seed_intercom_mcp_app
Create Date: 2026-08-13 00:00:00.000000

"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260813_backfill_image_edit_ability"
down_revision: Union[str, None] = "20260812_seed_intercom_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The same markers core/model/image/gemini.py already infers from, so this only
# writes down what the runtime would have concluded from the model name anyway.
EDIT_CAPABLE_NAME_MARKERS = ("edit", "3-pro", "gpt-image")

# sa.JSON, not json.dumps: a text parameter into Postgres' json column raises
# "expression is of type text". SQLite hides this -- its JSON is text.
UPDATE_ABILITIES = sa.text(
    "UPDATE models SET abilities = :abilities WHERE id = :id"
).bindparams(sa.bindparam("abilities", type_=sa.JSON))


def _declares_edit(model_name: str) -> bool:
    lowered = (model_name or "").lower()
    return any(marker in lowered for marker in EDIT_CAPABLE_NAME_MARKERS)


def upgrade() -> None:
    conn = op.get_bind()
    # models is created by metadata, not by a migration, so a fresh database can
    # reach this revision before the table exists.
    if not sa.inspect(conn).has_table("models"):
        return

    rows = conn.execute(
        sa.text("SELECT id, model_name, abilities FROM models WHERE category = 'image'")
    ).fetchall()

    for row in rows:
        abilities = row.abilities
        if isinstance(abilities, str):
            abilities = json.loads(abilities)
        abilities = list(abilities or ["generate"])
        if "edit" in abilities or not _declares_edit(row.model_name):
            continue
        conn.execute(
            UPDATE_ABILITIES, {"abilities": abilities + ["edit"], "id": row.id}
        )


def downgrade() -> None:
    # Not reversible: an "edit" written here is indistinguishable from one an
    # operator configured by hand, and dropping theirs would disable a live tool.
    pass
