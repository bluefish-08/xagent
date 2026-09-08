"""backfill the general model slot for agents created without one

Revision ID: 20260908_backfill_agent_general_model
Revises: 20260901_seed_zendesk_mcp_app
Create Date: 2026-09-08 00:00:00.000000

"""

import json
import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1000

revision: str = "20260908_backfill_agent_general_model"
down_revision: Union[str, tuple[str, str], None] = "20260901_seed_zendesk_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Fill an unset `general` slot from the owner's default model.

    Server-side agent creation passed no model config at all, so those agents
    persisted an empty config: the builder rendered "--" for Main Model and
    refused to save until the owner picked one by hand. Runtime already fell
    back to the owner's default (`get_default_model`), so this only writes
    down what execution was resolving anyway.

    Note that `Agent.models` is `Column(JSON)` with SQLAlchemy's default
    `none_as_null=False`, so an unset config is stored as the JSON text
    `null`, not SQL NULL. `WHERE models IS NULL` matches none of these rows,
    which is why each payload is read back and inspected in Python.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if not {"agents", "user_default_models", "models"} <= table_names:
        return

    agent_columns = {col["name"] for col in inspector.get_columns("agents")}
    if not {"models", "user_id"} <= agent_columns:
        return

    # Read `models` as text and deserialize per row: binding it as sa.JSON
    # would decode during result iteration, where one malformed payload
    # aborts the whole run instead of just its own row. On PostgreSQL the
    # driver still hands back a decoded object for a json column, so the
    # loop below accepts either shape. Writes go through the JSON-typed
    # table so the value is encoded the same way the ORM encodes it.
    agents_read = sa.table(
        "agents",
        sa.column("id", sa.Integer),
        sa.column("user_id", sa.Integer),
        sa.column("models", sa.Text),
    )
    agents_write = sa.table(
        "agents",
        sa.column("id", sa.Integer),
        sa.column("models", sa.JSON),
    )
    user_defaults = sa.table(
        "user_default_models",
        sa.column("user_id", sa.Integer),
        sa.column("model_id", sa.Integer),
        sa.column("config_type", sa.String),
    )
    db_models = sa.table(
        "models",
        sa.column("id", sa.Integer),
        sa.column("is_active", sa.Boolean),
    )

    # Only the owner's own default is used. Resolving the shared-default
    # fallback that ModelStore applies would mean replicating its
    # visibility rules here, which drift as that code changes.
    default_by_user = {
        int(row["user_id"]): int(row["model_id"])
        for row in bind.execute(
            sa.select(user_defaults.c.user_id, user_defaults.c.model_id)
            .join(db_models, user_defaults.c.model_id == db_models.c.id)
            .where(
                user_defaults.c.config_type == "general",
                db_models.c.is_active.is_(True),
            )
        ).mappings()
    }
    if not default_by_user:
        logger.info(
            "No user has an active general default model; "
            "no agent model config was backfilled"
        )
        return

    # Ids first, payloads a chunk at a time: `models` is small but the row
    # count is unbounded, and Connection.execute buffers a whole result set.
    # Reading ids to exhaustion also keeps the updates below off an open
    # cursor over the same table. Chunks are contiguous slices of the sorted
    # id list, so they are addressed by range rather than by an IN list --
    # SQLite caps bound parameters at 999 before 3.32.0.
    agent_ids = list(
        bind.execute(sa.select(agents_read.c.id).order_by(agents_read.c.id)).scalars()
    )

    filled = 0
    skipped = 0
    malformed = 0
    for start in range(0, len(agent_ids), _CHUNK_SIZE):
        chunk = agent_ids[start : start + _CHUNK_SIZE]
        rows = bind.execute(
            sa.select(
                agents_read.c.id, agents_read.c.user_id, agents_read.c.models
            ).where(agents_read.c.id.between(chunk[0], chunk[-1]))
        )
        for row in rows.mappings():
            raw = row["models"]
            if isinstance(raw, (str, bytes)):
                try:
                    config = json.loads(raw)
                except ValueError:
                    malformed += 1
                    continue
            else:
                config = raw
            if config is not None and not isinstance(config, dict):
                malformed += 1
                continue
            # An explicit null is a deliberate "no main model" and is left
            # alone, matching AgentManagementService's model validation.
            if config and "general" in config:
                continue
            model_id = default_by_user.get(int(row["user_id"]))
            if model_id is None:
                skipped += 1
                continue
            bind.execute(
                agents_write.update()
                .where(agents_write.c.id == row["id"])
                .values(models={**(config or {}), "general": model_id})
            )
            filled += 1

    logger.info(
        "Backfilled general model for %s agent(s); %s left empty (owner has "
        "no active general default), %s skipped as unreadable config",
        filled,
        skipped,
        malformed,
    )


def downgrade() -> None:
    # Not reversed: a filled slot is indistinguishable from one the owner
    # picked themselves after upgrading, so clearing it would discard a
    # real choice.
    pass
