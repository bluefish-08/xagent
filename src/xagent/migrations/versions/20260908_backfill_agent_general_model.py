"""backfill the general model slot for agents created without one

Revision ID: 20260908_backfill_agent_general_model
Revises: 20260904_add_auto_model_config
Create Date: 2026-09-08 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "20260908_backfill_agent_general_model"
down_revision: Union[str, tuple[str, str], None] = "20260904_add_auto_model_config"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Fill an empty `general` slot from the owner's default model.

    Server-side template creation passed no model config at all, so those
    agents persisted `models = NULL`: the builder rendered "--" for Main
    Model and refused to save until the owner picked one by hand. Runtime
    already fell back to the owner's default (`get_default_model`), so this
    only writes down what execution was resolving anyway.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if not {"agents", "user_default_models", "models"} <= table_names:
        return

    agent_columns = {col["name"] for col in inspector.get_columns("agents")}
    if not {"models", "user_id"} <= agent_columns:
        return

    agents = sa.table(
        "agents",
        sa.column("id", sa.Integer),
        sa.column("user_id", sa.Integer),
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
        return

    filled = 0
    skipped = 0
    rows = bind.execute(sa.select(agents.c.id, agents.c.user_id, agents.c.models))
    for row in rows.mappings():
        config = row["models"]
        if config is not None and not isinstance(config, dict):
            continue
        if config and config.get("general") is not None:
            continue
        model_id = default_by_user.get(int(row["user_id"]))
        if model_id is None:
            skipped += 1
            continue
        bind.execute(
            agents.update()
            .where(agents.c.id == row["id"])
            .values(models={**(config or {}), "general": model_id})
        )
        filled += 1

    if filled or skipped:
        logger.info(
            "Backfilled general model for %s agent(s); %s left empty "
            "(owner has no active general default)",
            filled,
            skipped,
        )


def downgrade() -> None:
    # Not reversed: a filled slot is indistinguishable from one the owner
    # picked themselves after upgrading, so clearing it would discard a
    # real choice.
    pass
