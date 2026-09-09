"""backfill the general model slot for agents created without one

Revision ID: 20260908_backfill_agent_general_model
Revises: 370740d9125a
Create Date: 2026-09-08 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# Hidden from every user-facing read path; see the scope note in upgrade().
WORKFORCE_MANAGER_ORIGIN = "workforce_generated_manager"

revision: str = "20260908_backfill_agent_general_model"
down_revision: Union[str, tuple[str, str], None] = "370740d9125a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Fill an unset `general` slot from the owner's default model.

    Server-side agent creation passed no model config at all, so those agents
    persisted an unset config: the builder rendered "--" for Main Model and
    refused to save until the owner picked one by hand.

    `Agent.models` is `Column(JSON)` with SQLAlchemy's default
    `none_as_null=False`, so an unset config is the JSON text `null`, **not**
    SQL NULL -- `WHERE models IS NULL` matches none of these rows.

    Scope, and why it is this narrow:

    * Only an empty config: JSON `null` (what every server-side creation
      path left) or `{}` (what `migration/loaders.py` wrote for imported
      agents before this change). Both render as "--". A *non-empty* object
      that merely omits `general` is left alone -- preserving its other
      slots would need a read-modify-write, and no creation path produces
      that shape.
    * Only private agents (`team_id IS NULL AND NOT share_enabled`). An
      unset config is resolved per runner at execution time against *that
      user's* default (`llm_utils.resolve_llms_from_names` ->
      `get_configured_defaults(user_id)`), so filling it on a shared agent
      would silently switch every other runner onto the owner's model.
      Private agents have exactly one runner, so there is no such change.
    * Not `workforce_generated_manager`, which `list_agent_items` and
      `get_owned_agent` filter out of every user-facing read: no UI ever
      showed "--" for these, so filling them buys nothing.

    One UPDATE per owner, so the row's state is re-tested by the database at
    write time. A read-then-write would let a `PUT /api/agents/{id}` from an
    old replica during a rolling deploy land in between and be clobbered.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if not {"agents", "user_default_models", "models", "user_models"} <= table_names:
        return

    agent_columns = {col["name"] for col in inspector.get_columns("agents")}
    required = {"models", "user_id", "team_id", "share_enabled", "origin"}
    if not required <= agent_columns:
        return

    agents = sa.table(
        "agents",
        sa.column("id", sa.Integer),
        sa.column("user_id", sa.Integer),
        sa.column("team_id", sa.Integer),
        sa.column("share_enabled", sa.Boolean),
        sa.column("origin", sa.String),
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
    user_models = sa.table(
        "user_models",
        sa.column("model_id", sa.Integer),
        sa.column("user_id", sa.Integer),
        sa.column("is_shared", sa.Boolean),
    )

    # Only the owner's own default, matching AgentStore's create-time
    # fallback. Neither side consults ModelStore's shared-default layer, so
    # an agent's slot never depends on when it was created.
    #
    # The reachability test is deliberately weaker than the application's:
    # `_is_model_visible_to_user` resolves visibility through the current
    # team graph, and freezing a copy of those rules into a historical
    # revision would leave this statement asserting something that stops
    # being true. Requiring a `user_models` row that the owner either holds
    # or that someone shares is stable, and it still keeps an orphaned
    # default -- one whose model rows are all gone -- from being written
    # into an agent where the builder would render it blank.
    reachable = sa.exists().where(
        sa.and_(
            user_models.c.model_id == user_defaults.c.model_id,
            sa.or_(
                user_models.c.user_id == user_defaults.c.user_id,
                user_models.c.is_shared.is_(True),
            ),
        )
    )
    default_by_user = {
        int(row["user_id"]): int(row["model_id"])
        for row in bind.execute(
            sa.select(user_defaults.c.user_id, user_defaults.c.model_id)
            .join(db_models, user_defaults.c.model_id == db_models.c.id)
            .where(
                user_defaults.c.config_type == "general",
                db_models.c.is_active.is_(True),
                reachable,
            )
        ).mappings()
    }
    if not default_by_user:
        logger.info(
            "No user has an active general default model; "
            "no agent model config was backfilled"
        )
        return

    # An empty config compares as its text form on both backends: SQLite
    # stores the column as TEXT, and PostgreSQL's json has no equality
    # operator to use instead. Casting is what keeps this one statement
    # dialect-agnostic. Both spellings are what json.dumps emits, so no
    # whitespace variant can hide from this.
    unset = sa.cast(agents.c.models, sa.Text).in_(("null", "{}"))
    eligible = sa.and_(
        agents.c.team_id.is_(None),
        agents.c.share_enabled.is_(False),
        agents.c.origin != WORKFORCE_MANAGER_ORIGIN,
        unset,
    )

    filled = 0
    for user_id, model_id in default_by_user.items():
        result = bind.execute(
            agents.update()
            .where(agents.c.user_id == user_id, eligible)
            .values(models={"general": model_id})
        )
        filled += max(result.rowcount or 0, 0)

    remaining = bind.execute(
        sa.select(sa.func.count()).select_from(agents).where(eligible)
    ).scalar_one()
    logger.info(
        "Backfilled general model for %s agent(s); %s left unset (owner has "
        "no active general default)",
        filled,
        remaining,
    )


def downgrade() -> None:
    # Not reversed: a filled slot is indistinguishable from one the owner
    # picked themselves after upgrading, so clearing it would discard a
    # real choice.
    pass
