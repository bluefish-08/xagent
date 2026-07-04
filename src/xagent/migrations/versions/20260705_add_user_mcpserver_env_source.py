"""add env_source to user_mcpservers

Records which env layer a user picked for an MCP server: "own" | "shared" |
"platform". NULL keeps the legacy fallback (global < shared < user).

Revision ID: 20260705_add_user_mcpserver_env_source
Revises: 20260703_seed_google_maps_mcp_app
Create Date: 2026-07-05 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260705_add_user_mcpserver_env_source"
down_revision: Union[str, None] = "20260703_seed_google_maps_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_mcpservers",
        sa.Column("env_source", sa.String(length=16), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_mcpservers", "env_source")
