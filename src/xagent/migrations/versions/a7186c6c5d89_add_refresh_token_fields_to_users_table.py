"""Add refresh_token fields to users table

Revision ID: a7186c6c5d89
Revises: b74d4cf2f479
Create Date: 2025-11-14 11:56:11.622532

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7186c6c5d89"
down_revision: Union[str, None] = "b74d4cf2f479"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add refresh token fields to users table
    op.add_column(
        "users", sa.Column("refresh_token", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "users",
        sa.Column(
            "refresh_token_expires_at", sa.DateTime(timezone=True), nullable=True
        ),
    )


def downgrade() -> None:
    # Remove refresh token fields from users table
    op.drop_column("users", "refresh_token_expires_at")
    op.drop_column("users", "refresh_token")
