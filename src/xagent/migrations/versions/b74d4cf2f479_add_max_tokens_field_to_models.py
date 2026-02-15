"""add max_tokens field to models

Revision ID: b74d4cf2f479
Revises: 441d4f5d399c
Create Date: 2025-11-03 21:09:44.186547

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b74d4cf2f479"
down_revision: Union[str, None] = "441d4f5d399c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("models", sa.Column("max_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "max_tokens")
