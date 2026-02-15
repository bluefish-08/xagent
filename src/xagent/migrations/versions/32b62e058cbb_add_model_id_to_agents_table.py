"""add model_id to agents table

Revision ID: 32b62e058cbb
Revises: 9800a4c3abe5
Create Date: 2026-01-31 23:17:50.576086

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "32b62e058cbb"
down_revision: Union[str, None] = "9800a4c3abe5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use batch mode for SQLite to add column with foreign key
    with op.batch_alter_table("agents", recreate="auto") as batch_op:
        batch_op.add_column(sa.Column("model_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_agents_model_id_models", "models", ["model_id"], ["id"]
        )


def downgrade() -> None:
    # Use batch mode for SQLite to drop foreign key and column
    with op.batch_alter_table("agents", recreate="auto") as batch_op:
        batch_op.drop_constraint("fk_agents_model_id_models", type_="foreignkey")
        batch_op.drop_column("model_id")
