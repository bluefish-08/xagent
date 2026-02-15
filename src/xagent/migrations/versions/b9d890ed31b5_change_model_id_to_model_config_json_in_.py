"""change model_id to model_config json in agents

Revision ID: b9d890ed31b5
Revises: 32b62e058cbb
Create Date: 2026-01-31 23:20:57.039344

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b9d890ed31b5"
down_revision: Union[str, None] = "32b62e058cbb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use batch mode for SQLite to modify table
    with op.batch_alter_table("agents", recreate="auto") as batch_op:
        # Drop the foreign key constraint and column
        batch_op.drop_constraint("fk_agents_model_id_models", type_="foreignkey")
        batch_op.drop_column("model_id")
        # Add model_config JSON column
        batch_op.add_column(sa.Column("model_config", sa.JSON(), nullable=True))


def downgrade() -> None:
    # Use batch mode for SQLite to revert changes
    with op.batch_alter_table("agents", recreate="auto") as batch_op:
        # Drop model_config column
        batch_op.drop_column("model_config")
        # Add back model_id column with foreign key
        batch_op.add_column(sa.Column("model_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_agents_model_id_models", "models", ["model_id"], ["id"]
        )
