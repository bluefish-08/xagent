"""rename model_config to models in agents

Revision ID: f79da474c69d
Revises: b9d890ed31b5
Create Date: 2026-01-31 23:22:03.212451

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f79da474c69d"
down_revision: Union[str, None] = "b9d890ed31b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use batch mode for SQLite to rename column
    with op.batch_alter_table("agents", recreate="auto") as batch_op:
        batch_op.alter_column("model_config", new_column_name="models")


def downgrade() -> None:
    # Use batch mode for SQLite to revert column name
    with op.batch_alter_table("agents", recreate="auto") as batch_op:
        batch_op.alter_column("models", new_column_name="model_config")
