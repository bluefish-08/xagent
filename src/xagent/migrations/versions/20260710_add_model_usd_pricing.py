"""add optional USD pricing fields to models

Revision ID: 20260710_add_model_usd_pricing
Revises: 1c2ae61b5a6d
Create Date: 2026-07-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20260710_add_model_usd_pricing"
down_revision: Union[str, None] = "1c2ae61b5a6d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = ("input_usd_per_1m", "output_usd_per_1m")


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if "models" not in inspector.get_table_names():
        return

    existing = [col["name"] for col in inspector.get_columns("models")]
    for name in _COLUMNS:
        if name not in existing:
            op.add_column("models", sa.Column(name, sa.Float(), nullable=True))


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if "models" not in inspector.get_table_names():
        return

    existing = [col["name"] for col in inspector.get_columns("models")]
    for name in _COLUMNS:
        if name in existing:
            op.drop_column("models", name)
