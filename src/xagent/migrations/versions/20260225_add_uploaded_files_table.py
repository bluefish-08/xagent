from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260225_add_uploaded_files"
down_revision: Union[str, None] = "20250209_add_agent_id_to_tasks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "uploaded_files",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("file_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("storage_path", sa.String(length=2048), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_id"),
        sa.UniqueConstraint("storage_path"),
    )
    op.create_index(
        op.f("ix_uploaded_files_id"), "uploaded_files", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_uploaded_files_file_id"), "uploaded_files", ["file_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_uploaded_files_file_id"), table_name="uploaded_files")
    op.drop_index(op.f("ix_uploaded_files_id"), table_name="uploaded_files")
    op.drop_table("uploaded_files")
