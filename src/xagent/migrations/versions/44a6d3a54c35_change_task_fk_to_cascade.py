"""change_task_fk_to_cascade

Revision ID: 44a6d3a54c35
Revises: a0f42ff986b2
Create Date: 2026-03-11 00:47:06.197244

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "44a6d3a54c35"
down_revision: Union[str, None] = "a0f42ff986b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite doesn't support ALTER CONSTRAINT directly
    # Need to recreate the table with the new foreign key
    op.execute("PRAGMA foreign_keys=off")

    # Create new table with CASCADE constraint
    op.execute("""
        CREATE TABLE uploaded_files_new (
            id INTEGER PRIMARY KEY,
            file_id VARCHAR(36) UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            task_id INTEGER,
            filename VARCHAR(512) NOT NULL,
            storage_path VARCHAR(2048) NOT NULL UNIQUE,
            mime_type VARCHAR(255),
            file_size INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME,
            updated_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
        )
    """)

    # Copy data from old table to new table
    op.execute("""
        INSERT INTO uploaded_files_new
        SELECT * FROM uploaded_files
    """)

    # Drop old table
    op.execute("DROP TABLE uploaded_files")

    # Rename new table
    op.execute("ALTER TABLE uploaded_files_new RENAME TO uploaded_files")

    # Recreate indexes
    op.execute("CREATE INDEX ix_uploaded_files_id ON uploaded_files (id)")
    op.execute("CREATE INDEX ix_uploaded_files_file_id ON uploaded_files (file_id)")

    op.execute("PRAGMA foreign_keys=on")


def downgrade() -> None:
    # Revert back to SET NULL
    op.execute("PRAGMA foreign_keys=off")

    # Create table with SET NULL constraint
    op.execute("""
        CREATE TABLE uploaded_files_new (
            id INTEGER PRIMARY KEY,
            file_id VARCHAR(36) UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            task_id INTEGER,
            filename VARCHAR(512) NOT NULL,
            storage_path VARCHAR(2048) NOT NULL UNIQUE,
            mime_type VARCHAR(255),
            file_size INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME,
            updated_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE SET NULL
        )
    """)

    # Copy data
    op.execute("""
        INSERT INTO uploaded_files_new
        SELECT * FROM uploaded_files
    """)

    # Drop and rename
    op.execute("DROP TABLE uploaded_files")
    op.execute("ALTER TABLE uploaded_files_new RENAME TO uploaded_files")

    # Recreate indexes
    op.execute("CREATE INDEX ix_uploaded_files_id ON uploaded_files (id)")
    op.execute("CREATE INDEX ix_uploaded_files_file_id ON uploaded_files (file_id)")

    op.execute("PRAGMA foreign_keys=on")
