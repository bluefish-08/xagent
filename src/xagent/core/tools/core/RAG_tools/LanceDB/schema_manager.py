from __future__ import annotations

import logging

import pyarrow as pa  # type: ignore
from lancedb.db import DBConnection

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_documents_table",
    "ensure_parses_table",
    "ensure_chunks_table",
    "ensure_embeddings_table",
    "ensure_main_pointers_table",
    "ensure_prompt_templates_table",
    "ensure_ingestion_runs_table",
]


def _table_exists(conn: DBConnection, name: str) -> bool:
    try:
        conn.open_table(name)
        return True
    except Exception:
        return False


def _validate_schema_fields(
    conn: DBConnection, table_name: str, required_fields: list[str]
) -> None:
    """Validate that an existing table contains all required fields.

    Args:
        conn: LanceDB connection
        table_name: Name of the table to validate
        required_fields: List of required field names

    Raises:
        ValueError: If the table exists but is missing required fields.
    """
    if not _table_exists(conn, table_name):
        return

    try:
        table = conn.open_table(table_name)
        existing_schema = table.schema
        existing_field_names = {field.name for field in existing_schema}

        missing_fields = [f for f in required_fields if f not in existing_field_names]

        if missing_fields:
            error_msg = (
                f"Table '{table_name}' exists but is missing required fields: {missing_fields}. "
                f"This is likely due to a schema upgrade. "
                f"Please delete the existing table or manually add the missing fields. "
                f"Note: During development, we do not provide automatic migration scripts. "
                f"To upgrade, you can either:\n"
                f"1. Delete the table (data will be lost): conn.drop_table('{table_name}')\n"
                f"2. Manually add the missing fields using LanceDB's schema update capabilities"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    except ValueError:
        # Re-raise ValueError (our validation error)
        raise
    except Exception as e:
        # Log other errors but don't fail - schema validation is best-effort
        logger.warning(
            f"Could not validate schema for table '{table_name}': {e}. "
            f"Proceeding with table creation/usage."
        )


def _create_table(conn: DBConnection, name: str, schema: object | None = None) -> None:
    if _table_exists(conn, name):
        return
    conn.create_table(name, schema=schema)


def ensure_documents_table(conn: DBConnection) -> None:
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("uploaded_at", pa.timestamp("us")),
            pa.field("title", pa.string()),
            pa.field("language", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, "documents"):
        try:
            table = conn.open_table("documents")
            if "user_id" not in table.schema.names:
                logger.info(
                    "Migrating 'documents' table: adding missing 'user_id' column"
                )
                # Add user_id column with null default, cast to bigint (int64)
                table.add_columns({"user_id": "cast(null as bigint)"})
        except Exception as e:
            logger.warning(f"Failed to check/migrate 'documents' table schema: {e}")

    _create_table(conn, "documents", schema=schema)


def ensure_parses_table(conn: DBConnection) -> None:
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("parser", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("params_json", pa.string()),
            pa.field("parsed_content", pa.large_string()),
            pa.field("user_id", pa.int64()),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, "parses"):
        try:
            table = conn.open_table("parses")
            if "user_id" not in table.schema.names:
                logger.info("Migrating 'parses' table: adding missing 'user_id' column")
                table.add_columns({"user_id": "cast(null as bigint)"})
        except Exception as e:
            logger.warning(f"Failed to check/migrate 'parses' table schema: {e}")

    _create_table(conn, "parses", schema=schema)


def ensure_chunks_table(conn: DBConnection) -> None:
    """Ensure the chunks table exists with proper schema.

    This function creates the table if it doesn't exist, and validates that
    existing tables contain all required fields (especially 'metadata').

    Args:
        conn: LanceDB connection

    Raises:
        ValueError: If the table exists but is missing required fields.
            This typically happens when an old table schema doesn't include
            the 'metadata' field. During development, we do not provide
            automatic migration scripts. Users must either delete the table
            or manually add the missing fields.

    Note:
        There's no upgrade path for existing chunks tables. Any deployment
        with an existing table will hit schema-mismatch errors once the pipeline
        starts writing a column that doesn't exist. If you encounter this error,
        you need to either delete the existing table or manually add the missing
        'metadata' field.
    """
    # Required fields that must exist in the table (especially for schema validation)
    required_fields = ["metadata"]

    # Validate existing table schema before creating/using it
    _validate_schema_fields(conn, "chunks", required_fields)

    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("index", pa.int32()),
            pa.field("text", pa.large_string()),
            pa.field("page_number", pa.int32()),
            pa.field("section", pa.string()),
            pa.field("anchor", pa.string()),
            pa.field("json_path", pa.string()),
            pa.field("chunk_hash", pa.string()),
            pa.field("config_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    _create_table(conn, "chunks", schema=schema)


def ensure_embeddings_table(
    conn: DBConnection, model_tag: str, vector_dim: int | None = None
) -> None:
    """Ensure the embeddings table exists with proper schema.

    This function creates the table if it doesn't exist, and validates that
    existing tables contain all required fields (especially 'metadata').

    Args:
        conn: LanceDB connection
        model_tag: Model tag used to construct the table name (e.g., 'bge_large')
        vector_dim: Optional vector dimension for fixed-size vectors

    Raises:
        ValueError: If the table exists but is missing required fields.
            This typically happens when an old table schema doesn't include
            the 'metadata' field. During development, we do not provide
            automatic migration scripts. Users must either delete the table
            or manually add the missing fields.

    Note:
        There's no upgrade path for existing embeddings tables. Any deployment
        with an existing table will hit schema-mismatch errors once the pipeline
        starts writing a column that doesn't exist. If you encounter this error,
        you need to either delete the existing table or manually add the missing
        'metadata' field.
    """
    table_name = f"embeddings_{model_tag}"

    # Required fields that must exist in the table (especially for schema validation)
    required_fields = ["metadata"]

    # Validate existing table schema before creating/using it
    _validate_schema_fields(conn, table_name, required_fields)

    # Support dynamic vector dimension: if provided, create a FixedSizeList; otherwise allow variable-length
    vector_field_type = (
        pa.list_(pa.float32(), list_size=vector_dim)
        if vector_dim is not None
        else pa.list_(pa.float32())
    )
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("model", pa.string()),
            pa.field("vector", vector_field_type),
            pa.field("vector_dimension", pa.int32()),
            pa.field("text", pa.large_string()),
            pa.field("chunk_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    _create_table(
        conn,
        table_name,
        schema=schema,
    )


def ensure_main_pointers_table(conn: DBConnection) -> None:
    """Ensure the main_pointers table exists with proper schema.

    Args:
        conn: LanceDB connection
    """
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("step_type", pa.string()),
            pa.field("model_tag", pa.string()),
            pa.field("semantic_id", pa.string()),
            pa.field("technical_id", pa.string()),
            pa.field("created_at", pa.timestamp("ms")),
            pa.field("updated_at", pa.timestamp("ms")),
            pa.field("operator", pa.string()),
        ]
    )
    _create_table(conn, "main_pointers", schema=schema)


def ensure_prompt_templates_table(conn: DBConnection) -> None:
    """Ensure the prompt_templates table exists with proper schema.

    Args:
        conn: LanceDB connection
    """
    table_name = "prompt_templates"
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("template", pa.string()),
            pa.field("version", pa.int64()),
            pa.field("is_latest", pa.bool_()),
            pa.field("metadata", pa.string()),  # JSON string, nullable
            pa.field("user_id", pa.int64()),  # Multi-tenancy support
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, table_name):
        try:
            table = conn.open_table(table_name)
            if "user_id" not in table.schema.names:
                logger.info(
                    f"Migrating '{table_name}' table: adding missing 'user_id' column"
                )
                table.add_columns({"user_id": "cast(null as bigint)"})
        except Exception as e:
            logger.warning(f"Failed to check/migrate '{table_name}' table schema: {e}")

    _create_table(conn, table_name, schema=schema)


def ensure_ingestion_runs_table(conn: DBConnection) -> None:
    """Ensure the ingestion_runs table exists with proper schema.

    This table tracks the status of document ingestion processes.

    Args:
        conn: LanceDB connection
    """
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("message", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
            pa.field("user_id", pa.int64()),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, "ingestion_runs"):
        try:
            table = conn.open_table("ingestion_runs")
            if "user_id" not in table.schema.names:
                logger.info(
                    "Migrating 'ingestion_runs' table: adding missing 'user_id' column"
                )
                table.add_columns({"user_id": "cast(null as bigint)"})
        except Exception as e:
            logger.warning(
                f"Failed to check/migrate 'ingestion_runs' table schema: {e}"
            )

    _create_table(conn, "ingestion_runs", schema=schema)
