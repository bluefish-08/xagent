"""LanceDB migration: Backfill user_id for chunks and embeddings tables.

This migration script backfills the user_id field in chunks and embeddings tables
by joining with the documents table. This is necessary for multi-tenancy data isolation.

Usage:
    python -m xagent.migrations.lancedb.backfill_user_id --dry-run  # Preview changes
    python -m xagent.migrations.lancedb.backfill_user_id           # Apply migration
"""

import argparse
import logging
import os
import sys

# Add parent directories to path for imports
# This must be done before importing project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import after path modification (required for standalone migration scripts)
# ruff: noqa: E402
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_chunks_table,
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
from xagent.core.tools.core.RAG_tools.utils.string_utils import (
    build_safe_collection_filter,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def backfill_chunks_table(dry_run: bool = False) -> dict:
    """Backfill user_id for chunks table by joining with documents table.

    Args:
        dry_run: If True, only report what would be changed without making changes

    Returns:
        Dictionary with migration statistics
    """
    conn = get_connection_from_env()
    ensure_chunks_table(conn)
    ensure_documents_table(conn)

    chunks_table = conn.open_table("chunks")
    docs_table = conn.open_table("documents")

    logger.info("Starting chunks table user_id backfill...")

    # Get all chunks with NULL user_id
    null_chunks = query_to_list(chunks_table.search().where("user_id IS NULL"))
    logger.info(f"Found {len(null_chunks)} chunks with NULL user_id")

    if not null_chunks:
        return {"table": "chunks", "total": 0, "backfilled": 0, "skipped": 0}

    # Build a mapping of doc_id -> user_id from documents table
    doc_user_map = {}
    all_doc_ids = set(
        chunk.get("doc_id") for chunk in null_chunks if chunk.get("doc_id")
    )

    logger.info(f"Looking up user_id for {len(all_doc_ids)} unique documents...")

    for doc_id in all_doc_ids:
        # Query documents table for this doc_id
        doc_filter = build_safe_collection_filter("*", f"doc_id = '{doc_id}'")
        docs = query_to_list(docs_table.search().where(doc_filter).limit(1))

        if docs and docs[0].get("user_id") is not None:
            doc_user_map[doc_id] = docs[0].get("user_id")

    logger.info(f"Found user_id mappings for {len(doc_user_map)} documents")

    # Prepare updates
    updates = []
    skipped = 0

    for chunk in null_chunks:
        doc_id = chunk.get("doc_id")
        chunk_id = chunk.get("chunk_id")
        parse_hash = chunk.get("parse_hash")

        if doc_id in doc_user_map:
            user_id = doc_user_map[doc_id]
            updates.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "parse_hash": parse_hash,
                    "user_id": user_id,
                }
            )
        else:
            skipped += 1

    logger.info(
        f"Prepared {len(updates)} chunk updates, {skipped} skipped (no matching document)"
    )

    if dry_run:
        logger.info("[DRY RUN] Would backfill %d chunks", len(updates))
        return {
            "table": "chunks",
            "total": len(null_chunks),
            "backfilled": 0,
            "skipped": skipped,
            "dry_run": True,
        }

    # Execute updates
    # LanceDB doesn't support UPDATE with JOIN, so we update individually
    backfilled_count = 0
    failed_count = 0

    for update in updates:
        try:
            # Build filter to identify the specific chunk
            filter_parts = [
                f"doc_id = '{update['doc_id']}'",
                f"chunk_id = '{update['chunk_id']}'",
                f"parse_hash = '{update['parse_hash']}'",
            ]
            filter_str = " and ".join(filter_parts)

            # Update only if user_id is currently NULL
            chunks_table.update(filter_str, {"user_id": update["user_id"]})
            backfilled_count += 1

        except Exception as e:
            failed_count += 1
            logger.warning(f"Failed to update chunk {update['chunk_id']}: {e}")

    logger.info(
        f"Chunks backfill complete: {backfilled_count} updated, {failed_count} failed"
    )

    return {
        "table": "chunks",
        "total": len(null_chunks),
        "backfilled": backfilled_count,
        "skipped": skipped,
    }


def backfill_embeddings_table(dry_run: bool = False) -> dict:
    """Backfill user_id for embeddings tables by joining with documents table.

    Args:
        dry_run: If True, only report what would be changed without making changes

    Returns:
        Dictionary with migration statistics aggregated across all embeddings tables
    """
    conn = get_connection_from_env()
    ensure_documents_table(conn)

    # Get all embeddings tables
    table_names_fn = getattr(conn, "table_names", None)
    if table_names_fn is None:
        logger.warning("LanceDB connection missing table_names() method")
        return {
            "table": "embeddings",
            "total": 0,
            "backfilled": 0,
            "skipped": 0,
            "details": [],
        }

    try:
        table_names = list(table_names_fn())
    except Exception as e:
        logger.warning(f"Failed to list LanceDB tables: {e}")
        return {
            "table": "embeddings",
            "total": 0,
            "backfilled": 0,
            "skipped": 0,
            "details": [],
        }

    embeddings_tables = [t for t in table_names if t.startswith("embeddings_")]

    logger.info(f"Found {len(embeddings_tables)} embeddings tables")

    if not embeddings_tables:
        return {
            "table": "embeddings",
            "total": 0,
            "backfilled": 0,
            "skipped": 0,
            "details": [],
        }

    docs_table = conn.open_table("documents")

    # Build doc_id -> user_id mapping (reuse across all embeddings tables)
    all_unique_doc_ids = set()

    for table_name in embeddings_tables:
        embeddings_table = conn.open_table(table_name)
        null_embeddings = query_to_list(
            embeddings_table.search().where("user_id IS NULL").limit(10000)
        )

        for emb in null_embeddings:
            if emb.get("doc_id"):
                all_unique_doc_ids.add(emb.get("doc_id"))

    logger.info(
        f"Found {len(all_unique_doc_ids)} unique documents needing user_id lookup"
    )

    # Build doc_id -> user_id mapping once
    doc_user_map = {}
    for doc_id in all_unique_doc_ids:
        doc_filter = build_safe_collection_filter("*", f"doc_id = '{doc_id}'")
        docs = query_to_list(docs_table.search().where(doc_filter).limit(1))

        if docs and docs[0].get("user_id") is not None:
            doc_user_map[doc_id] = docs[0].get("user_id")

    logger.info(f"Built user_id mapping for {len(doc_user_map)} documents")

    # Process each embeddings table
    all_results = []

    for table_name in embeddings_tables:
        logger.info(f"Processing {table_name}...")
        embeddings_table = conn.open_table(table_name)

        # Find embeddings with NULL user_id
        null_embeddings = query_to_list(
            embeddings_table.search().where("user_id IS NULL")
        )

        if not null_embeddings:
            logger.info(f"  No NULL user_id records found in {table_name}")
            all_results.append(
                {"table": table_name, "total": 0, "backfilled": 0, "skipped": 0}
            )
            continue

        logger.info(f"  Found {len(null_embeddings)} records with NULL user_id")

        # Prepare updates
        updates = []
        skipped = 0

        for emb in null_embeddings:
            doc_id = emb.get("doc_id")
            chunk_id = emb.get("chunk_id")
            parse_hash = emb.get("parse_hash")
            model = emb.get("model")

            if doc_id in doc_user_map:
                updates.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "parse_hash": parse_hash,
                        "model": model,
                        "user_id": doc_user_map[doc_id],
                    }
                )
            else:
                skipped += 1

        logger.info(f"  Prepared {len(updates)} updates, {skipped} skipped")

        if dry_run:
            logger.info(
                f"[DRY RUN] {table_name}: Would backfill {len(updates)} records"
            )
            all_results.append(
                {
                    "table": table_name,
                    "total": len(null_embeddings),
                    "backfilled": 0,
                    "skipped": skipped,
                    "dry_run": True,
                }
            )
            continue

        # Execute updates
        backfilled_count = 0
        failed_count = 0

        for update in updates:
            try:
                filter_parts = [
                    f"doc_id = '{update['doc_id']}'",
                    f"chunk_id = '{update['chunk_id']}'",
                    f"parse_hash = '{update['parse_hash']}'",
                    f"model = '{update['model']}'",
                ]
                filter_str = " and ".join(filter_parts)

                embeddings_table.update(filter_str, {"user_id": update["user_id"]})
                backfilled_count += 1

            except Exception as e:
                failed_count += 1
                logger.warning(f"  Failed to update embedding: {e}")

        logger.info(f"  Backfilled {backfilled_count} records, {failed_count} failed")
        all_results.append(
            {
                "table": table_name,
                "total": len(null_embeddings),
                "backfilled": backfilled_count,
                "skipped": skipped,
            }
        )

    # Aggregate results
    total = sum(r["total"] for r in all_results)
    backfilled = sum(r["backfilled"] for r in all_results)
    skipped = sum(r["skipped"] for r in all_results)

    return {
        "table": "embeddings",
        "total": total,
        "backfilled": backfilled,
        "skipped": skipped,
        "details": all_results,
    }


def backfill_all(dry_run: bool = False) -> dict:
    """Run backfill for both chunks and embeddings tables.

    Args:
        dry_run: If True, only report what would be changed without making changes

    Returns:
        Dictionary with aggregated migration statistics
    """
    logger.info("=" * 60)
    logger.info("LanceDB User ID Backfill Migration")
    logger.info("=" * 60)

    if dry_run:
        logger.info("Running in DRY RUN mode - no changes will be made")
    else:
        logger.warning("Running in LIVE mode - changes will be made to the database")

    logger.info("")

    # Backfill chunks table
    logger.info("Step 1: Backfilling chunks table...")
    chunks_result = backfill_chunks_table(dry_run=dry_run)
    logger.info("")

    # Backfill embeddings tables
    logger.info("Step 2: Backfilling embeddings tables...")
    embeddings_result = backfill_embeddings_table(dry_run=dry_run)
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info("Chunks table:")
    logger.info(f"  Total NULL records: {chunks_result['total']}")
    logger.info(f"  Backfilled: {chunks_result['backfilled']}")
    logger.info(f"  Skipped: {chunks_result['skipped']}")
    logger.info("")
    logger.info("Embeddings tables:")
    logger.info(f"  Total NULL records: {embeddings_result['total']}")
    logger.info(f"  Backfilled: {embeddings_result['backfilled']}")
    logger.info(f"  Skipped: {embeddings_result['skipped']}")

    if embeddings_result.get("details"):
        logger.info("")
        logger.info("Per-table details:")
        for detail in embeddings_result["details"]:
            logger.info(f"  {detail['table']}:")
            logger.info(
                f"    Total: {detail['total']}, "
                f"Backfilled: {detail['backfilled']}, "
                f"Skipped: {detail['skipped']}"
            )

    logger.info("=" * 60)

    return {
        "chunks": chunks_result,
        "embeddings": embeddings_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill user_id for LanceDB chunks and embeddings tables"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode without making changes",
    )
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Only backfill chunks table",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Only backfill embeddings tables",
    )

    args = parser.parse_args()

    try:
        if args.chunks_only:
            result = backfill_chunks_table(dry_run=args.dry_run)
        elif args.embeddings_only:
            result = backfill_embeddings_table(dry_run=args.dry_run)
        else:
            result = backfill_all(dry_run=args.dry_run)

        # Exit with appropriate code
        total_skipped = result.get("chunks", {}).get("skipped", 0) + result.get(
            "embeddings", {}
        ).get("skipped", 0)
        if total_skipped > 0:
            logger.warning(f"Migration completed with {total_skipped} skipped records")
            sys.exit(1)  # Warning exit code

        sys.exit(0)  # Success

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(2)  # Error exit code
