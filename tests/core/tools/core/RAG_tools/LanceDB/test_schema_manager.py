from __future__ import annotations

from pathlib import Path

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_chunks_table,
    ensure_documents_table,
    ensure_embeddings_table,
    ensure_parses_table,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env


def test_ensure_tables(tmp_path: Path, monkeypatch) -> None:
    db_dir = tmp_path / "db"
    monkeypatch.setenv("LANCEDB_DIR", str(db_dir))
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    ensure_parses_table(conn)
    ensure_chunks_table(conn)
    ensure_embeddings_table(conn, to_model_tag("BAAI/bge-large-zh-v1.5"))

    # open_table should not raise
    for name in [
        "documents",
        "parses",
        "chunks",
        "embeddings_BAAI_bge_large_zh_v1_5",
    ]:
        conn.open_table(name)
