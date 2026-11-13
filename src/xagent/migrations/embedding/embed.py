"""Precompute vectors for a new embedding model (migration phase 1).

Online and non-destructive: every collection keeps its bound model, so search
and ingestion behave exactly as before. ``embed_document_chunks`` only embeds
chunks the target model lacks, which makes the command resumable and idempotent
and lets the switch phase re-run it to pick up documents added since.

Run it inside the backend environment, with the same ``LANCEDB_DIR`` and model
hub the application uses:

    python -m xagent.migrations.embedding.embed --model <model-id> --dry-run
    python -m xagent.migrations.embedding.embed --model <model-id> --rate-limit 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EmbeddingReadResponse,
    IngestionConfig,
)
from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
    embed_document_chunks,
    is_spreadsheet_source,
)
from xagent.core.tools.core.RAG_tools.storage.vector_backend import KBStorageBackend
from xagent.core.tools.core.RAG_tools.utils.model_resolver import (
    resolve_embedding_adapter,
)
from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    read_chunks_for_embedding,
)
from xagent.migrations.embedding.scope import (
    SKIP_OTHER_ENGINE,
    DocumentScope,
    MigrationCliError,
    add_scope_args,
    collection_backend,
    iter_scope_documents,
    resolve_collection_names,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# A broken provider or credential fails every document alike; stop instead of
# walking the rest of the deployment.
MAX_CONSECUTIVE_DOCUMENT_FAILURES = 5


class RateLimiter:
    """Spaces embedding calls so a run never exceeds ``rate`` texts per second."""

    def __init__(self, rate: float) -> None:
        self._rate = float(rate)
        self._next_start = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, count: int) -> None:
        if self._rate <= 0 or count <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + count / self._rate
        wait = start - now
        if wait > 0:
            time.sleep(wait)


class PacedEmbedding(BaseEmbedding):
    """Spends rate-limit budget for every encode call the stage makes."""

    def __init__(self, inner: BaseEmbedding, limiter: RateLimiter) -> None:
        self._inner = inner
        self._limiter = limiter

    def encode(
        self,
        text: Union[str, List[str]],
        dimension: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        self._limiter.acquire(1 if isinstance(text, str) else len(text))
        return self._inner.encode(text, dimension=dimension, instruct=instruct)

    def get_dimension(self) -> Optional[int]:
        return self._inner.get_dimension()

    @property
    def abilities(self) -> List[str]:
        return self._inner.abilities


@dataclass
class DocumentFailure:
    doc_id: str
    error: str


@dataclass
class CollectionReport:
    name: str
    engine: str = ""
    collection_skip: Optional[str] = None
    documents_total: int = 0
    documents_embedded: int = 0
    documents_unchanged: int = 0
    chunks_embedded: int = 0
    pending_chunks: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    failures: List[DocumentFailure] = field(default_factory=list)
    aborted: bool = False


@dataclass
class EmbedReport:
    model_id: str
    dry_run: bool
    collections: List[CollectionReport] = field(default_factory=list)
    aborted: bool = False

    @property
    def failed(self) -> bool:
        return self.aborted or any(item.failures for item in self.collections)


def embed_collections(
    model_id: str,
    *,
    collections: Optional[Sequence[str]] = None,
    team: Optional[int] = None,
    dry_run: bool = False,
    rate_limit: float = 0.0,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    use_async: bool = False,
) -> EmbedReport:
    """Embed what ``model_id`` is missing for every collection in scope."""
    names = resolve_collection_names(collections=collections, team=team)
    report = EmbedReport(model_id=model_id, dry_run=dry_run)
    if not names:
        # Silence here is indistinguishable from success, and the usual cause
        # is a LANCEDB_DIR that does not match the backend's.
        logger.warning("No collections in scope; check --collection/--team")
        return report

    embedding_config, adapter = resolve_target_model(model_id)
    cfg = IngestionConfig(
        embedding_model_id=embedding_config.id,
        embedding_batch_size=batch_size,
        embedding_use_async=use_async,
    )
    paced = PacedEmbedding(adapter, RateLimiter(rate_limit))
    if rate_limit <= 0:
        logger.warning(
            "No --rate-limit set; the run will use the provider's whole quota"
        )
    logger.info(
        "Target model: %s (dimension %s)",
        embedding_config.id,
        embedding_config.dimension,
    )
    logger.info("Collections in scope (%d): %s", len(names), names)

    for name in names:
        logger.info("Collection %s: starting", name)
        try:
            item = embed_collection(
                name,
                model_id=embedding_config.id,
                embedding_config=embedding_config,
                embedding_adapter=paced,
                cfg=cfg,
                dry_run=dry_run,
            )
        except Exception as exc:  # noqa: BLE001 - one collection must not stop the run
            item = CollectionReport(name=name)
            item.failures.append(DocumentFailure("*", f"{type(exc).__name__}: {exc}"))
            logger.error("Collection %s: failed: %s", name, exc, exc_info=True)
        report.collections.append(item)
        logger.info("Collection %s: %s", name, _describe(item, dry_run=dry_run))
        if item.aborted:
            report.aborted = True
            break
    return report


def resolve_target_model(model_id: str) -> Tuple[EmbeddingModelConfig, BaseEmbedding]:
    # The hub resolves a bare name only among the models the scope may use;
    # the command acts for the deployment, so it resolves as an admin.
    with user_scope_context(None, True):
        return resolve_embedding_adapter(model_id)


def embed_collection(
    name: str,
    *,
    model_id: str,
    embedding_config: EmbeddingModelConfig,
    embedding_adapter: BaseEmbedding,
    cfg: IngestionConfig,
    dry_run: bool,
) -> CollectionReport:
    """Embed what ``model_id`` lacks in one collection; dry-run only counts."""
    report = CollectionReport(name=name)
    report.engine = collection_backend(name)
    if report.engine != KBStorageBackend.LANCEDB.value:
        # Other engines are a later phase; skipping beats failing the whole run.
        report.collection_skip = SKIP_OTHER_ENGINE
        logger.warning(
            "%s: %s engine is not covered by this command; skipped", name, report.engine
        )
        return report
    # Chunks kept from an earlier parse of the same content are out of scope;
    # the switch's acceptance has to catch that difference.
    consecutive_failures = 0

    for doc in iter_scope_documents(name):
        report.documents_total += 1
        if doc.skip_reason or not doc.parse_hash or doc.user_id is None:
            reason = doc.skip_reason or "unknown"
            report.skipped[reason] = report.skipped.get(reason, 0) + 1
            continue
        parse_hash = doc.parse_hash

        if dry_run:
            pending = _pending_chunks(name, doc, parse_hash, model_id)
            report.pending_chunks += pending
            if pending:
                report.documents_embedded += 1
            else:
                report.documents_unchanged += 1
            continue

        try:
            stage = embed_document_chunks(
                collection=name,
                doc_id=doc.doc_id,
                parse_hash=parse_hash,
                embedding_config=embedding_config,
                embedding_adapter=embedding_adapter,
                cfg=cfg,
                user_id=doc.user_id,
                is_admin=False,
                is_spreadsheet=is_spreadsheet_source(doc.source_path or ""),
            )
        except Exception as exc:  # noqa: BLE001 - continue with the next document
            consecutive_failures += 1
            report.failures.append(
                DocumentFailure(doc.doc_id, f"{type(exc).__name__}: {exc}")
            )
            logger.error(
                "%s/%s: embedding failed: %s", name, doc.doc_id, exc, exc_info=True
            )
            if consecutive_failures >= MAX_CONSECUTIVE_DOCUMENT_FAILURES:
                report.aborted = True
                logger.error(
                    "%s: aborting after %d consecutive document failures",
                    name,
                    consecutive_failures,
                )
                return report
            continue
        consecutive_failures = 0

        if stage.pending_count:
            report.documents_embedded += 1
            report.chunks_embedded += stage.embedding_count
            logger.info(
                "%s/%s: embedded %d of %d pending chunks",
                name,
                doc.doc_id,
                stage.embedding_count,
                stage.pending_count,
            )
        else:
            report.documents_unchanged += 1
    return report


def _pending_chunks(
    collection: str, doc: DocumentScope, parse_hash: str, model_id: str
) -> int:
    response = read_chunks_for_embedding(
        collection=collection,
        doc_id=doc.doc_id,
        parse_hash=parse_hash,
        model=model_id,
        user_id=doc.user_id,
        is_admin=False,
    )
    read_model = (
        response
        if isinstance(response, EmbeddingReadResponse)
        else EmbeddingReadResponse.model_validate(response)
    )
    return int(read_model.pending_count)


def _describe(report: CollectionReport, *, dry_run: bool) -> str:
    if report.collection_skip == SKIP_OTHER_ENGINE:
        return f"skipped ({report.engine} engine)"
    verb = "would embed" if dry_run else "embedded"
    parts = [
        f"{report.documents_total} documents",
        f"{verb} {report.documents_embedded}",
        f"unchanged {report.documents_unchanged}",
    ]
    if dry_run:
        parts.append(f"{report.pending_chunks} pending chunks")
    else:
        parts.append(f"{report.chunks_embedded} chunks")
    if report.skipped:
        parts.append(
            "skipped "
            + ", ".join(f"{k}={v}" for k, v in sorted(report.skipped.items()))
        )
    if report.failures:
        parts.append(f"failed {len(report.failures)}")
    return "; ".join(parts)


def _log_report(report: EmbedReport) -> None:
    logger.info("=" * 60)
    logger.info(
        "%s report (model %s)",
        "Dry-run" if report.dry_run else "Migration",
        report.model_id,
    )
    for item in report.collections:
        logger.info("%s: %s", item.name, _describe(item, dry_run=report.dry_run))
        for failure in item.failures:
            logger.error("%s/%s: %s", item.name, failure.doc_id, failure.error)
    if report.aborted:
        logger.error("Run aborted before every collection was processed")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute vectors for a new embedding model (migration phase 1).\n\n"
            "Collections keep their bound model; retrieval is unaffected. Already\n"
            "embedded chunks are skipped, so the command is safe to re-run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Target embedding model id from the model hub",
    )
    add_scope_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be embedded without calling the model or writing",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.0,
        metavar="N",
        help=(
            "Maximum texts embedded per second across the run (one per chunk; "
            "a batch request counts its items); 0 = unlimited"
        ),
    )
    parser.add_argument(
        "--use-async",
        action="store_true",
        help="Encode chunks concurrently; needed by models without batch support",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        metavar="N",
        help=f"Chunks per embedding request (default {DEFAULT_EMBEDDING_BATCH_SIZE})",
    )
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        logger.error("--batch-size must be positive")
        return 2
    if args.rate_limit < 0:
        logger.error("--rate-limit must not be negative")
        return 2

    try:
        report = embed_collections(
            args.model,
            collections=args.collection,
            team=args.team,
            dry_run=args.dry_run,
            rate_limit=args.rate_limit,
            batch_size=args.batch_size,
            use_async=args.use_async,
        )
    except MigrationCliError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        logger.error("Embedding migration failed: %s", exc, exc_info=True)
        return 2

    _log_report(report)
    if not report.collections:
        return 1
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
