"""Place the jieba dictionary lance's FTS tokenizer loads at runtime."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_HOME = Path.home() / ".local" / "share" / "lance" / "language_models"
_once = threading.Lock()
_done = False


def _language_model_home() -> Path:
    configured = os.environ.get("LANCE_LANGUAGE_MODEL_HOME")
    return Path(configured) if configured else _DEFAULT_MODEL_HOME


def ensure_jieba_dictionary() -> None:
    """Copy the PyPI jieba dictionary into lance's language-model home.

    Only the macOS wheel embeds a dictionary; on Linux both index creation and
    querying raise without this file, so it is placed on every connection path
    rather than only where the index is built.
    """
    target = _language_model_home() / "jieba" / "default" / "dict.txt"
    if target.exists():
        return

    try:
        import jieba

        source = Path(jieba.__file__).parent / "dict.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent workers race on the same path; os.replace is atomic, a
        # partially copied dict.txt would not be.
        staged = target.with_name(f"dict.txt.{os.getpid()}.tmp")
        shutil.copyfile(source, staged)
        os.replace(staged, target)
        logger.info("Installed the jieba dictionary for LanceDB FTS at %s", target)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not install the jieba dictionary at %s (%s); LanceDB FTS index "
            "creation and full-text queries will fail until it exists",
            target,
            exc,
        )


def ensure_jieba_dictionary_once() -> None:
    """Run :func:`ensure_jieba_dictionary` at most once per process."""
    global _done
    if _done:
        return
    with _once:
        if _done:
            return
        ensure_jieba_dictionary()
        _done = True
