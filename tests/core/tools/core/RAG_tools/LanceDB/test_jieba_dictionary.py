"""Tests for the jieba dictionary bootstrap."""

from __future__ import annotations

import builtins
import logging
from pathlib import Path

import jieba
import pytest

from xagent.core.tools.core.RAG_tools.LanceDB.jieba_dictionary import (
    ensure_jieba_dictionary,
)


@pytest.fixture
def model_home(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("LANCE_LANGUAGE_MODEL_HOME", str(tmp_path))
    return tmp_path


def test_ensure_jieba_dictionary_copies_the_package_dictionary(model_home: Path):
    ensure_jieba_dictionary()

    target = model_home / "jieba" / "default" / "dict.txt"
    source = Path(jieba.__file__).parent / "dict.txt"
    assert target.read_bytes() == source.read_bytes()
    # A leftover staging file would be read by lance as another dictionary.
    assert sorted(p.name for p in target.parent.iterdir()) == ["dict.txt"]


def test_ensure_jieba_dictionary_leaves_an_existing_file_alone(model_home: Path):
    target = model_home / "jieba" / "default" / "dict.txt"
    target.parent.mkdir(parents=True)
    target.write_text("existing")
    mtime = target.stat().st_mtime_ns

    ensure_jieba_dictionary()

    assert target.read_text() == "existing"
    assert target.stat().st_mtime_ns == mtime


def test_ensure_jieba_dictionary_warns_instead_of_raising(
    model_home: Path, monkeypatch, caplog
):
    real_import = builtins.__import__

    def no_jieba(name, *args, **kwargs):
        if name == "jieba":
            raise ImportError("no jieba here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_jieba)

    with caplog.at_level(logging.WARNING):
        ensure_jieba_dictionary()

    assert not (model_home / "jieba").exists()
    assert "jieba" in caplog.text and "will fail" in caplog.text
