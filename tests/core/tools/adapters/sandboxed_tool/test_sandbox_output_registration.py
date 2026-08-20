"""Sandbox-produced files must get their file_id from the host process.

The sandbox runner has no database credentials, so a file_id minted in there
names no real record -- which is what made an agent overwrite a generated
.docx with a placeholder to obtain a "real" one.
"""

import asyncio
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.core.tools.adapters.sandboxed_tool.conftest import FakeBaseTool
from xagent.config import SANDBOX_TOOL_RUNNER
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import sandbox_config
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
)
from xagent.core.workspace import SANDBOX_FILE_ID_PREFIX, TaskWorkspace

SANDBOX_MINTED_FILE_ID = "sandbox-only-file-id"


@sandbox_config()
class _FakeGeneratingTool(FakeBaseTool):
    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "fake_generating_tool"


def _make_sandbox(payload: dict) -> MagicMock:
    def _exec(*args, **kwargs):
        result = MagicMock()
        result.exit_code = 0
        result.stdout = json.dumps(payload) if args[0] == "cat" else ""
        result.stderr = ""
        return result

    sandbox = MagicMock()
    sandbox.name = "sandbox-test"
    sandbox.exec = AsyncMock(side_effect=_exec)
    sandbox.write_file = AsyncMock()
    return sandbox


def test_host_process_reregisters_sandbox_generated_files(tmp_path):
    workspace = TaskWorkspace("test_sandbox_outputs", str(tmp_path))
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"PK\x03\x04 not really a docx")

    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["report.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "report.docx",
                        "file_path": str(generated),
                    }
                ],
                "artifacts": [],
            }
        ),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert result["file_refs"], "host registration must not drop the generated file"
    file_ref = result["file_refs"][0]
    assert file_ref["file_id"] != SANDBOX_MINTED_FILE_ID
    assert file_ref["filename"] == "report.docx"
    assert file_ref["size"] == generated.stat().st_size
    assert result["generated_files"] == ["report.docx"]


def test_unreachable_sandbox_paths_are_left_untouched(tmp_path):
    """No host-visible path at all: the early return keeps the sandbox refs."""
    workspace = TaskWorkspace("test_sandbox_guest_paths", str(tmp_path))
    payload = {
        "success": True,
        "generated_files": ["report.docx"],
        "file_refs": [
            {
                "file_id": SANDBOX_MINTED_FILE_ID,
                "filename": "report.docx",
                "file_path": "/guest/only/report.docx",
            }
        ],
        "artifacts": [],
    }
    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(payload),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert result["file_refs"][0]["file_id"] == SANDBOX_MINTED_FILE_ID


def test_register_files_inside_sandbox_runner_never_touches_the_database(
    tmp_path, monkeypatch
):
    workspace = TaskWorkspace("test_sandbox_runner", str(tmp_path))
    target = workspace.output_dir / "note.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello")

    def _fail(*args, **kwargs):
        raise AssertionError("sandbox runner must not reach the metadata store")

    monkeypatch.setattr(TaskWorkspace, "_register_files_locked", _fail)
    monkeypatch.setenv(SANDBOX_TOOL_RUNNER, "1")

    assert (
        workspace.register_file(str(target), file_id="requested-id") == "requested-id"
    )
    assert workspace.register_file(str(target)) == "requested-id"

    monkeypatch.delenv(SANDBOX_TOOL_RUNNER)
    with pytest.raises(AssertionError):
        workspace.register_file(str(target))


def test_partially_visible_refs_keep_the_sandbox_entry(tmp_path):
    workspace = TaskWorkspace("test_sandbox_mixed", str(tmp_path))
    visible = workspace.output_dir / "visible.docx"
    visible.parent.mkdir(parents=True, exist_ok=True)
    visible.write_bytes(b"PK\x03\x04 visible")

    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["visible.docx", "guest.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "visible.docx",
                        "file_path": str(visible),
                    },
                    {
                        "file_id": "guest-only-id",
                        "filename": "guest.docx",
                        "file_path": "/guest/only/guest.docx",
                    },
                ],
                "artifacts": [],
            }
        ),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert [ref["filename"] for ref in result["file_refs"]] == [
        "visible.docx",
        "guest.docx",
    ]
    assert result["file_refs"][0]["file_id"] != SANDBOX_MINTED_FILE_ID
    assert result["file_refs"][1]["file_id"] == "guest-only-id"
    assert result["generated_files"] == ["visible.docx", "guest.docx"]
    # The unregistered ref stays in artifacts on purpose; its id is recognizably
    # not database-backed rather than silently dropped.
    artifact_ids = [artifact.get("file_id") for artifact in result["artifacts"]]
    assert artifact_ids[0] == result["file_refs"][0]["file_id"]
    assert artifact_ids[1] == "guest-only-id"


def test_failed_host_registration_keeps_the_sandbox_metadata(tmp_path, monkeypatch):
    workspace = TaskWorkspace("test_sandbox_failed_host", str(tmp_path))
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"PK\x03\x04 report")

    def _boom(*args, **kwargs):
        raise RuntimeError("metadata store unreachable")

    monkeypatch.setattr(TaskWorkspace, "register_file", _boom)
    monkeypatch.setattr(TaskWorkspace, "get_file_id_from_path", lambda self, path: None)

    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["report.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "report.docx",
                        "file_path": str(generated),
                    }
                ],
                "artifacts": [],
            }
        ),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert result["generated_files"] == ["report.docx"]
    assert result["file_refs"][0]["file_id"] == SANDBOX_MINTED_FILE_ID
    assert result["artifacts"] == []


def test_sandbox_runner_reuses_one_prefixed_id_per_path(tmp_path, monkeypatch):
    workspace = TaskWorkspace("test_sandbox_stable_ids", str(tmp_path))
    target = workspace.output_dir / "note.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello")
    monkeypatch.setenv(SANDBOX_TOOL_RUNNER, "1")

    first = workspace.register_file(str(target))
    assert first.startswith(SANDBOX_FILE_ID_PREFIX)
    assert workspace.register_file(str(target)) == first
    assert workspace.get_file_id_from_path(str(target)) == first


def test_symlinked_guest_spelling_still_merges(tmp_path):
    """A guest mount keeps the unresolved spelling; the merge must still match."""
    real_base = tmp_path / "real"
    real_base.mkdir()
    linked_base = tmp_path / "linked"
    linked_base.symlink_to(real_base, target_is_directory=True)

    workspace = TaskWorkspace("test_sandbox_symlink", str(real_base))
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"PK\x03\x04 report")

    guest_path = str(linked_base / generated.relative_to(real_base.resolve()))
    assert guest_path != str(generated)

    wrapper = SandboxedToolWrapper(
        _FakeGeneratingTool(workspace=workspace),
        _make_sandbox(
            {
                "success": True,
                "generated_files": ["report.docx"],
                "file_refs": [
                    {
                        "file_id": SANDBOX_MINTED_FILE_ID,
                        "filename": "report.docx",
                        "file_path": guest_path,
                    }
                ],
                "artifacts": [],
            }
        ),
    )

    result = asyncio.run(wrapper.run_json_async({}))

    assert result["file_refs"][0]["file_id"] != SANDBOX_MINTED_FILE_ID


def test_regenerated_output_is_re_registered(tmp_path, monkeypatch):
    """A second run over the same path must re-stage, not reuse the old bytes."""
    workspace = TaskWorkspace("test_sandbox_regenerate", str(tmp_path))
    generated = workspace.output_dir / "report.docx"
    generated.parent.mkdir(parents=True, exist_ok=True)

    registered: list[str] = []
    original_register_file = TaskWorkspace.register_file

    def _counting_register_file(self, file_path, *args, **kwargs):
        registered.append(str(file_path))
        return original_register_file(self, file_path, *args, **kwargs)

    monkeypatch.setattr(TaskWorkspace, "register_file", _counting_register_file)

    def _run(payload_bytes: bytes):
        generated.write_bytes(payload_bytes)
        wrapper = SandboxedToolWrapper(
            _FakeGeneratingTool(workspace=workspace),
            _make_sandbox(
                {
                    "success": True,
                    "generated_files": ["report.docx"],
                    "file_refs": [
                        {
                            "file_id": SANDBOX_MINTED_FILE_ID,
                            "filename": "report.docx",
                            "file_path": str(generated),
                        }
                    ],
                    "artifacts": [],
                }
            ),
        )
        return asyncio.run(wrapper.run_json_async({}))

    _run(b"PK\x03\x04 draft")
    second = _run(b"PK\x03\x04 revised and longer")

    assert len(registered) == 2
    assert second["file_refs"][0]["size"] == generated.stat().st_size
