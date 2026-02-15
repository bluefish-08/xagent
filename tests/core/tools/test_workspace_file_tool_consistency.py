"""
Tests for workspace file tool consistency between write and read operations.
"""

import pytest

from xagent.core.tools.adapters.vibe.workspace_file_tool import WorkspaceFileTools
from xagent.core.workspace import TaskWorkspace


class TestWorkspaceFileToolConsistency:
    """Test that write and read operations work consistently."""

    def test_write_then_read_consistency(self, tmp_path):
        """Test that a file written can be immediately read back."""
        # Create workspace
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        # Test content
        test_content = "Hello, workspace!"
        test_filename = "test_file.txt"

        # Write file
        write_result = tools.write_file(test_filename, test_content)
        assert write_result is True

        # Verify file exists in output directory
        output_file = workspace.output_dir / test_filename
        assert output_file.exists()
        assert output_file.read_text() == test_content

        # Read file back
        read_content = tools.read_file(test_filename)
        assert read_content == test_content

    def test_write_then_read_with_relative_path(self, tmp_path):
        """Test that relative paths work consistently."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Relative path test"
        test_filename = "subdir/test_file.txt"

        # Write file with relative path
        write_result = tools.write_file(test_filename, test_content)
        assert write_result is True

        # Verify file exists
        output_file = workspace.output_dir / "subdir" / "test_file.txt"
        assert output_file.exists()
        assert output_file.read_text() == test_content

        # Read file back with same relative path
        read_content = tools.read_file(test_filename)
        assert read_content == test_content

    def test_write_then_read_with_different_default_dirs(self, tmp_path):
        """Test that write and read use consistent default directories."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        test_content = "Default dir test"
        test_filename = "test_default.txt"

        # Write to output directory (default for write_file)
        write_result = tools.write_file(test_filename, test_content)
        assert write_result is True

        # Read from output directory (should be default for read_file too)
        read_content = tools.read_file(test_filename)
        assert read_content == test_content

        # Verify the file is in output directory
        output_file = workspace.output_dir / test_filename
        assert output_file.exists()

    def test_file_not_found_error(self, tmp_path):
        """Test proper error when file doesn't exist."""
        workspace = TaskWorkspace("test_task", str(tmp_path))
        tools = WorkspaceFileTools(workspace)

        with pytest.raises(
            FileNotFoundError,
            match="File 'nonexistent.txt' not found in workspace directories",
        ):
            tools.read_file("nonexistent.txt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
