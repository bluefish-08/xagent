"""
Tests for DAG plan-execute pattern file outputs functionality.
"""

from unittest.mock import Mock

import pytest

from xagent.core.agent.pattern.dag_plan_execute import DAGPlanExecutePattern
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.workspace import TaskWorkspace


class TestDAGFileOutputs:
    """Test DAG plan-execute pattern file outputs functionality."""

    def test_extract_file_outputs_with_workspace(self, tmp_path):
        """Test that file outputs are extracted from workspace."""
        # Create a mock LLM
        mock_llm = Mock(spec=OpenAILLM)

        # Create a workspace with some output files
        workspace_dir = tmp_path / "test_workspace"
        workspace = TaskWorkspace("test_task", str(workspace_dir))

        # Create some test output files
        output_file1 = workspace.output_dir / "result.txt"
        output_file1.write_text("Test result content")

        output_file2 = workspace.output_dir / "data.json"
        output_file2.write_text('{"key": "value"}')

        # Create DAG pattern with workspace
        dag_pattern = DAGPlanExecutePattern(llm=mock_llm, workspace=workspace)

        # Test file extraction
        file_outputs = dag_pattern._extract_file_outputs()

        # Should contain both files as dictionaries
        filenames = [f.get("filename", "") for f in file_outputs]
        assert "result.txt" in filenames
        assert "data.json" in filenames
        file_paths = [f.get("file_path", "") for f in file_outputs]
        assert any(str(workspace.output_dir) in path for path in file_paths)

    def test_extract_file_outputs_without_workspace(self, tmp_path):
        """Test that file outputs fallback to execution results when no workspace."""
        # Create a mock LLM
        mock_llm = Mock(spec=OpenAILLM)

        # Create DAG pattern with workspace (required)
        workspace_dir = tmp_path / "test_workspace"
        workspace = TaskWorkspace(id="test_workspace", base_dir=str(workspace_dir))
        dag_pattern = DAGPlanExecutePattern(llm=mock_llm, workspace=workspace)

        # Test with execution results containing file references

        file_outputs = dag_pattern._extract_file_outputs()

        # Should be empty since no workspace and no fallback to execution results
        assert file_outputs == []

    def test_extract_file_outputs_empty_workspace(self, tmp_path):
        """Test that empty workspace returns no file outputs."""
        # Create a mock LLM
        mock_llm = Mock(spec=OpenAILLM)

        # Create an empty workspace
        workspace_dir = tmp_path / "empty_workspace"
        workspace = TaskWorkspace("empty_task", str(workspace_dir))

        # Create DAG pattern with empty workspace
        dag_pattern = DAGPlanExecutePattern(llm=mock_llm, workspace=workspace)

        # Test file extraction
        file_outputs = dag_pattern._extract_file_outputs()

        # Should be empty
        assert file_outputs == []

    def test_extract_file_outputs_mixed_sources(self, tmp_path):
        """Test that workspace files take precedence over execution results."""
        # Create a mock LLM
        mock_llm = Mock(spec=OpenAILLM)

        # Create a workspace with output files
        workspace_dir = tmp_path / "mixed_workspace"
        workspace = TaskWorkspace("mixed_task", str(workspace_dir))

        # Create a test output file
        workspace_file = workspace.output_dir / "workspace_result.txt"
        workspace_file.write_text("Workspace content")

        # Create DAG pattern with workspace
        dag_pattern = DAGPlanExecutePattern(llm=mock_llm, workspace=workspace)

        # Test with workspace files only (no execution results fallback)
        file_outputs = dag_pattern._extract_file_outputs()

        # Should only contain workspace files
        filenames = [f.get("filename", "") for f in file_outputs]
        assert "workspace_result.txt" in filenames
        file_paths = [f.get("file_path", "") for f in file_outputs]
        assert any(str(workspace.output_dir) in path for path in file_paths)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
