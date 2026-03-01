"""
Agent-aware workspace management for xagent

This module provides workspace management that supports multiple concurrent agents,
ensuring that each agent has its own isolated workspace context.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Agent execution context"""

    id: str
    workspace: Optional["TaskWorkspace"] = None


class TaskWorkspace:
    """
    Task workspace manager that provides isolated working directories for tasks.

    Each task gets its own workspace with:
    - input/: For input files
    - output/: For output files
    - temp/: For temporary files

    The workspace also supports access to external user directories (e.g., knowledge base files)
    through an allowed external directories whitelist.
    """

    def __init__(
        self,
        id: str,
        base_dir: str = "uploads",
        allowed_external_dirs: Optional[List[str]] = None,
    ):
        self.id = id
        self.base_dir = Path(base_dir)

        # Create workspace directory
        self.workspace_dir = self.base_dir / id
        self.input_dir = self.workspace_dir / "input"
        self.output_dir = self.workspace_dir / "output"
        self.temp_dir = self.workspace_dir / "temp"

        # Allowed external directories (e.g., user upload directories with knowledge base files)
        self.allowed_external_dirs: List[Path] = []
        if allowed_external_dirs:
            for dir_path in allowed_external_dirs:
                path = Path(dir_path).resolve()
                if path.exists():
                    self.allowed_external_dirs.append(path)
                else:
                    logger.warning(
                        f"Allowed external directory does not exist: {dir_path}"
                    )

        # Create directory structure
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all workspace directories exist"""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    def get_allowed_dirs(self) -> List[str]:
        """Get list of allowed directories for this workspace"""
        dirs = [
            str(self.workspace_dir),
            str(self.input_dir),
            str(self.output_dir),
            str(self.temp_dir),
        ]
        # Add external allowed directories (e.g., user upload directories)
        dirs.extend([str(d) for d in self.allowed_external_dirs])
        return dirs

    def resolve_path(self, file_path: str, default_dir: str = "output") -> Path:
        """
        Resolve a file path within the workspace or allowed external directories.

        Args:
            file_path: Relative or absolute file path
            default_dir: Default subdirectory if path is relative

        Returns:
            Resolved absolute path within workspace or allowed external directories

        Raises:
            ValueError: If path is outside both workspace and allowed external directories
        """
        path = Path(file_path)

        if path.is_absolute():
            # For absolute paths, verify it's within workspace or allowed external directories
            abs_path = path.resolve()

            # Check if within workspace
            workspace_abs = self.workspace_dir.resolve()
            if abs_path == workspace_abs or abs_path.is_relative_to(workspace_abs):
                return abs_path

            # Check if within any allowed external directory
            for allowed_dir in self.allowed_external_dirs:
                if abs_path.is_relative_to(allowed_dir):
                    logger.debug(
                        f"Accessing external file via allowed directory: {abs_path}"
                    )
                    return abs_path

            # Not in any allowed directory
            allowed_dirs_str = ", ".join(
                [str(self.workspace_dir)] + [str(d) for d in self.allowed_external_dirs]
            )
            raise ValueError(
                f"Path {file_path} is outside allowed directories: {allowed_dirs_str}"
            )
        else:
            # For relative paths, resolve relative to default directory
            if default_dir == "input":
                return (self.input_dir / path).resolve()
            elif default_dir == "output":
                return (self.output_dir / path).resolve()
            elif default_dir == "temp":
                return (self.temp_dir / path).resolve()
            else:
                return (self.workspace_dir / path).resolve()

    def resolve_path_with_search(self, file_path: str) -> Path:
        """
        Resolve a file path within the workspace with intelligent directory search.
        Searches for the file in input -> output -> temp -> workspace root order.
        For absolute paths, checks workspace and allowed external directories.

        Args:
            file_path: Relative or absolute file path

        Returns:
            Resolved absolute path within workspace or allowed external directories

        Raises:
            ValueError: If path is outside both workspace and allowed external directories
            FileNotFoundError: If relative path doesn't exist in any searched directory
        """
        path = Path(file_path)

        if path.is_absolute():
            # For absolute paths, verify it's within workspace or allowed external directories
            abs_path = path.resolve()

            # Check if within workspace
            workspace_abs = self.workspace_dir.resolve()
            if abs_path == workspace_abs or abs_path.is_relative_to(workspace_abs):
                return abs_path

            # Check if within any allowed external directory
            for allowed_dir in self.allowed_external_dirs:
                if abs_path.is_relative_to(allowed_dir):
                    logger.debug(
                        f"Accessing external file via allowed directory: {abs_path}"
                    )
                    return abs_path

            # Not in any allowed directory
            allowed_dirs_str = ", ".join(
                [str(self.workspace_dir)] + [str(d) for d in self.allowed_external_dirs]
            )
            raise ValueError(
                f"Path {file_path} is outside allowed directories: {allowed_dirs_str}"
            )
        else:
            # For relative paths, search in priority order
            # Strip directory prefixes if present to avoid duplicates
            clean_path = path
            if len(path.parts) > 0:
                first_part = path.parts[0].lower()
                if first_part in ["input", "output", "temp"]:
                    # Strip the prefix to avoid duplicate directories
                    clean_path = Path(*path.parts[1:])

            # 1. Try input directory first (most likely for images)
            input_path = (self.input_dir / clean_path).resolve()
            if input_path.exists():
                return input_path

            # 2. Try output directory
            output_path = (self.output_dir / clean_path).resolve()
            if output_path.exists():
                return output_path

            # 3. Try temp directory
            temp_path = (self.temp_dir / clean_path).resolve()
            if temp_path.exists():
                return temp_path

            # 4. If not found, raise error
            raise FileNotFoundError(
                f"File '{file_path}' not found in workspace directories "
                f"(tried: input, output, temp)"
            )

    def get_output_files(self, include_subdirs: bool = True) -> List[Dict[str, Any]]:
        """
        Get all output files in the workspace.

        Args:
            include_subdirs: Whether to include files in subdirectories

        Returns:
            List of file information dictionaries
        """
        output_files = []

        if include_subdirs:
            # Recursively scan output directory
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    output_files.append(self._get_file_info(file_path, "output"))
        else:
            # Only scan top-level of output directory
            for file_path in self.output_dir.iterdir():
                if file_path.is_file():
                    output_files.append(self._get_file_info(file_path, "output"))

        return output_files

    def get_all_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all files in workspace categorized by directory"""
        result: Dict[str, List[Dict[str, Any]]] = {
            "input": [],
            "output": [],
            "temp": [],
            "workspace": [],
        }

        # Scan input directory
        for file_path in self.input_dir.rglob("*"):
            if file_path.is_file():
                result["input"].append(self._get_file_info(file_path, "input"))

        # Scan output directory
        for file_path in self.output_dir.rglob("*"):
            if file_path.is_file():
                result["output"].append(self._get_file_info(file_path, "output"))

        # Scan temp directory
        for file_path in self.temp_dir.rglob("*"):
            if file_path.is_file():
                result["temp"].append(self._get_file_info(file_path, "temp"))

        # Scan workspace root (excluding subdirs)
        for file_path in self.workspace_dir.iterdir():
            if file_path.is_file() and file_path.name not in [
                "input",
                "output",
                "temp",
            ]:
                result["workspace"].append(self._get_file_info(file_path, "workspace"))

        return result

    def _get_file_info(self, file_path: Path, location: str) -> Dict[str, Any]:
        """Get file information for a given path"""
        stat = file_path.stat()

        return {
            "file_path": str(file_path),
            "relative_path": str(file_path.relative_to(self.workspace_dir)),
            "location": location,
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "is_readable": os.access(file_path, os.R_OK),
            "is_writable": os.access(file_path, os.W_OK),
        }

    def clean_temp_files(self) -> None:
        """Clean up temporary files"""
        for file_path in self.temp_dir.rglob("*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                except OSError:
                    pass

    def cleanup(self) -> None:
        """Clean up the entire workspace"""
        if self.workspace_dir.exists():
            logger.info(f"Removing workspace directory: {self.workspace_dir}")
            shutil.rmtree(self.workspace_dir)
            logger.info(f"Workspace directory removed: {self.workspace_dir}")

    def copy_to_workspace(self, source_path: str, target_subdir: str = "input") -> Path:
        """
        Copy a file to the workspace.

        Args:
            source_path: Source file path
            target_subdir: Target subdirectory (input, output, temp)

        Returns:
            Path to the copied file
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if target_subdir == "input":
            target_dir = self.input_dir
        elif target_subdir == "output":
            target_dir = self.output_dir
        elif target_subdir == "temp":
            target_dir = self.temp_dir
        else:
            target_dir = self.workspace_dir

        target_path = target_dir / source.name
        shutil.copy2(source, target_path)
        return target_path

    def __enter__(self) -> "TaskWorkspace":
        """Context manager entry"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit"""
        # Don't automatically cleanup on exit, let the caller decide
        pass


# Simple workspace management functions
def create_workspace(id: str, base_dir: str = "uploads") -> TaskWorkspace:
    """
    Create a new workspace for the given id.

    Args:
        id: Workspace identifier
        base_dir: Base directory for workspaces

    Returns:
        TaskWorkspace instance
    """
    return TaskWorkspace(id, base_dir)


def get_workspace_output_files(
    id: str, base_dir: str = "uploads"
) -> List[Dict[str, Any]]:
    """
    Get output files for a specific workspace.

    Args:
        id: Workspace identifier
        base_dir: Base directory for workspaces

    Returns:
        List of output file information
    """
    workspace = TaskWorkspace(id, base_dir)
    return workspace.get_output_files()


class WorkspaceManager:
    """
    Manager for creating and accessing workspaces.

    Provides a centralized way to manage workspaces with proper cleanup
    and lifecycle management.
    """

    def __init__(self) -> None:
        self._workspaces: Dict[str, TaskWorkspace] = {}

    def get_or_create_workspace(self, base_dir: str, task_id: str) -> TaskWorkspace:
        """
        Get existing workspace or create new one.

        Args:
            base_dir: Base directory for workspaces
            task_id: Task/workspace identifier

        Returns:
            TaskWorkspace instance
        """
        cache_key = f"{base_dir}:{task_id}"

        if cache_key not in self._workspaces:
            workspace = TaskWorkspace(task_id, base_dir)
            self._workspaces[cache_key] = workspace

        return self._workspaces[cache_key]

    def cleanup_workspace(self, base_dir: str, task_id: str) -> None:
        """
        Clean up a specific workspace.

        Args:
            base_dir: Base directory for workspaces
            task_id: Task/workspace identifier
        """
        cache_key = f"{base_dir}:{task_id}"

        if cache_key in self._workspaces:
            workspace = self._workspaces[cache_key]
            workspace.cleanup()
            del self._workspaces[cache_key]

    def cleanup_all_workspaces(self) -> None:
        """Clean up all managed workspaces."""
        for workspace in self._workspaces.values():
            workspace.cleanup()
        self._workspaces.clear()


# Global workspace instance, used in yaml server
_global_workspace: Optional[TaskWorkspace] = None


def init_global_workspace(
    id: str = "default", base_dir: str = "default_workspace"
) -> TaskWorkspace:
    """Initialize the global workspace."""
    global _global_workspace
    if _global_workspace is None:
        _global_workspace = TaskWorkspace(id, base_dir)
    return _global_workspace


def get_global_workspace() -> TaskWorkspace:
    """Get the global workspace instance."""
    global _global_workspace
    if _global_workspace is None:
        raise RuntimeError(
            "Global workspace not initialized. Call init_global_workspace() first."
        )
    return _global_workspace


class MockWorkspace:
    """
    Mock workspace that doesn't create actual directories on disk.

    This is used for scenarios like tool listing where we need a workspace
    object for tool creation but don't want to create directories on disk.

    All paths are virtual and won't be created. File operations will fail if
    attempted, which is fine for read-only operations like tool metadata retrieval.
    """

    def __init__(
        self,
        id: str = "_mock_",
        base_dir: str = "/mock/workspace",
    ):
        """
        Initialize mock workspace.

        Args:
            id: Workspace identifier
            base_dir: Virtual base directory (won't be created)
        """
        self.id = id
        self.base_dir = Path(base_dir)

        # Virtual paths (not created on disk)
        self.workspace_dir = self.base_dir / id
        self.input_dir = self.workspace_dir / "input"
        self.output_dir = self.workspace_dir / "output"
        self.temp_dir = self.workspace_dir / "temp"

        # No external allowed directories for mock
        self.allowed_external_dirs: List[Path] = []

        logger.debug(
            f"Created mock workspace: {self.workspace_dir} (not created on disk)"
        )

    def get_allowed_dirs(self) -> List[str]:
        """Get list of allowed directories for this workspace (virtual paths)."""
        return [
            str(self.workspace_dir),
            str(self.input_dir),
            str(self.output_dir),
            str(self.temp_dir),
        ]

    def resolve_path(self, file_path: str, default_dir: str = "output") -> Path:
        """
        Resolve a file path within the workspace.

        For mock workspace, this returns a virtual path without creating it.

        Args:
            file_path: Relative or absolute file path
            default_dir: Default subdirectory if path is relative

        Returns:
            Resolved absolute path (virtual, not created)
        """
        path = Path(file_path)

        # If absolute path, just return it (for mock workspace)
        if path.is_absolute():
            return path

        # Relative path - resolve to default directory
        if default_dir == "input":
            return self.input_dir / file_path
        elif default_dir == "output":
            return self.output_dir / file_path
        elif default_dir == "temp":
            return self.temp_dir / file_path
        else:
            return self.workspace_dir / file_path

    def __repr__(self) -> str:
        return f"MockWorkspace(id='{self.id}', path='{self.workspace_dir}')"
