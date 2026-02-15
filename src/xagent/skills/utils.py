"""
Skill utilities - Utility functions for creating skill_manager
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .manager import SkillManager


def create_skill_manager(skills_roots: Optional[List[Path]] = None) -> "SkillManager":
    """
    Create skill_manager (not initialized)

    Args:
        skills_roots: Optional list of skills directories, defaults to built-in paths

    Returns:
        SkillManager instance (not initialized)
    """
    from .manager import SkillManager

    # Default skills directories
    if skills_roots is None:
        builtin_skills_dir = Path(__file__).parent / "builtin"
        user_skills_dir = Path(".xagent/skills")
        user_skills_dir.mkdir(parents=True, exist_ok=True)
        skills_roots = [builtin_skills_dir, user_skills_dir]

    # Create skill_manager (not initialized)
    skill_manager = SkillManager(skills_roots=skills_roots)

    return skill_manager
