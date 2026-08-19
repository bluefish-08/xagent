"""Skill manager stub for the `build_load_skill_tool` path."""

from typing import Any


class FakeSkillManager:
    def __init__(self, skills: list[dict[str, Any]]) -> None:
        self.skills = {skill["name"]: skill for skill in skills}

    async def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "name": skill["name"],
                "description": skill.get("description", ""),
                "when_to_use": skill.get("when_to_use", ""),
            }
            for skill in self.skills.values()
        ]

    async def get_skill(self, name: str) -> dict[str, Any] | None:
        return self.skills.get(name)
