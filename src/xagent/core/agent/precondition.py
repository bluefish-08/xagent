from typing import Optional

from .context import AgentContext


class PreconditionResolver:
    """
    Checks required fields in AgentContext.state and prompts for missing ones.
    """

    def __init__(self, required_fields: list[str], questions: dict[str, str]):
        self.required_fields = required_fields
        self.questions = questions

    def resolve(self, context: AgentContext) -> Optional[dict]:
        """
        Checks if all required fields are filled. If not, returns a prompt dict.
        """
        for field in self.required_fields:
            if field not in context.state:
                return {
                    "need_user_input": True,
                    "field": field,
                    "question": self.questions.get(field, f"Please provide {field}:"),
                }
        return None
