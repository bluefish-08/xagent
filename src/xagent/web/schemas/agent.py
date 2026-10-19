"""Shared agent request/response schema pieces."""

from typing import Annotated, List

from pydantic import AfterValidator, BaseModel, Field

PROMPT_CONTEXT_FIELD_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$"
PROMPT_CONTEXT_MAX_FIELDS = 32
PROMPT_CONTEXT_MIN_LENGTH = 1
PROMPT_CONTEXT_MAX_LENGTH = 1024


class PromptContextFieldDeclaration(BaseModel):
    """One field the unsigned widget entry may pass to prompt context."""

    name: str = Field(..., pattern=PROMPT_CONTEXT_FIELD_NAME_PATTERN)
    max_length: int = Field(
        ...,
        ge=PROMPT_CONTEXT_MIN_LENGTH,
        le=PROMPT_CONTEXT_MAX_LENGTH,
    )


def _unique_field_names(
    declarations: List[PromptContextFieldDeclaration],
) -> List[PromptContextFieldDeclaration]:
    names = [declaration.name for declaration in declarations]
    if len(names) != len(set(names)):
        raise ValueError("prompt_context_fields names must be unique")
    return declarations


PromptContextFieldList = Annotated[
    List[PromptContextFieldDeclaration],
    Field(max_length=PROMPT_CONTEXT_MAX_FIELDS),
    AfterValidator(_unique_field_names),
]
