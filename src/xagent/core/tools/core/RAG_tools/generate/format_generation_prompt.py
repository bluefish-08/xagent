import logging

from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def format_generation_prompt(
    prompt_template: str,
    formatted_contexts: str,
) -> str:
    """Formats a prompt template and contexts into a single string for LLM input.

    This function takes a base prompt template and a string of formatted contexts,
    and combines them into a single, cohesive prompt string suitable for
    sending to a Large Language Model (LLM). It ensures that both the
    prompt template and contexts are provided.

    Args:
        prompt_template: The base template for the prompt, which may include placeholders.
        formatted_contexts: A string containing the relevant contexts, already
                            formatted for LLM input (e.g., from search results).

    Returns:
        A single string representing the full prompt ready for LLM consumption.

    Raises:
        ConfigurationError: If `prompt_template` is empty.

    Examples:
        >>> template = "Answer the question based on the following context: {context}"
        >>> contexts = "Context: The capital of France is Paris."
        >>> full_prompt = format_generation_prompt(template, contexts)
        >>> print(full_prompt)
        Answer the question based on the following context: {context}

        Context:
        The capital of France is Paris.

        Answer:
    """
    if not prompt_template:
        raise ConfigurationError("Prompt template cannot be empty.")
    if not formatted_contexts:
        # NOTE: Depending on the use case, empty contexts might be valid.
        # For RAG, we generally expect contexts.
        logger.warning(
            "Formatted contexts are empty, which might lead to non-grounded generation."
        )

    full_prompt = f"{prompt_template}\n\nContext:\n{formatted_contexts}\n\nAnswer:"
    logger.debug(f"Formatted prompt length: {len(full_prompt)} chars.")

    return full_prompt
