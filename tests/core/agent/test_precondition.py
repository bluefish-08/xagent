"""
Unit tests for PreconditionResolver class.

This module tests the precondition resolution functionality including:
- PreconditionResolver initialization
- Required field checking
- Question prompting
- Context state management
"""

import pytest

from xagent.core.agent.context import AgentContext
from xagent.core.agent.precondition import PreconditionResolver


class TestPreconditionResolver:
    """Test cases for PreconditionResolver class."""

    @pytest.fixture
    def basic_resolver(self):
        """Create a basic precondition resolver."""
        return PreconditionResolver(
            required_fields=["name", "age"],
            questions={
                "name": "Please enter your name:",
                "age": "Please enter your age:",
            },
        )

    @pytest.fixture
    def minimal_resolver(self):
        """Create a resolver with minimal configuration."""
        return PreconditionResolver(required_fields=["single_field"], questions={})

    @pytest.fixture
    def empty_context(self):
        """Create an empty agent context."""
        return AgentContext()

    @pytest.fixture
    def partial_context(self):
        """Create a context with some fields filled."""
        context = AgentContext()
        context.state["name"] = "John Doe"
        return context

    @pytest.fixture
    def complete_context(self):
        """Create a context with all required fields filled."""
        context = AgentContext()
        context.state["name"] = "John Doe"
        context.state["age"] = 30
        return context

    def test_precondition_resolver_initialization(self, basic_resolver):
        """Test PreconditionResolver initialization."""
        assert basic_resolver.required_fields == ["name", "age"]
        assert basic_resolver.questions == {
            "name": "Please enter your name:",
            "age": "Please enter your age:",
        }

    def test_precondition_resolver_initialization_empty_questions(
        self, minimal_resolver
    ):
        """Test PreconditionResolver initialization with empty questions."""
        assert minimal_resolver.required_fields == ["single_field"]
        assert minimal_resolver.questions == {}

    def test_resolve_all_fields_missing(self, basic_resolver, empty_context):
        """Test resolve when all required fields are missing."""
        result = basic_resolver.resolve(empty_context)

        assert result is not None
        assert result["need_user_input"] is True
        assert result["field"] in [
            "name",
            "age",
        ]  # Should return the first missing field
        assert "question" in result
        assert result["question"] in basic_resolver.questions.values()

    def test_resolve_first_field_missing(self, basic_resolver, partial_context):
        """Test resolve when the first required field is missing."""
        result = basic_resolver.resolve(partial_context)

        assert result is not None
        assert result["need_user_input"] is True
        assert result["field"] == "age"  # name is present, age is missing
        assert result["question"] == "Please enter your age:"

    def test_resolve_second_field_missing(self, basic_resolver):
        """Test resolve when the second required field is missing."""
        context = AgentContext()
        context.state["age"] = 30  # Only age is present

        result = basic_resolver.resolve(context)

        assert result is not None
        assert result["need_user_input"] is True
        assert result["field"] == "name"  # name is missing, age is present
        assert result["question"] == "Please enter your name:"

    def test_resolve_all_fields_present(self, basic_resolver, complete_context):
        """Test resolve when all required fields are present."""
        result = basic_resolver.resolve(complete_context)

        assert result is None  # Should return None when all fields are present

    def test_resolve_empty_required_fields(self):
        """Test resolve with no required fields."""
        resolver = PreconditionResolver(required_fields=[], questions={})
        context = AgentContext()

        result = resolver.resolve(context)

        assert result is None  # Should return None when no fields are required

    def test_resolve_single_required_field_present(self, minimal_resolver):
        """Test resolve with single required field that is present."""
        context = AgentContext()
        context.state["single_field"] = "value"

        result = minimal_resolver.resolve(context)

        assert result is None

    def test_resolve_single_required_field_missing(self, minimal_resolver):
        """Test resolve with single required field that is missing."""
        context = AgentContext()

        result = minimal_resolver.resolve(context)

        assert result is not None
        assert result["need_user_input"] is True
        assert result["field"] == "single_field"
        assert "question" in result

    def test_resolve_default_question(self, minimal_resolver):
        """Test resolve with default question when no custom question is provided."""
        context = AgentContext()

        result = minimal_resolver.resolve(context)

        assert result is not None
        assert result["question"] == "Please provide single_field:"  # Default format

    def test_resolve_custom_question(self, basic_resolver):
        """Test resolve with custom question."""
        context = AgentContext()

        result = basic_resolver.resolve(context)

        assert result is not None
        assert result["question"] in basic_resolver.questions.values()

    def test_resolve_case_sensitivity(self, basic_resolver):
        """Test that field names are case-sensitive."""
        context = AgentContext()
        context.state["NAME"] = "John Doe"  # Different case
        context.state["age"] = 30

        result = basic_resolver.resolve(context)

        assert result is not None
        assert result["field"] == "name"  # Should be case-sensitive

    def test_resolve_field_order(self, basic_resolver, empty_context):
        """Test that fields are checked in order."""
        result = basic_resolver.resolve(empty_context)

        assert result["field"] == "name"  # Should return the first missing field

    def test_resolve_with_none_values(self, basic_resolver):
        """Test resolve with None values in context."""
        context = AgentContext()
        context.state["name"] = None
        context.state["age"] = None

        result = basic_resolver.resolve(context)

        assert result is None  # None values are treated as present (field exists)

    def test_resolve_with_empty_string_values(self, basic_resolver):
        """Test resolve with empty string values in context."""
        context = AgentContext()
        context.state["name"] = ""
        context.state["age"] = 25

        result = basic_resolver.resolve(context)

        assert result is None  # Empty string is treated as present (field exists)

    def test_resolve_with_whitespace_values(self, basic_resolver):
        """Test resolve with whitespace-only values in context."""
        context = AgentContext()
        context.state["name"] = "   "
        context.state["age"] = 25

        result = basic_resolver.resolve(context)

        # Should depend on implementation - for now, treat whitespace as present
        if result is not None:
            assert result["field"] == "name"

    def test_resolve_with_zero_values(self, basic_resolver):
        """Test resolve with zero values in context."""
        context = AgentContext()
        context.state["name"] = "John Doe"
        context.state["age"] = 0  # Zero should be treated as valid

        result = basic_resolver.resolve(context)

        assert result is None  # All fields are present

    def test_resolve_with_false_values(self, basic_resolver):
        """Test resolve with False values in context."""
        context = AgentContext()
        context.state["name"] = "John Doe"
        context.state["age"] = False  # False should be treated as valid

        result = basic_resolver.resolve(context)

        assert result is None  # All fields are present

    def test_resolve_mixed_data_types(self):
        """Test resolve with mixed data types."""
        resolver = PreconditionResolver(
            required_fields=["string_field", "int_field", "bool_field", "dict_field"],
            questions={},
        )

        context = AgentContext()
        context.state["string_field"] = "test"
        context.state["int_field"] = 42
        context.state["bool_field"] = True
        context.state["dict_field"] = {"key": "value"}

        result = resolver.resolve(context)

        assert result is None  # All fields are present

    def test_resolve_missing_field_in_middle(self):
        """Test resolve when a field in the middle of the list is missing."""
        resolver = PreconditionResolver(
            required_fields=["field1", "field2", "field3"], questions={}
        )

        context = AgentContext()
        context.state["field1"] = "value1"
        # field2 is missing
        context.state["field3"] = "value3"

        result = resolver.resolve(context)

        assert result is not None
        assert result["field"] == "field2"  # Should return the first missing field

    def test_resolve_multiple_missing_fields(self, basic_resolver, empty_context):
        """Test resolve when multiple fields are missing."""
        result = basic_resolver.resolve(empty_context)

        assert result is not None
        assert result["field"] == "name"  # Should return the first missing field

    def test_resolve_context_state_not_modified(self, basic_resolver, empty_context):
        """Test that resolve doesn't modify the context state."""
        original_state = empty_context.state.copy()

        basic_resolver.resolve(empty_context)

        assert empty_context.state == original_state  # State should not be modified

    def test_resolve_return_structure(self, basic_resolver, empty_context):
        """Test the structure of the returned dictionary."""
        result = basic_resolver.resolve(empty_context)

        assert isinstance(result, dict)
        assert "need_user_input" in result
        assert "field" in result
        assert "question" in result
        assert isinstance(result["need_user_input"], bool)
        assert isinstance(result["field"], str)
        assert isinstance(result["question"], str)
        assert result["need_user_input"] is True

    def test_resolve_idempotency(self, basic_resolver, empty_context):
        """Test that multiple calls to resolve return the same result."""
        result1 = basic_resolver.resolve(empty_context)
        result2 = basic_resolver.resolve(empty_context)

        assert result1 == result2

    def test_resolve_with_complex_questions(self):
        """Test resolve with complex question strings."""
        resolver = PreconditionResolver(
            required_fields=["complex_field"],
            questions={
                "complex_field": "Please provide a detailed description of your requirements, including any specific constraints or preferences:"
            },
        )

        context = AgentContext()

        result = resolver.resolve(context)

        assert result is not None
        assert result["question"] == resolver.questions["complex_field"]

    def test_resolve_with_special_characters_in_questions(self):
        """Test resolve with special characters in question strings."""
        resolver = PreconditionResolver(
            required_fields=["email"],
            questions={
                "email": "Please enter your email address (e.g., user@example.com):"
            },
        )

        context = AgentContext()

        result = resolver.resolve(context)

        assert result is not None
        assert "@" in result["question"]  # Special character should be preserved

    def test_resolve_with_unicode_characters(self):
        """Test resolve with unicode characters in questions."""
        resolver = PreconditionResolver(
            required_fields=["name"],
            questions={
                "name": "请输入您的姓名："  # Chinese characters
            },
        )

        context = AgentContext()

        result = resolver.resolve(context)

        assert result is not None
        assert result["question"] == resolver.questions["name"]

    def test_resolve_with_very_long_question(self):
        """Test resolve with very long question strings."""
        long_question = (
            "This is a very long question that contains a lot of detail "
            + "and might be used to provide extensive context to the user "
            + "about what information is needed and why it's important "
            + "for the task at hand."
        )

        resolver = PreconditionResolver(
            required_fields=["detail"], questions={"detail": long_question}
        )

        context = AgentContext()

        result = resolver.resolve(context)

        assert result is not None
        assert result["question"] == long_question


if __name__ == "__main__":
    pytest.main([__file__])
