"""
Unit tests for conditional branching support in DAG plan-execute pattern.
"""

import pytest

from xagent.core.agent.pattern.dag_plan_execute.models import (
    ExecutionPlan,
    PlanStep,
    StepStatus,
    extract_branch_key_from_final_answer,
)


class TestPlanStepConditional:
    """Test PlanStep conditional node support"""

    def test_is_conditional_true(self):
        """Test is_conditional property returns True when branches defined"""
        step = PlanStep(
            id="test_step",
            name="Test Step",
            description="A test conditional step",
            conditional_branches={"human": "step_1", "kb": "step_2"},
        )
        assert step.is_conditional is True

    def test_is_conditional_false(self):
        """Test is_conditional property returns False when no branches"""
        step = PlanStep(
            id="test_step", name="Test Step", description="A test regular step"
        )
        assert step.is_conditional is False

    def test_can_execute_with_required_branch(self):
        """Test can_execute respects required_branch"""
        step = PlanStep(
            id="test_step",
            name="Test Step",
            description="A test step",
            dependencies=["parent_conditional"],
            required_branch="human",
        )

        # When parent selects "human" branch, step can execute
        assert (
            step.can_execute(
                completed_steps={"parent_conditional"},
                skipped_steps=set(),
                active_branches={"parent_conditional": "human"},
            )
            is True
        )

        # When parent selects "kb" branch, step cannot execute
        assert (
            step.can_execute(
                completed_steps={"parent_conditional"},
                skipped_steps=set(),
                active_branches={"parent_conditional": "kb"},
            )
            is False
        )


class TestExtractBranchKey:
    """Test branch key extraction from LLM output"""

    def test_extract_branch_exact_match(self):
        """Test extraction with exact branch key"""
        result = extract_branch_key_from_final_answer(
            "Based on the user request, I select: human", ["human", "kb"]
        )
        assert result == "human"

    def test_extract_branch_with_marker(self):
        """Test extraction with [BRANCH: key] marker"""
        result = extract_branch_key_from_final_answer(
            "Analysis complete. [BRANCH: kb] This is the best choice.", ["human", "kb"]
        )
        assert result == "kb"

    def test_extract_branch_start(self):
        """Test extraction when branch key at start"""
        result = extract_branch_key_from_final_answer(
            "kb: The user wants knowledge base search", ["human", "kb"]
        )
        assert result == "kb"

    def test_extract_branch_fuzzy(self):
        """Test fuzzy matching for branch key"""
        result = extract_branch_key_from_final_answer(
            "After analysis, I think we should go with the kb approach", ["human", "kb"]
        )
        assert result == "kb"

    def test_extract_branch_not_found(self):
        """Test when branch key cannot be found"""
        result = extract_branch_key_from_final_answer(
            "I cannot determine the answer", ["human", "kb"]
        )
        assert result is None

    def test_extract_branch_chinese(self):
        """Test extraction with Chinese text"""
        result = extract_branch_key_from_final_answer(
            "根据分析，选择 human 分支处理", ["human", "kb"]
        )
        assert result == "human"


class TestExecutionPlanConditional:
    """Test ExecutionPlan conditional branch management"""

    def test_set_active_branch(self):
        """Test setting active branch for conditional node"""
        plan = ExecutionPlan(
            id="test_plan",
            goal="Test goal",
            steps=[
                PlanStep(
                    id="conditional_step",
                    name="Conditional",
                    description="Test conditional",
                    conditional_branches={"human": "step_1", "kb": "step_2"},
                )
            ],
        )

        plan.set_active_branch("conditional_step", "human")
        assert plan.active_branches["conditional_step"] == "human"

    def test_set_active_branch_invalid_step(self):
        """Test set_active_branch with invalid step ID"""
        plan = ExecutionPlan(id="test_plan", goal="Test goal", steps=[])

        with pytest.raises(ValueError, match="Step.*not found"):
            plan.set_active_branch("nonexistent", "human")

    def test_set_active_branch_invalid_key(self):
        """Test set_active_branch with invalid branch key"""
        plan = ExecutionPlan(
            id="test_plan",
            goal="Test goal",
            steps=[
                PlanStep(
                    id="conditional_step",
                    name="Conditional",
                    description="Test",
                    conditional_branches={"human": "step_1"},
                )
            ],
        )

        with pytest.raises(ValueError, match="Invalid branch key"):
            plan.set_active_branch("conditional_step", "kb")

    def test_get_executable_steps_with_branches(self):
        """Test get_executable_steps respects active branches"""
        plan = ExecutionPlan(
            id="test_plan",
            goal="Test goal",
            steps=[
                PlanStep(id="start", name="Start", description="Start node"),
                PlanStep(
                    id="conditional",
                    name="Conditional",
                    description="Conditional node",
                    dependencies=["start"],  # Conditional depends on start
                    conditional_branches={"human": "human_step", "kb": "kb_step"},
                ),
                PlanStep(
                    id="human_step",
                    name="Human Response",
                    description="Human response",
                    dependencies=["conditional"],
                    required_branch="human",
                ),
                PlanStep(
                    id="kb_step",
                    name="KB Search",
                    description="KB search",
                    dependencies=["conditional"],
                    required_branch="kb",
                ),
            ],
        )

        # Start node should always be executable
        executable = plan.get_executable_steps(set(), set())
        assert len(executable) == 1
        assert executable[0].id == "start"

        # After start and conditional complete, with "human" branch selected
        plan.active_branches["conditional"] = "human"
        plan.steps[0].status = StepStatus.COMPLETED
        plan.steps[1].status = StepStatus.COMPLETED

        executable = plan.get_executable_steps({"start", "conditional"}, set())
        assert len(executable) == 1
        assert executable[0].id == "human_step"  # Only human_step should be executable
        # kb_step should be skipped because it requires "kb" branch

        # Same with "kb" branch selected
        plan.active_branches["conditional"] = "kb"
        executable = plan.get_executable_steps({"start", "conditional"}, set())
        assert len(executable) == 1
        assert executable[0].id == "kb_step"


class TestConditionalWorkflow:
    """Integration test for conditional workflow execution"""

    def test_conditional_workflow_structure(self):
        """Test a complete conditional workflow can be constructed"""
        # Simulating task 56 structure: check if human needed
        plan = ExecutionPlan(
            id="task_56_plan",
            goal="Handle customer support query",
            steps=[
                PlanStep(
                    id="check_human",
                    name="Check Human Request",
                    description="Determine if user explicitly requested human assistance",
                    conditional_branches={"human": "human_response", "kb": "kb_search"},
                ),
                PlanStep(
                    id="human_response",
                    name="Human Response",
                    description="Reply with human connection message",
                    dependencies=["check_human"],
                    required_branch="human",
                ),
                PlanStep(
                    id="kb_search",
                    name="KB Search",
                    description="Search knowledge base",
                    dependencies=["check_human"],
                    required_branch="kb",
                    tool_names=["knowledge_base_search"],
                ),
                PlanStep(
                    id="end",
                    name="End",
                    description="End node",
                    dependencies=["human_response", "kb_search"],
                ),
            ],
        )

        # Verify structure
        assert plan.steps[0].is_conditional is True
        assert len(plan.steps[0].conditional_branches) == 2
        assert plan.steps[1].required_branch == "human"
        assert plan.steps[2].required_branch == "kb"

        # Simulate "kb" branch selection
        plan.set_active_branch("check_human", "kb")

        # Verify only kb_search is executable (human_response is skipped)
        plan.steps[0].status = StepStatus.COMPLETED
        executable = plan.get_executable_steps({"check_human"}, set())
        assert len(executable) == 1
        assert executable[0].id == "kb_search"


class TestConditionalValidation:
    """Test validation of conditional branches"""

    def test_set_active_branch_invalid_step_id(self):
        """Test set_active_branch with non-existent step ID"""
        plan = ExecutionPlan(
            id="test_plan",
            goal="Test goal",
            steps=[
                PlanStep(
                    id="conditional_step",
                    name="Conditional",
                    description="Test",
                    conditional_branches={"human": "step_1", "kb": "step_2"},
                )
            ],
        )

        # Try to set a branch that points to non-existent step
        # This should have been validated at plan creation time
        # But we test set_active_branch validation as well
        with pytest.raises(ValueError, match="Step.*not found"):
            plan.set_active_branch("nonexistent_step", "human")

    def test_conditional_branch_must_point_to_existing_step(self):
        """Test that conditional branches must point to existing steps"""
        from xagent.core.agent.exceptions import DAGPlanGenerationError
        from xagent.core.agent.pattern.dag_plan_execute.plan_generator import (
            PlanGenerator,
        )

        plan = ExecutionPlan(
            id="invalid_plan",
            goal="Test with invalid branches",
            steps=[
                PlanStep(
                    id="conditional",
                    name="Conditional",
                    description="Test",
                    conditional_branches={
                        "human": "human_step",  # This step exists
                        "kb": "nonexistent_step",  # This step doesn't exist!
                    },
                ),
                PlanStep(id="human_step", name="Human", description="Human response"),
            ],
        )

        # Create a PlanGenerator to validate
        llm = None  # Not needed for validation
        generator = PlanGenerator(llm)

        # Validation should fail because "kb" branch points to non-existent step
        with pytest.raises(DAGPlanGenerationError, match="non-existent steps"):
            generator._validate_plan(plan, tools=[])

    def test_conditional_branch_validation_passes(self):
        """Test that validation passes when all branches point to existing steps"""
        from xagent.core.agent.pattern.dag_plan_execute.plan_generator import (
            PlanGenerator,
        )

        plan = ExecutionPlan(
            id="valid_plan",
            goal="Test with valid branches",
            steps=[
                PlanStep(
                    id="conditional",
                    name="Conditional",
                    description="Test",
                    conditional_branches={
                        "human": "human_step",  # Both exist
                        "kb": "kb_step",
                    },
                ),
                PlanStep(id="human_step", name="Human", description="Human response"),
                PlanStep(id="kb_step", name="KB Search", description="KB search"),
            ],
        )

        # Create a PlanGenerator to validate
        llm = None  # Not needed for validation
        generator = PlanGenerator(llm)

        # Should not raise any exception
        generator._validate_plan(plan, tools=[])

        # And set_active_branch should work
        plan.set_active_branch("conditional", "human")
        assert plan.active_branches["conditional"] == "human"
