"""
Simple memory integration utilities for existing agent patterns.
Provides straightforward helper functions to add memory functionality.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...memory import MemoryStore
from ...memory.core import MemoryNote

logger = logging.getLogger(__name__)


def enhance_goal_with_memory(goal: str, memories: List[Dict[str, Any]]) -> str:
    """
    Enhance goal description with relevant memories using simple natural text.

    Args:
        goal: Original goal description
        memories: List of relevant memories

    Returns:
        Enhanced goal description
    """
    if not memories:
        logger.info("No memories found, returning original goal")
        return goal

    logger.info(f"Enhancing goal with {len(memories)} memories")

    # Simply combine memories as natural text context
    context_parts = []

    for mem in memories:
        content = mem.get("content", "")
        if content.strip():
            context_parts.append(f"• {content}")

    # Combine goal with context
    if context_parts:
        context_text = "\n".join(context_parts)
        enhanced_goal = (
            f"{goal}\n\nRelevant Context from Previous Experience:\n{context_text}"
        )
        logger.info(f"Goal enhanced with {len(context_parts)} insights")
        return enhanced_goal
    else:
        return goal


def store_plan_generation_memory(
    memory_store: MemoryStore,
    goal: str,
    steps_count: int,
    plan_id: Optional[str] = None,
    planning_insights: Optional[str] = None,
    classification: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Store memory for plan generation with useful insights.

    Args:
        memory_store: Memory store instance
        goal: The goal that was planned
        steps_count: Number of steps generated
        plan_id: Optional plan identifier
        planning_insights: Insights and strategies learned during planning
        classification: Classification data from comprehensive insights

    Returns:
        Memory ID if successful, None otherwise
    """
    try:
        # Use classification data from comprehensive insights or fallback
        if classification:
            domain_keywords = classification.get("keywords", [])
            task_type = classification.get("task_type", "unknown")
        else:
            domain_keywords = []
            task_type = "unknown"

        content = f"Task Planning Experience: {goal}\n\n"
        content += (
            f"Planning Strategy: Generated execution plan with {steps_count} steps"
        )
        if planning_insights:
            content += f"\n\nKey Insights:\n{planning_insights}"

        note = MemoryNote(
            content=content,
            keywords=["task planning", "strategy design", "step decomposition"]
            + domain_keywords,
            category="dag_plan_execute_memory",
            metadata={
                "operation": "dag_plan_generation",
                "goal_type": task_type,
                "steps_count": steps_count,
                "plan_id": plan_id,
                "domain_keywords": domain_keywords,
                "timestamp": datetime.now().isoformat(),
            },
        )

        response = memory_store.add(note)
        logger.info(f"Stored planning memory for goal: {goal[:50]}...")
        return response.memory_id if response.success else None
    except Exception as e:
        logger.error(f"Failed to store plan generation memory: {e}")
        return None


def store_execution_result_memory(
    memory_store: MemoryStore,
    results: List[Any],
    goal: str,
    plan_id: Optional[str] = None,
    execution_insights: Optional[str] = None,
    failures_and_learnings: Optional[str] = None,
    classification: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Store memory for execution results with useful insights.

    Args:
        memory_store: Memory store instance
        results: Execution results
        goal: Original goal for context
        plan_id: Optional plan identifier
        execution_insights: Insights learned during execution
        failures_and_learnings: Failure analysis and lessons learned
        classification: Classification data from comprehensive insights

    Returns:
        Memory ID if successful, None otherwise
    """
    try:
        # Analyze execution results
        successful_steps = len([r for r in results if r.get("status") == "completed"])
        failed_steps = len([r for r in results if r.get("status") == "failed"])

        # Build natural language memory content focused on user preferences and insights
        content_parts = []

        # Basic execution information
        content_parts.append(f"Goal: {goal}")
        content_parts.append(
            f"Execution result: {successful_steps} successful steps out of {len(results)} total steps"
        )

        # User preferences and behavioral patterns (most important!)
        if classification and classification.get("user_preferences"):
            content_parts.append(
                f"User preferences identified: {classification['user_preferences']}"
            )

        if classification and classification.get("behavioral_patterns"):
            content_parts.append(
                f"User behavior patterns: {classification['behavioral_patterns']}"
            )

        # Execution insights as natural text
        if execution_insights and execution_insights.strip():
            content_parts.append(f"Key execution insights: {execution_insights}")

        # Success factors (very valuable for future tasks)
        if classification and classification.get("success_factors"):
            content_parts.append(
                f"Success factors: {classification['success_factors']}"
            )

        # Learned patterns and strategies
        if classification and classification.get("learned_patterns"):
            content_parts.append(
                f"Reusable patterns: {classification['learned_patterns']}"
            )

        # Failure analysis and improvement suggestions
        if failed_steps > 0:
            if failures_and_learnings and failures_and_learnings.strip():
                content_parts.append(f"Failure analysis: {failures_and_learnings}")
            if classification and classification.get("improvement_suggestions"):
                content_parts.append(
                    f"Improvement suggestions: {classification['improvement_suggestions']}"
                )

        # Additional context from classification
        if classification:
            extra_insights = []
            if classification.get("execution_insights"):
                extra_insights.append(
                    f"Execution approach: {classification['execution_insights']}"
                )
            if classification.get("failure_analysis"):
                extra_insights.append(
                    f"Failure analysis: {classification['failure_analysis']}"
                )

            if extra_insights:
                content_parts.append("Additional insights:")
                content_parts.extend(f"  - {insight}" for insight in extra_insights)

        # Join all parts with newlines
        content = "\n".join(content_parts)

        # Let LLM handle semantic search with minimal keywords
        keywords = ["execution", "planning"]

        note = MemoryNote(
            content=content,
            keywords=keywords,
            category="execution_memory",
            metadata={
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "total_steps": len(results),
                "has_user_preferences": bool(
                    classification and classification.get("user_preferences")
                ),
                "timestamp": datetime.now().isoformat(),
            },
        )

        response = memory_store.add(note)
        logger.info(f"Stored execution memory with user preferences: {goal[:50]}...")
        return response.memory_id if response.success else None
    except Exception as e:
        logger.error(f"Failed to store execution result memory: {e}")
        return None


def store_react_task_memory(
    memory_store: MemoryStore,
    task: str,
    result: Dict[str, Any],
    tool_usage_insights: Optional[str] = None,
    reasoning_strategy: Optional[str] = None,
    classification: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Store memory for ReAct task completion as simple, actionable insights.

    Args:
        memory_store: Memory store instance
        task: The task that was executed
        result: Task execution result
        tool_usage_insights: Insights about tool usage effectiveness
        reasoning_strategy: Strategy used for reasoning and problem solving
        classification: Classification data (will be converted to natural text)

    Returns:
        Memory ID if successful, None otherwise
    """
    try:
        success = result.get("success", False)

        # Build focused memory content - only essential insights
        content_parts = []

        # Core information
        content_parts.append(f"Task: {task}")
        content_parts.append(f"Outcome: {'Success' if success else 'Failed'}")

        # Focus on the most valuable insights only
        if classification:
            # User preferences (highest priority)
            if classification.get("user_preferences"):
                content_parts.append(f"User: {classification['user_preferences']}")

            # Core insight if available (from new prompt format)
            if classification.get("core_insight"):
                content_parts.append(f"Core: {classification['core_insight']}")

            # Failure patterns (valuable for avoiding future mistakes)
            if classification.get("failure_patterns"):
                content_parts.append(f"Failure: {classification['failure_patterns']}")

            # Success patterns (only if truly exceptional)
            if classification.get("success_patterns"):
                content_parts.append(f"Success: {classification['success_patterns']}")

        # Only include tool insights if they reveal something non-obvious
        if tool_usage_insights and len(tool_usage_insights.strip()) > 20:
            # Avoid generic "tool used effectively" descriptions
            content_parts.append(f"Tools: {tool_usage_insights}")

        content = "\n".join(content_parts)

        # Let LLM generate appropriate keywords through semantic search
        keywords = ["react", "execution"]

        note = MemoryNote(
            content=content,
            keywords=keywords,
            category="react_memory",
            metadata={
                "task": task,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            },
        )

        response = memory_store.add(note)
        logger.info(f"Stored ReAct memory: {task[:50]}...")
        return response.memory_id if response.success else None
    except Exception as e:
        logger.error(f"Failed to store ReAct task memory: {e}")
        return None


# Domain classification and keyword extraction is now handled by the comprehensive insights generation
# This reduces multiple LLM calls to a single comprehensive analysis


def lookup_relevant_memories(
    memory_store: MemoryStore,
    query: str,
    category: Optional[str] = None,
    include_general: bool = True,
    limit: int = 5,
    similarity_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Look up relevant memories.

    Args:
        memory_store: Memory store instance
        query: Search query
        category: Optional memory category filter (system memory category)
        include_general: Whether to also include general (user) memories
        limit: Maximum number of results
        similarity_threshold: Optional similarity threshold for vector search (0.1-2.0)

    Returns:
        List of relevant memories
    """
    try:
        all_memories = []
        search_categories = []

        # If category specified, search that category (system memories)
        if category:
            search_categories.append(category)

        # Also search general memories if requested
        if include_general:
            search_categories.append("general")

        logger.info(
            f"Looking up memories for query: '{query[:100]}...' in categories: {search_categories}"
        )

        # Search each category
        for cat in search_categories:
            filters = {"category": cat}
            category_memories = memory_store.search(
                query=query,
                k=limit,
                filters=filters,
                similarity_threshold=similarity_threshold,
            )
            all_memories.extend(category_memories)
            logger.info(f"Found {len(category_memories)} memories in category '{cat}'")

        # Remove duplicates based on content (handle both string and bytes)
        seen_contents = set()
        unique_memories = []
        for memory in all_memories:
            # Normalize content to string for comparison
            content = memory.content
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            if content not in seen_contents:
                seen_contents.add(content)
                unique_memories.append(memory)

        # Limit total results
        final_memories = unique_memories[:limit]

        logger.info(
            f"Returning {len(final_memories)} unique memories (removed {len(all_memories) - len(final_memories)} duplicates)"
        )

        # Log memory contents for debugging
        for i, memory in enumerate(final_memories):
            category = getattr(memory, "category", "unknown")
            logger.info(f"Memory {i + 1}: [{category}] {memory.content[:100]!r}...")

        return [
            {
                "content": memory.content,
                "keywords": memory.keywords,
                "metadata": memory.metadata,
            }
            for memory in final_memories
        ]
    except Exception as e:
        logger.error(
            f"Failed to lookup relevant memories for query '{query[:50]}...': {e}"
        )
        return []


# Simple usage examples:
#
# In DAG plan generation:
# memories = lookup_relevant_memories(memory_store, goal, "dag_plan_execute_memory")
# enhanced_goal = enhance_goal_with_memory(goal, memories)
# # Use enhanced_goal for planning
# store_plan_generation_memory(memory_store, goal, len(plan.steps))
#
# In ReAct pattern:
# memories = lookup_relevant_memories(memory_store, task, "react_memory")
# enhanced_task = enhance_goal_with_memory(task, memories)
# # Use enhanced_task for execution
# store_react_task_memory(memory_store, task, result)
