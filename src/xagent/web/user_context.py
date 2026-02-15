"""
User context management for MCP tools

This module provides utilities for managing user context in MCP tool execution,
ensuring proper user isolation and security.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


class UserContext:
    """Manages user context for MCP tool execution."""

    def __init__(self, user_id: Optional[str] = None):
        """Initialize user context.

        Args:
            user_id: User ID for context
        """
        self.user_id = user_id
        self._original_user_id: Optional[str] = None

    @contextmanager
    def set_context(self) -> Iterator[None]:
        """Set user context as environment variable for MCP tools."""
        try:
            # Store original user ID
            self._original_user_id = os.environ.get("XAGENT_USER_ID")

            # Set current user ID
            if self.user_id:
                os.environ["XAGENT_USER_ID"] = self.user_id
                logger.debug(f"Set user context to: {self.user_id}")

            yield

        finally:
            # Restore original user ID
            if self._original_user_id is not None:
                os.environ["XAGENT_USER_ID"] = self._original_user_id
            elif "XAGENT_USER_ID" in os.environ:
                del os.environ["XAGENT_USER_ID"]

            logger.debug("Restored original user context")

    def get_current_user(self) -> Optional[str]:
        """Get current user ID from environment."""
        return os.environ.get("XAGENT_USER_ID")


def get_user_context_from_request(request: Any) -> Optional[str]:
    """Extract user context from FastAPI request.

    Args:
        request: FastAPI request object

    Returns:
        User ID string or None
    """
    try:
        from .auth_dependencies import get_current_user_optional
        from .models.database import get_db

        # Try to get user from authentication
        db = next(get_db())
        user = get_current_user_optional(request, db)

        if user:
            return str(user.id)

    except Exception as e:
        logger.error(f"Failed to get user context from request: {e}")

    return None


def create_user_context(
    user_id: Optional[str] = None, request: Optional[Any] = None
) -> UserContext:
    """Create user context from various sources.

    Args:
        user_id: Explicit user ID
        request: FastAPI request object

    Returns:
        UserContext instance
    """
    if user_id:
        return UserContext(user_id)

    if request:
        user_id = get_user_context_from_request(request)
        if user_id:
            return UserContext(user_id)

    # Fallback to environment variable
    env_user_id = os.environ.get("XAGENT_USER_ID")
    if env_user_id:
        return UserContext(env_user_id)

    # No user context available
    return UserContext()


def validate_user_access(user_id: Optional[str], allowed_users: Optional[list]) -> bool:
    """Validate user access to MCP tools.

    Args:
        user_id: Current user ID
        allowed_users: List of allowed user IDs

    Returns:
        True if user has access, False otherwise
    """
    if not user_id:
        # No user ID provided - check if system access is allowed
        return allowed_users is None or "system" in allowed_users

    if allowed_users is None:
        # No specific restrictions
        return True

    return user_id in allowed_users
