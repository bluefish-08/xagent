"""User-isolated memory store factory for web application."""

from .memory_utils import create_memory_store
from .user_isolated_memory import UserIsolatedMemoryStore

# Create base memory store instance, then wrap as user-isolated memory store
base_memory_store = create_memory_store()
global_memory_store = UserIsolatedMemoryStore(base_memory_store)

__all__ = ["global_memory_store"]
