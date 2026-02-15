"""
Vector store providers module.

This module provides various vector storage backends that implement
the standard VectorStore interface.
"""

from xagent.providers.vector_store.base import VectorStore
from xagent.providers.vector_store.chroma import ChromaVectorStore
from xagent.providers.vector_store.lancedb import (
    LanceDBConnectionManager,
    LanceDBVectorStore,
)

__all__ = [
    "VectorStore",
    "ChromaVectorStore",
    "LanceDBVectorStore",
    "LanceDBConnectionManager",
]
