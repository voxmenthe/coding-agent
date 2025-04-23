from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List


@dataclass
class MemoryDoc:
    """Represents a single document or piece of information in the memory system."""
    id: str  # Unique identifier (e.g., UUID)
    text: str  # The actual content
    tags: List[str] = field(default_factory=list)
    source_agent: str = ""  # Identifier for the agent that created this memory
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)  # Any other relevant info
    embedding_path: str | None = None # Path to the saved embedding (.npy file)
    score: float | None = None # Relevance score from a query


class MemoryAdapter(ABC):
    """Abstract Base Class defining the interface for memory storage and retrieval."""

    @abstractmethod
    def add(self, doc: MemoryDoc) -> str:
        """Adds a MemoryDoc to the memory store.

        Args:
            doc: The MemoryDoc object to add.

        Returns:
            The ID of the added document.
        """
        ...

    @abstractmethod
    def query(self, query_text: str, *, k: int = 10,
              filter_tags: List[str] | None = None,
              filter_source_agents: List[str] | None = None) -> List[MemoryDoc]:
        """Queries the memory store for relevant documents.

        Args:
            query_text: The text to search for.
            k: The maximum number of results to return.
            filter_tags: Optional list of tags to filter results by.
            filter_source_agents: Optional list of source agent IDs to filter by.

        Returns:
            A list of relevant MemoryDoc objects, potentially ranked.
        """
        ...

    # Optional: Add methods for deletion, updating, fetching by ID etc. if needed later
    # @abstractmethod
    # def get_by_id(self, doc_id: str) -> MemoryDoc | None:
    #     ...

    # @abstractmethod
    # def delete(self, doc_id: str) -> bool:
    #     ...
