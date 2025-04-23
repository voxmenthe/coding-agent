import logging
import threading
from typing import Optional, Type

from .adapter import MemoryDoc, MemoryAdapter
from .hybrid_sqlite import HybridSQLiteAdapter # Default adapter

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class MemoryService:
    """Singleton service providing access to the memory system.

    Manages an instance of a MemoryAdapter (defaulting to HybridSQLiteAdapter)
    and provides methods for adding and querying memories.
    Handles adapter initialization and teardown.
    """
    _instance: Optional["MemoryService"] = None
    _lock: threading.Lock = threading.Lock()
    _adapter_instance: Optional[MemoryAdapter] = None

    def __new__(cls, adapter_class: Type[MemoryAdapter] = HybridSQLiteAdapter, 
                db_path: str = ".memory_db/memory.db", 
                embedding_dir: str = ".memory_db/embeddings") -> "MemoryService":
        """Implement the singleton pattern.

        Ensures only one instance of MemoryService exists. Initializes the adapter
        on first creation.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    log.info("Creating MemoryService instance.")
                    cls._instance = super().__new__(cls)
                    # Initialize the adapter *within* the lock during first creation
                    try:
                        log.info(f"Initializing adapter: {adapter_class.__name__}")
                        cls._adapter_instance = adapter_class(db_path_str=db_path, embedding_dir_str=embedding_dir)
                        log.info("MemoryService adapter initialized successfully.")
                    except Exception as e:
                        log.error(f"Failed to initialize memory adapter: {e}", exc_info=True)
                        # Prevent service creation if adapter fails
                        cls._instance = None 
                        cls._adapter_instance = None
                        raise RuntimeError("MemoryService failed to initialize adapter") from e
                else:
                    log.debug("MemoryService instance already exists (inner check).")
        else:
            log.debug("MemoryService instance already exists (outer check).")
        
        return cls._instance

    def add_memory(self, doc: MemoryDoc) -> str:
        """Adds a memory document using the configured adapter.

        Args:
            doc: The MemoryDoc to add.

        Returns:
            The ID of the added document.
            
        Raises:
            RuntimeError: If the adapter is not initialized.
        """
        if self._adapter_instance is None:
            log.error("Memory adapter not initialized. Cannot add memory.")
            raise RuntimeError("MemoryService adapter is not initialized.")
        
        log.debug(f"MemoryService delegating add_memory for doc ID: {doc.id}")
        # In Week 1, this directly calls the synchronous adapter method.
        # In Week 2, this will become async and put the doc onto a queue.
        return self._adapter_instance.add(doc)

    def query_memory(self, query_text: str, *, k: int = 10, 
                     filter_tags: list[str] | None = None, 
                     filter_source_agents: list[str] | None = None) -> list[MemoryDoc]:
        """Queries for memories using the configured adapter.
        
        Args:
            query_text: The text query.
            k: Max number of results.
            filter_tags: Optional list of tags to filter by.
            filter_source_agents: Optional list of agent IDs to filter by.
            
        Returns:
            A list of relevant MemoryDoc objects.
            
        Raises:
            RuntimeError: If the adapter is not initialized.
        """
        if self._adapter_instance is None:
            log.error("Memory adapter not initialized. Cannot query memory.")
            raise RuntimeError("MemoryService adapter is not initialized.")
            
        log.debug(f"MemoryService delegating query_memory for: '{query_text[:50]}...', k={k}")
        # This remains a direct call to the adapter's query method.
        return self._adapter_instance.query(
            query_text,
            k=k,
            filter_tags=filter_tags,
            filter_source_agents=filter_source_agents
        )

    def close_adapter(self):
        """Safely closes the adapter's resources (e.g., DB connection)."""
        if self._adapter_instance and hasattr(self._adapter_instance, 'close'):
            log.info("Closing memory adapter resources.")
            self._adapter_instance.close()
        else:
            log.debug("No adapter instance or close method found.")

# Optional: Function to cleanly shutdown the service if needed application-wide
# import atexit
# atexit.register(MemoryService().close_adapter)

# Example usage (demonstrates singleton and basic calls)
if __name__ == '__main__':
    import uuid
    import time

    print("\n--- Testing MemoryService Singleton ---")
    service1 = MemoryService(db_path=":memory:") # Use in-memory DB
    service2 = MemoryService() # Should return the same instance

    print(f"Service 1 ID: {id(service1)}")
    print(f"Service 2 ID: {id(service2)}")
    assert id(service1) == id(service2)
    print("Singleton test passed.")

    print("\n--- Testing MemoryService Add/Query ---")
    doc1_id = str(uuid.uuid4())
    doc1 = MemoryDoc(
        id=doc1_id,
        text="First memory added via MemoryService about cats.",
        tags=["service_test", "animal"],
        source_agent="service_tester"
    )
    added_id = service1.add_memory(doc1)
    print(f"Added doc ID: {added_id}")
    assert added_id == doc1_id

    time.sleep(0.1) # Ensure timestamp difference

    doc2_id = str(uuid.uuid4())
    doc2 = MemoryDoc(
        id=doc2_id,
        text="Second memory about dogs, also added via MemoryService.",
        tags=["service_test", "animal"],
        source_agent="service_tester"
    )
    added_id_2 = service1.add_memory(doc2)
    print(f"Added doc ID: {added_id_2}")
    assert added_id_2 == doc2_id

    print("\nQuerying for 'cats':")
    results_cats = service1.query_memory("cats")
    for doc in results_cats:
        print(f" - Found: {doc.text[:50]}... (ID: {doc.id})")
        assert doc.id == doc1_id
    assert len(results_cats) == 1
    
    print("\nQuerying for 'memory':")
    results_memory = service1.query_memory("memory")
    for doc in results_memory:
        print(f" - Found: {doc.text[:50]}... (ID: {doc.id})")
    assert len(results_memory) == 2 # Both docs contain 'memory'

    print("\nAdd/Query test passed.")

    # Clean up
    service1.close_adapter()
    print("\nMemoryService testing complete.")
