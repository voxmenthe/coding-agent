import logging
import threading
from typing import Optional, Type
from sentence_transformers import SentenceTransformer

from .adapter import MemoryDoc, MemoryAdapter
from .hybrid_sqlite import HybridSQLiteAdapter # Default adapter
from src.config import config

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
    _embedding_model: Optional[SentenceTransformer] = None # model storage

    def __new__(cls, adapter_class: Type[MemoryAdapter] = HybridSQLiteAdapter, 
                db_path: str = ".memory_db/memory.db", 
                embedding_dir: str = ".memory_db/embeddings", 
                embedding_model_name: str = config["EMBEDDING_MODEL"] or "intfloat/multilingual-e5-small") -> "MemoryService":
        """Implement the singleton pattern.

        Ensures only one instance of MemoryService exists. Initializes the adapter
        and loads the embedding model on first creation.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    log.info("Creating MemoryService instance.")
                    cls._instance = super().__new__(cls)
                    # Load embedding model first
                    try:
                        log.info(f"Loading embedding model: {embedding_model_name}")
                        cls._embedding_model = SentenceTransformer(embedding_model_name)
                        log.info("Embedding model loaded successfully.")
                    except Exception as e:
                        log.error(f"Failed to load embedding model: {embedding_model_name} - {e}", exc_info=True)
                        cls._instance = None
                        cls._embedding_model = None
                        raise RuntimeError("MemoryService failed to load embedding model") from e
                    
                    # Initialize the adapter *within* the lock during first creation
                    try:
                        log.info(f"Initializing adapter: {adapter_class.__name__}")
                        # Pass the loaded embedding model to the adapter
                        cls._adapter_instance = adapter_class(
                            db_path_str=db_path, 
                            embedding_dir_str=embedding_dir,
                            embedding_model=cls._embedding_model
                        )
                        log.info("MemoryService adapter initialized successfully.")
                    except Exception as e:
                        log.error(f"Failed to initialize memory adapter: {e}", exc_info=True)
                        # Prevent service creation if adapter fails
                        cls._instance = None 
                        cls._adapter_instance = None
                        cls._embedding_model = None
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
                     query_mode: str = "fts", # "fts" or "semantic"
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
            ValueError: If an invalid query_mode is provided.
        """
        if self._adapter_instance is None:
            log.error("Memory adapter not initialized. Cannot query memory.")
            raise RuntimeError("MemoryService adapter is not initialized.")
            
        log.debug(f"MemoryService delegating query_memory (mode: {query_mode}) for: '{query_text[:50]}...', k={k}")
        
        if query_mode == "fts":
            # The original query method might return dicts, needs checking/updating
            # For now, assume adapter.query returns list[MemoryDoc] or is adapted
            return self._adapter_instance.query(
                query_text,
                k=k,
                filter_tags=filter_tags,
                filter_source_agents=filter_source_agents
            )
            # Ensure results are MemoryDoc objects (adapter needs update if not)
            if results and not isinstance(results[0], MemoryDoc):
                log.warning("Adapter query method did not return MemoryDoc objects. Attempting conversions")
                # Convert assuming dict structure
                return [MemoryDoc(id=r.get('uuid'), text=r.get('text_content')) for r in results]
            return results
        elif query_mode == "semantic":
            if hasattr(self._adapter_instance, 'semantic_query'):
                return self._adapter_instance.semantic_query(
                    query_text,
                    k=k,
                    filter_tags=filter_tags,
                filter_source_agents=filter_source_agents
            )
        else:
            log.error(f"Adapter {type(self._adapter_instance).__name__} does not support semantic_query mode.")
            raise NotImplementedError("Semantic query not supported by current adapter.")

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
    assert service1._embedding_model is not None
    assert service1._adapter_instance is not None
    print("Singleton test passed.")

    print("\n--- Testing MemoryService Add/Query ---")
    # Use unique IDs for MemoryDoc objects
    doc1_id_str = str(uuid.uuid4())
    doc1 = MemoryDoc(
        id=doc1_id_str,
        text="First memory added via MemoryService about cats.",
        tags=["service_test", "animal"],
        source_agent="service_tester"
        # No need to manually set timestamp or metadata if defaults are ok
    )
    added_id = service1.add_memory(doc1)
    print(f"Added doc ID: {added_id}")
    assert added_id == doc1_id_str

    time.sleep(0.1) # Ensure timestamp difference

    doc2_id_str = str(uuid.uuid4())
    doc2 = MemoryDoc(
        id=doc2_id_str,
        text="Second memory about dogs, also added via MemoryService.",
        tags=["service_test", "animal"],
        source_agent="service_tester"
    )
    added_id_2 = service1.add_memory(doc2)
    print(f"Added doc ID: {added_id_2}")
    assert added_id_2 == doc2_id_str

    print("\nQuerying for 'cats':")
    results_cats = service1.query_memory("cats")
    for doc in results_cats:
        print(f" - Found: {doc.text[:50]}... (ID: {doc.id})")
        assert doc.id == doc1_id_str
    assert len(results_cats) == 1
    
    print("\nQuerying for 'memory':")
    results_memory = service1.query_memory("memory")
    for doc in results_memory:
        print(f" - Found: {doc.text[:50]}... (ID: {doc.id})")
    assert len(results_memory) == 2 # Both docs contain 'memory'

    print("\nAdding third doc via MemoryService:")
    doc3_id_str = str(uuid.uuid4())
    doc3 = MemoryDoc(
        id=doc3_id_str,
        text="Let's play fetch in the park.",
        tags=["service_test", "activity"],
        source_agent="service_tester",
    )
    added_id_3 = service1.add_memory(doc3)
    print(f"Added doc ID: {added_id_3}")
    assert added_id_3 == doc3_id_str

    print("\nQuerying FTS for 'sunny':")
    results_fts = service1.query_memory("sunny", query_mode="fts")
    print(f"Found {len(results_fts)} FTS results.")
    for doc in results_fts:
        print(f" - Found: {doc.text[:50]}... (ID: {doc.id})")
        assert isinstance(doc, MemoryDoc) # Check type
        assert doc.id == doc2_id_str
    assert len(results_fts) == 1

    print("\nQuerying Semantic for 'pleasant weather':")
    results_semantic = service1.query_memory("pleasant weather", k=2, query_mode="semantic")
    print(f"Found {len(results_semantic)} Semantic results.")
    for doc in results_semantic:
        print(f" - Found: {doc.text[:50]}... (ID: {doc.id}, Score: {doc.score:.4f})")
        assert isinstance(doc, MemoryDoc)
        assert doc.score is not None
    # Expecting doc1 and doc2 to be most similar
    assert len(results_semantic) == 2
    result_ids_semantic = {doc.id for doc in results_semantic}
    assert doc1_id_str in result_ids_semantic
    assert doc2_id_str in result_ids_semantic
    # Check ranking (higher score first)
    assert results_semantic[0].score >= results_semantic[1].score

    # Verify embeddings were saved (requires access to adapter details, or test separately)
    # Example: Check if embedding_path exists for the first doc added
    # This check is better suited for adapter-level tests
    # retrieved_doc1 = service1._adapter_instance.get_by_id(doc1_id_str) # Assuming get_by_id exists
    # assert retrieved_doc1 and retrieved_doc1.embedding_path is not None
    # assert Path(retrieved_doc1.embedding_path).exists()
    # TODO: Add adapter-level tests
    # Once we've implemented the full functionality of all classes we can move these tests
    # into the test module

    print("\nAdd/Query test passed.")

    # Clean up
    service1.close_adapter()
    print("\nMemoryService testing complete.")
