import pytest
import sqlite3
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from unittest.mock import patch, MagicMock

from src.memory.adapter import MemoryDoc
from src.memory.hybrid_sqlite import HybridSQLiteAdapter
# Assuming EmbeddingManager is importable relative to HybridSQLiteAdapter for mocking
# from src.memory.embedding_manager import EmbeddingManager 

# Fixture to create a temporary directory for test databases/embeddings
@pytest.fixture(scope="function") # Use function scope for isolation between tests
def temp_db_dir(tmp_path):
    db_dir = tmp_path / "memory_db"
    emb_dir = db_dir / "embeddings"
    # Adapter init should create dirs
    # db_dir.mkdir(parents=True, exist_ok=True)
    # emb_dir.mkdir(parents=True, exist_ok=True)
    return db_dir, emb_dir

# Renamed fixture to provide paths only
@pytest.fixture
def db_paths(temp_db_dir):
    db_dir, emb_dir = temp_db_dir
    db_path = db_dir / "test_memory.db"
    emb_path_str = str(emb_dir)
    return str(db_path), emb_path_str

@pytest.fixture
def sample_doc_data():
    """Provides dictionary data for a sample MemoryDoc."""
    return {
        "id": str(uuid.uuid4()),
        "text": "This is a sample document.",
        "timestamp": datetime.now(timezone.utc),
        "source_agent": "TestAgent1",
        "tags": ["sample", "test"],
        "metadata": {"key": "value", "source": "fixture"}
    }

@pytest.fixture
def mock_embedding_manager():
    """Provides a fully configured MagicMock instance simulating EmbeddingManager."""
    manager = MagicMock(name="MockEmbeddingManagerInstance")
    manager.embedding_model = MagicMock(name="MockEmbeddingModel") # Simulate having a model
    manager.embedding_dir = Path("/mock/embeddings") # Use a distinct mock path
    
    # Mock generate_and_save_embedding to return a predictable mock path
    def mock_gen_save(text, doc_id):
        # Return the mock path, adapter should store this string
        mock_path = manager.embedding_dir / f"{doc_id}.npy"
        return mock_path
    manager.generate_and_save_embedding.side_effect = mock_gen_save
    manager.generate_and_save_embedding.__name__ = 'generate_and_save_embedding' # Help with debugging

    # Mock other methods needed for semantic_query tests later
    manager.generate_embedding.return_value = np.array([0.1, 0.2, 0.3])
    manager.load_embedding.return_value = np.array([0.4, 0.5, 0.6])
    manager.calculate_similarity.return_value = 0.95
    
    return manager

def test_initialization(db_paths):
    """Test if adapter initializes correctly and creates files/dirs and EmbeddingManager."""
    db_path_str, emb_path_str = db_paths
    db_path = Path(db_path_str)
    emb_dir = Path(emb_path_str)
    adapter = None
    mock_model = MagicMock(name="MockSentenceTransformer") # Mock the actual model object passed
    try:
        # Patch EmbeddingManager specifically during adapter init if needed,
        # but easier to mock the whole class as done in add/query tests.
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, 
                                      embedding_dir_str=emb_path_str,
                                      embedding_model=mock_model) 

        assert db_path.parent.exists()
        assert emb_dir.exists() # Embedding dir should be created by EmbeddingManager init
        assert db_path.exists()
        assert adapter.conn is not None
        # Check if EmbeddingManager was initialized correctly inside the adapter
        assert adapter.embedding_manager is not None
        assert adapter.embedding_manager.embedding_model == mock_model
        assert adapter.embedding_manager.embedding_dir == emb_dir

    finally:
        if adapter:
            adapter.close()

def test_schema_creation(db_paths):
    """Verify that the necessary tables and columns are created."""
    db_path_str, emb_path_str = db_paths
    adapter = None
    try:
        # No model needed just to check schema
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, embedding_dir_str=emb_path_str)
        cursor = adapter.conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories';")
        assert cursor.fetchone() is not None, "'memories' table not found."

        cursor.execute("PRAGMA table_info(memories);")
        # Use dict comprehension for easy lookup
        columns = {row['name']: row['type'] for row in cursor.fetchall()}
        
        # Verify all expected columns exist (now includes embedding_path etc.)
        expected_columns = {
            'id': 'INTEGER',
            'uuid': 'TEXT',
            'text_content': 'TEXT',
            'embedding_path': 'TEXT', # New column
            'timestamp': 'TEXT',      # New column
            'source_agent': 'TEXT',   # New column
            'tags_json': 'TEXT',      # New column
            'metadata_json': 'TEXT'   # New column
        }
        # Check if the set of column names match exactly
        assert set(columns.keys()) == set(expected_columns.keys()), \
               f"Column mismatch: Got {columns.keys()}, Expected {expected_columns.keys()}"
        # Check types individually
        for col_name, col_type in expected_columns.items():
            assert columns[col_name] == col_type, f"Column '{col_name}' has wrong type: expected {col_type}, got {columns[col_name]}"

        # Check FTS table remains correctly configured
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts';")
        assert cursor.fetchone() is not None, "'memories_fts' table not found."
        cursor.execute("SELECT sql FROM sqlite_master WHERE name='memories_fts';")
        fts_sql = cursor.fetchone()['sql']
        # Normalize whitespace somewhat, but keep internal structures
        normalized_fts_sql = ' '.join(fts_sql.lower().split())
        
        assert 'using fts5' in normalized_fts_sql
        assert 'content = memories' in normalized_fts_sql
        assert "content_rowid = 'id'" in normalized_fts_sql
        assert "tokenize = 'porter'" in normalized_fts_sql
        
    finally:
        if adapter:
            adapter.close()

@patch('src.memory.hybrid_sqlite.EmbeddingManager') 
def test_add_memory_basic(MockEmbeddingManagerClass, db_paths, sample_doc_data, mock_embedding_manager):
    """Test adding a MemoryDoc, verifying DB content and EmbeddingManager interaction."""
    db_path_str, emb_path_str = db_paths
    
    # Configure the mock instance that will be returned by MockEmbeddingManagerClass()
    # Use the pre-configured mock from the fixture
    MockEmbeddingManagerClass.return_value = mock_embedding_manager 
    # Ensure the mock manager simulates having a model attached
    mock_embedding_manager.embedding_model = MagicMock(name="MockModelOnInstance") 
    
    adapter = None
    try:
        # Initialize adapter; it will get the mocked EmbeddingManager instance upon creation
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, 
                                      embedding_dir_str=emb_path_str,
                                      embedding_model=MagicMock()) # Pass a dummy model object
        
        test_doc = MemoryDoc(**sample_doc_data)
        # Determine the expected path based on the mock manager's logic
        expected_emb_path_str = str((mock_embedding_manager.embedding_dir / f"{test_doc.id}.npy").resolve())
        
        # --- Action --- 
        added_id = adapter.add(test_doc)
        
        # --- Assertions --- 
        assert added_id == test_doc.id

        # Verify EmbeddingManager's method was called correctly by the adapter
        mock_embedding_manager.generate_and_save_embedding.assert_called_once_with(test_doc.text, test_doc.id)

        # Verify data in 'memories' table
        cursor = adapter.conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE uuid = ?", (test_doc.id,))
        row = cursor.fetchone()
        assert row is not None, "Row not found in database after add."
        assert row['uuid'] == test_doc.id
        assert row['text_content'] == test_doc.text
        assert row['embedding_path'] == expected_emb_path_str # Check correct mock path was stored
        assert row['timestamp'] == test_doc.timestamp.isoformat() # Check ISO format string
        assert row['source_agent'] == test_doc.source_agent
        assert json.loads(row['tags_json']) == test_doc.tags # Check JSON decoded tags
        assert json.loads(row['metadata_json']) == test_doc.metadata # Check JSON decoded metadata
        
        # Verify FTS entry exists for the corresponding rowid
        integer_id = row['id']
        cursor.execute("SELECT count(*) FROM memories_fts WHERE rowid = ?;", (integer_id,))
        assert cursor.fetchone()[0] == 1, "FTS entry not found for added document."

    finally:
        if adapter:
            adapter.close()

@patch('src.memory.hybrid_sqlite.EmbeddingManager')
def test_add_memory_no_embedding(MockEmbeddingManagerClass, db_paths, sample_doc_data, mock_embedding_manager):
    """Test adding a MemoryDoc when embedding generation fails or model is None."""
    db_path_str, emb_path_str = db_paths
    
    # Simulate embedding failure by having the mock return None
    # Clear side_effect first, as it takes precedence over return_value
    mock_embedding_manager.generate_and_save_embedding.side_effect = None 
    mock_embedding_manager.generate_and_save_embedding.return_value = None 
    MockEmbeddingManagerClass.return_value = mock_embedding_manager
    mock_embedding_manager.embedding_model = MagicMock() # Still need the attribute
    
    adapter = None
    try:
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, embedding_dir_str=emb_path_str, embedding_model=MagicMock())
        test_doc = MemoryDoc(**sample_doc_data)
        
        # --- Action --- 
        adapter.add(test_doc)

        # --- Assertions --- 
        # Verify manager was still called
        mock_embedding_manager.generate_and_save_embedding.assert_called_once_with(test_doc.text, test_doc.id)
        
        # Verify embedding_path is NULL in the database
        cursor = adapter.conn.cursor()
        cursor.execute("SELECT embedding_path FROM memories WHERE uuid = ?", (test_doc.id,))
        row = cursor.fetchone()
        assert row is not None, "Row not found in database."
        assert row['embedding_path'] is None, "Embedding path should be NULL when generation fails."
    finally:
        if adapter:
            adapter.close()

def test_query_memory_fts(db_paths):
    """Test querying memory using FTS (simplified schema)."""
    db_path_str, emb_path_str = db_paths
    adapter = None
    try:
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, embedding_dir_str=emb_path_str)
        results = adapter.query("anything", k=5)
        assert len(results) == 0
    finally:
        if adapter:
            adapter.close()

# --- Fixture for Semantic Query Tests --- 

@pytest.fixture
def adapter_with_docs(db_paths, mock_embedding_manager):
    """Sets up an adapter instance and adds multiple documents for semantic tests."""
    db_path_str, emb_path_str = db_paths
    
    # Use the mock manager provided by the fixture
    # Configure its behavior for the add operations
    saved_paths = {}
    def mock_gen_save_capture(text, doc_id):
        mock_path = mock_embedding_manager.embedding_dir / f"{doc_id}.npy"
        saved_paths[doc_id] = mock_path # Store path for later reference if needed
        return mock_path
    mock_embedding_manager.generate_and_save_embedding.side_effect = mock_gen_save_capture
    mock_embedding_manager.embedding_model = MagicMock() # Ensure it has a model attribute
    
    # Patch the class globally for this fixture setup
    with patch('src.memory.hybrid_sqlite.EmbeddingManager', return_value=mock_embedding_manager):
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, 
                                      embedding_dir_str=emb_path_str,
                                      embedding_model=MagicMock()) # Pass dummy model

        docs_data = [
            {"id": "doc1", "text": "Apples are red fruit.", "tags": ["fruit", "food"], "source_agent": "AgentA"},
            {"id": "doc2", "text": "Oranges are orange citrus.", "tags": ["fruit", "citrus"], "source_agent": "AgentA"},
            {"id": "doc3", "text": "Bananas are yellow and long.", "tags": ["fruit", "berry"], "source_agent": "AgentB"},
            {"id": "doc4", "text": "Carrots are orange vegetables.", "tags": ["vegetable", "food"], "source_agent": "AgentB"},
            {"id": "doc5", "text": "Red cars go fast.", "tags": ["vehicle"], "source_agent": "AgentA"},
        ]
        
        doc_objects = []
        for data in docs_data:
            # Ensure timestamp and full MemoryDoc structure
            doc = MemoryDoc(
                id=data["id"],
                text=data["text"],
                timestamp=datetime.now(timezone.utc),
                source_agent=data["source_agent"],
                tags=data["tags"],
                metadata={"source": "fixture"}
            )
            adapter.add(doc)
            doc_objects.append(doc)
            
        yield adapter, mock_embedding_manager, doc_objects # Provide adapter, manager, and added docs to tests

    # Teardown: close connection
    if adapter:
        adapter.close()

# --- Semantic Query Tests --- 

MOCK_QUERY_EMBEDDING = np.array([0.1, 0.2, 0.3])
MOCK_DOC_EMBEDDINGS = {
    "doc1": np.array([0.11, 0.21, 0.31]), # High similarity
    "doc2": np.array([0.5, 0.5, 0.5]),   # Medium similarity
    "doc3": np.array([0.9, 0.9, 0.9]),   # Low similarity
    "doc4": np.array([0.8, 0.8, 0.8]),   # Low similarity
    "doc5": np.array([0.01, 0.02, 0.03]) # Very high similarity (to test ranking)
}

MOCK_SIMILARITIES = {
    ("query", "doc1"): 0.95,
    ("query", "doc2"): 0.60,
    ("query", "doc3"): 0.10,
    ("query", "doc4"): 0.15,
    ("query", "doc5"): 0.99
}

def test_semantic_query_basic(adapter_with_docs):
    """Test basic semantic query, returning results ranked by similarity."""
    adapter, manager, _ = adapter_with_docs
    query_text = "Tell me about red things"

    # --- Configure Mock EmbeddingManager for Query --- 
    # 1. Mock query embedding generation
    manager.generate_embedding.return_value = MOCK_QUERY_EMBEDDING

    # 2. Mock loading document embeddings (based on path stored during add)
    def mock_load(embedding_path):
        doc_id = Path(embedding_path).stem # Extract doc_id from filename
        return MOCK_DOC_EMBEDDINGS.get(doc_id) 
    manager.load_embedding.side_effect = mock_load

    # 3. Mock similarity calculation
    def mock_similarity(query_emb, doc_emb):
        # Find which doc_emb this is to look up the score
        # This is a bit simplified; assumes only MOCK_DOC_EMBEDDINGS are involved
        doc_id = None
        for d_id, d_emb in MOCK_DOC_EMBEDDINGS.items():
            if np.array_equal(doc_emb, d_emb):
                doc_id = d_id
                break
        if doc_id and np.array_equal(query_emb, MOCK_QUERY_EMBEDDING):
            return MOCK_SIMILARITIES.get(("query", doc_id), 0.0) # Default 0
        return 0.0 # Default similarity if lookup fails
    manager.calculate_similarity.side_effect = mock_similarity
    
    # --- Action --- 
    results = adapter.semantic_query(query_text, k=5)

    # --- Assertions --- 
    assert len(results) == 5, "Should return all 5 documents when k=5"
    
    # Check order based on MOCK_SIMILARITIES (highest first)
    expected_order = ["doc5", "doc1", "doc2", "doc4", "doc3"] # Based on scores: 0.99, 0.95, 0.60, 0.15, 0.10
    actual_order = [doc.id for doc in results]
    assert actual_order == expected_order, f"Results not in expected similarity order. Got: {actual_order}"
    
    # Check scores are included in results
    for doc in results:
        expected_score = MOCK_SIMILARITIES.get(("query", doc.id), 0.0)
        assert hasattr(doc, 'score'), f"Document {doc.id} should have a 'score' attribute"
        assert np.isclose(doc.score, expected_score), f"Doc {doc.id} score mismatch. Got {doc.score}, Expected {expected_score}"

def test_semantic_query_k_limit(adapter_with_docs):
    """Test that the 'k' parameter limits the number of results."""
    adapter, manager, _ = adapter_with_docs
    query_text = "Tell me about fruit"

    # Configure mock manager (same as basic test)
    manager.generate_embedding.return_value = MOCK_QUERY_EMBEDDING
    def mock_load(embedding_path):
        doc_id = Path(embedding_path).stem
        return MOCK_DOC_EMBEDDINGS.get(doc_id)
    manager.load_embedding.side_effect = mock_load
    def mock_similarity(query_emb, doc_emb):
        doc_id = None
        for d_id, d_emb in MOCK_DOC_EMBEDDINGS.items():
            if np.array_equal(doc_emb, d_emb):
                doc_id = d_id
                break
        if doc_id and np.array_equal(query_emb, MOCK_QUERY_EMBEDDING):
            return MOCK_SIMILARITIES.get(("query", doc_id), 0.0)
        return 0.0
    manager.calculate_similarity.side_effect = mock_similarity
    
    # --- Action --- 
    results = adapter.semantic_query(query_text, k=3)
    
    # --- Assertions --- 
    assert len(results) == 3, "Should return exactly 3 documents when k=3"
    # Check if they are the top 3 from the basic test
    expected_top_3 = ["doc5", "doc1", "doc2"]
    actual_order = [doc.id for doc in results]
    assert actual_order == expected_top_3, "Should return the top 3 most similar documents."

def test_semantic_query_filter_tags(adapter_with_docs):
    """Test filtering results by tags (single and multiple)."""
    adapter, manager, all_docs = adapter_with_docs
    query_text = "food items"
    
    # Configure mock manager (same as basic test)
    manager.generate_embedding.return_value = MOCK_QUERY_EMBEDDING
    manager.load_embedding.side_effect = lambda path: MOCK_DOC_EMBEDDINGS.get(Path(path).stem)
    manager.calculate_similarity.side_effect = lambda q_emb, d_emb: \
        next((score for (q, d_id), score in MOCK_SIMILARITIES.items() if q == "query" and np.array_equal(d_emb, MOCK_DOC_EMBEDDINGS[d_id])), 0.0) \
        if np.array_equal(q_emb, MOCK_QUERY_EMBEDDING) else 0.0

    # --- Test Single Tag Filter --- 
    results_fruit = adapter.semantic_query(query_text, k=5, filter_tags=["fruit"])
    # Expected: doc1, doc2, doc3 have 'fruit' tag
    expected_ids_fruit = {"doc1", "doc2", "doc3"}
    actual_ids_fruit = {doc.id for doc in results_fruit}
    assert actual_ids_fruit == expected_ids_fruit, "Filter by single tag 'fruit' failed."
    # Check order is still maintained within the filtered set
    expected_order_fruit = ["doc1", "doc2", "doc3"] # Scores: 0.95, 0.60, 0.10
    actual_order_fruit = [doc.id for doc in results_fruit]
    assert actual_order_fruit == expected_order_fruit
    
    # --- Test Multiple Tag Filter (AND logic) --- 
    results_food_veg = adapter.semantic_query(query_text, k=5, filter_tags=["food", "vegetable"])
    # Expected: Only doc4 has both 'food' and 'vegetable'
    expected_ids_food_veg = {"doc4"}
    actual_ids_food_veg = {doc.id for doc in results_food_veg}
    assert actual_ids_food_veg == expected_ids_food_veg, "Filter by multiple tags ['food', 'vegetable'] failed."
    assert len(results_food_veg) == 1
    assert results_food_veg[0].id == "doc4"

def test_semantic_query_filter_source_agent(adapter_with_docs):
    """Test filtering results by source_agent."""
    adapter, manager, _ = adapter_with_docs
    query_text = "Agent B documents"
    
    # Configure mock manager (same as basic test)
    manager.generate_embedding.return_value = MOCK_QUERY_EMBEDDING
    manager.load_embedding.side_effect = lambda path: MOCK_DOC_EMBEDDINGS.get(Path(path).stem)
    manager.calculate_similarity.side_effect = lambda q_emb, d_emb: \
        next((score for (q, d_id), score in MOCK_SIMILARITIES.items() if q == "query" and np.array_equal(d_emb, MOCK_DOC_EMBEDDINGS[d_id])), 0.0) \
        if np.array_equal(q_emb, MOCK_QUERY_EMBEDDING) else 0.0

    # --- Action --- 
    results = adapter.semantic_query(query_text, k=5, filter_source_agents=["AgentB"])
    
    # --- Assertions --- 
    # Expected: doc3, doc4 are from AgentB
    expected_ids = {"doc3", "doc4"}
    actual_ids = {doc.id for doc in results}
    assert actual_ids == expected_ids, "Filter by source_agent 'AgentB' failed."
    # Check order within filtered set
    expected_order = ["doc4", "doc3"] # Scores: 0.15, 0.10
    actual_order = [doc.id for doc in results]
    assert actual_order == expected_order

def test_semantic_query_filter_combined(adapter_with_docs):
    """Test filtering results by both tags and source_agent."""
    adapter, manager, _ = adapter_with_docs
    query_text = "Agent A fruit documents"
    
    # Configure mock manager (same as basic test)
    manager.generate_embedding.return_value = MOCK_QUERY_EMBEDDING
    manager.load_embedding.side_effect = lambda path: MOCK_DOC_EMBEDDINGS.get(Path(path).stem)
    manager.calculate_similarity.side_effect = lambda q_emb, d_emb: \
        next((score for (q, d_id), score in MOCK_SIMILARITIES.items() if q == "query" and np.array_equal(d_emb, MOCK_DOC_EMBEDDINGS[d_id])), 0.0) \
        if np.array_equal(q_emb, MOCK_QUERY_EMBEDDING) else 0.0

    # --- Action --- 
    results = adapter.semantic_query(query_text, k=5, filter_tags=["fruit"], filter_source_agents=["AgentA"])
    
    # --- Assertions --- 
    # Expected: doc1, doc2 are from AgentA AND have tag 'fruit'
    expected_ids = {"doc1", "doc2"}
    actual_ids = {doc.id for doc in results}
    assert actual_ids == expected_ids, "Combined filter for tag 'fruit' and source_agent 'AgentA' failed."
    # Check order
    expected_order = ["doc1", "doc2"] # Scores: 0.95, 0.60
    actual_order = [doc.id for doc in results]
    assert actual_order == expected_order

# TODO: Add tests for missing embeddings etc.
