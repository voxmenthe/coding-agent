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
