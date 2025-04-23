import pytest
import sqlite3
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone

from src.memory.adapter import MemoryDoc
from src.memory.hybrid_sqlite import HybridSQLiteAdapter

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

def test_initialization(db_paths):
    """Test if adapter initializes correctly and creates files/dirs."""
    db_path_str, emb_path_str = db_paths
    db_path = Path(db_path_str)
    emb_dir = Path(emb_path_str)
    adapter = None # Ensure adapter is defined for finally block
    try:
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, embedding_dir_str=emb_path_str)

        assert db_path.parent.exists()
        assert emb_dir.exists()
        assert db_path.exists()
        assert str(adapter.db_path) == db_path_str
        assert str(adapter.embedding_dir) == emb_path_str
        assert adapter.conn is not None
    finally:
        if adapter:
            adapter.close()

def test_schema_creation(db_paths):
    """Verify that the necessary tables are created."""
    db_path_str, emb_path_str = db_paths
    adapter = None
    try:
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, embedding_dir_str=emb_path_str)
        cursor = adapter.conn.cursor()
        
        # Check if 'memories' table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories';")
        assert cursor.fetchone() is not None, "'memories' table not found."

        # Check columns in 'memories' table
        cursor.execute("PRAGMA table_info(memories);")
        columns = {row['name']: row['type'] for row in cursor.fetchall()} # Use column names
        print(f"DIAG (test_schema): Found columns: {columns}") # DIAG PRINT
        assert len(columns) == 3, f"Expected 3 columns, found {len(columns)}"
        assert 'id' in columns and columns['id'] == 'INTEGER'
        assert 'uuid' in columns and columns['uuid'] == 'TEXT'
        assert 'text_content' in columns and columns['text_content'] == 'TEXT'

        # Check if 'memories_fts' table exists and is FTS5
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts';")
        assert cursor.fetchone() is not None, "'memories_fts' table not found."
        cursor.execute("SELECT sql FROM sqlite_master WHERE name='memories_fts';")
        fts_sql = cursor.fetchone()['sql']
        print(f"DIAG (test_schema): FTS SQL: {fts_sql}") # DIAG PRINT
        assert 'USING FTS5' in fts_sql.upper() # Check it's an FTS5 table

    finally:
        if adapter:
            adapter.close()

def test_add_memory_basic(db_paths):
    """Test adding a simple MemoryDoc (simplified schema)."""
    db_path_str, emb_path_str = db_paths
    adapter = None
    print(f"\n--- Starting test_add_memory_basic ---") # DIAG PRINT
    try:
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, embedding_dir_str=emb_path_str)
        doc_id = str(uuid.uuid4())
        test_doc = MemoryDoc(
            id=doc_id,
            text="This is a test document for FTS.",
            # Removed other fields for simplification
        )
        print(f"DIAG (test): Calling adapter.add() for UUID {doc_id}") # DIAG PRINT
        added_id = adapter.add(test_doc)
        assert added_id == doc_id
        print(f"DIAG (test): adapter.add() returned {added_id}") # DIAG PRINT

        # Verify data in 'memories' table
        cursor = adapter.conn.cursor()
        print(f"DIAG (test): Querying 'memories' table for UUID {doc_id}") # DIAG PRINT
        cursor.execute("SELECT id, uuid, text_content FROM memories WHERE uuid = ?", (doc_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row['uuid'] == doc_id
        assert row['text_content'] == test_doc.text
        integer_id = row['id'] # Get the auto-generated integer ID
        assert isinstance(integer_id, int)
        print(f"DIAG (test): Found row in 'memories', integer_id: {integer_id}") # DIAG PRINT

        # Verify data in 'memories_fts' using the integer id
        print(f"DIAG (test): Querying 'memories_fts' table for rowid {integer_id}") # DIAG PRINT
        cursor.execute("SELECT rowid FROM memories_fts WHERE rowid = ?;", (integer_id,))
        fts_row = cursor.fetchone()
        assert fts_row is not None, f"FTS table should contain the integer rowid {integer_id} after insert."
        print(f"DIAG (test): Found rowid {fts_row['rowid']} in 'memories_fts'.") # DIAG PRINT
        
        # Now check the specific text match via FTS query
        print(f"DIAG (test): Querying 'memories_fts' via MATCH 'test' for rowid {integer_id}") # DIAG PRINT
        cursor.execute("SELECT rowid FROM memories_fts WHERE text_content MATCH ? AND rowid = ?;", ("test", integer_id))
        match_row = cursor.fetchone()
        assert match_row is not None, f"FTS query for 'test' should match the inserted rowid {integer_id}."
        print(f"DIAG (test): Found rowid {match_row['rowid']} via MATCH 'test'.") # DIAG PRINT

    finally:
        if adapter:
            adapter.close()
        print(f"--- Finished test_add_memory_basic ---\n") # DIAG PRINT


def test_query_memory_fts(db_paths):
    """Test querying memory using FTS (simplified schema)."""
    db_path_str, emb_path_str = db_paths
    adapter = None
    print(f"\n--- Starting test_query_memory_fts ---") # DIAG PRINT
    try:
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, embedding_dir_str=emb_path_str)
        doc1_id = str(uuid.uuid4())
        doc1 = MemoryDoc(id=doc1_id, text="The quick brown fox jumps over the lazy dog.")
        doc2_id = str(uuid.uuid4())
        doc2 = MemoryDoc(id=doc2_id, text="Another test document about foxes.")
        doc3_id = str(uuid.uuid4())
        doc3 = MemoryDoc(id=doc3_id, text="This one is about dogs only.")

        print(f"DIAG (test): Adding doc1 (fox dog) UUID {doc1_id}") # DIAG PRINT
        adapter.add(doc1)
        print(f"DIAG (test): Adding doc2 (foxes) UUID {doc2_id}") # DIAG PRINT
        adapter.add(doc2)
        print(f"DIAG (test): Adding doc3 (dogs) UUID {doc3_id}") # DIAG PRINT
        adapter.add(doc3)

        # Query for "fox" - Expecting dict results
        print(f"DIAG (test): Calling adapter.query('fox')") # DIAG PRINT
        results_fox = adapter.query("fox", k=5)
        print(f"DIAG (test): adapter.query('fox') returned {len(results_fox)} results: {results_fox}") # DIAG PRINT
        
        assert len(results_fox) == 2, f"Expected 2 results for 'fox', got {len(results_fox)}"
        result_uuids_fox = {r['uuid'] for r in results_fox}
        assert doc1_id in result_uuids_fox
        assert doc2_id in result_uuids_fox
        
        # Query for "lazy"
        print(f"DIAG (test): Calling adapter.query('lazy')") # DIAG PRINT
        results_lazy = adapter.query("lazy", k=5)
        print(f"DIAG (test): adapter.query('lazy') returned {len(results_lazy)} results: {results_lazy}") # DIAG PRINT
        assert len(results_lazy) == 1
        assert results_lazy[0]['uuid'] == doc1_id

        # Query for "missing"
        print(f"DIAG (test): Calling adapter.query('missing')") # DIAG PRINT
        results_missing = adapter.query("missing", k=5)
        print(f"DIAG (test): adapter.query('missing') returned {len(results_missing)} results: {results_missing}") # DIAG PRINT
        assert len(results_missing) == 0

    finally:
        if adapter:
            adapter.close()
        print(f"--- Finished test_query_memory_fts ---\n") # DIAG PRINT


def test_query_empty_db(db_paths):
    """Test querying an empty database."""
    db_path_str, emb_path_str = db_paths
    adapter = None
    try:
        adapter = HybridSQLiteAdapter(db_path_str=db_path_str, embedding_dir_str=emb_path_str)
        results = adapter.query("anything", k=5)
        assert len(results) == 0
    finally:
        if adapter:
            adapter.close()
