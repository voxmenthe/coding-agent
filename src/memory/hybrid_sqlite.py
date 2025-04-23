import sqlite3
import json
import logging
from pathlib import Path
from .adapter import MemoryAdapter, MemoryDoc
from datetime import datetime, timezone
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class HybridSQLiteAdapter(MemoryAdapter):
    """Implements the MemoryAdapter interface using SQLite for metadata/FTS
       and separate files for embeddings.
    """

    def __init__(self, db_path_str: str = ".memory_db/memory.db", embedding_dir_str: str = ".memory_db/embeddings"):
        self.db_path = Path(db_path_str)
        self.embedding_dir = Path(embedding_dir_str) # Keep for now, though not used in simplified schema
        
        # Ensure parent directory for DB exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure embedding directory exists
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

        db_uri = f"file:{self.db_path}?mode=rwc" 
        log.info(f"DIAG: Connecting to DB: {db_uri}") # Print DB path
        try:
            # check_same_thread=False needed for potential multi-agent access? Review implications.
            self.conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row # Access columns by name
            log.info("SQLite connection established.")
            self._setup_schema()
        except sqlite3.Error as e:
            log.error(f"Error connecting to or setting up SQLite DB at {self.db_path}: {e}")
            raise

    def _setup_schema(self):
        """Creates simplified SQLite tables with manual FTS triggers and Porter tokenizer."""
        try:
            log.info("DIAG: Setting up simplified schema with manual FTS triggers + Porter tokenizer...")
            # Simplified main table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT UNIQUE NOT NULL,
                    text_content TEXT NOT NULL
                )
            """)
            log.info("DIAG: 'memories' table created (if not exists).")
            
            # Simplified FTS table - relying on triggers, explicit Porter tokenizer, INDEXED column
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    text_content,  -- Removed UNINDEXED
                    tokenize = 'porter'
                )
            """)
            log.info("DIAG: 'memories_fts' virtual table created (trigger-based sync, tokenizer=porter, indexed column).")

            # Triggers to keep FTS table synced with memories table
            # AFTER INSERT
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts (rowid, text_content) VALUES (new.id, new.text_content);
                END;
            """)
            # AFTER DELETE
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = old.id;
                END;
            """)
            # AFTER UPDATE
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    UPDATE memories_fts SET text_content = new.text_content WHERE rowid = old.id;
                END;
            """)
            log.info("DIAG: FTS synchronization triggers created (if not exist).")

            self.conn.commit()
            log.info("SQLite simplified schema setup complete (trigger-based FTS sync, tokenizer=porter, indexed column).")
        except sqlite3.Error as e:
            log.error(f"Error setting up simplified SQLite schema with triggers & tokenizer: {e}")
            raise

    def add(self, doc: MemoryDoc) -> str:
        """Adds a document to the memory store (simplified)."""
        if not doc.id:
            doc.id = str(uuid.uuid4())
        
        log.info(f"DIAG (add): Attempting to add UUID: {doc.id}, Text: '{doc.text[:30]}...'")
        cursor = None
        try:
            cursor = self.conn.cursor()
            log.info("DIAG (add): Cursor created.")
            
            print(f"DIAG (add): BEFORE INSERT for UUID {doc.id}") # DIAG PRINT
            cursor.execute("""
                INSERT INTO memories (uuid, text_content)
                VALUES (?, ?)
            """, (
                doc.id, 
                doc.text,
            ))
            print(f"DIAG (add): AFTER INSERT for UUID {doc.id}, lastrowid: {cursor.lastrowid}") # DIAG PRINT
            
            print(f"DIAG (add): BEFORE COMMIT for UUID {doc.id}") # DIAG PRINT
            self.conn.commit()
            print(f"DIAG (add): AFTER COMMIT for UUID {doc.id}") # DIAG PRINT

            log.info(f"Successfully added document UUID: {doc.id}")
            return doc.id 
        except sqlite3.IntegrityError as e:
            # Handle potential unique constraint violation on uuid
            if "UNIQUE constraint failed: memories.uuid" in str(e):
                log.error(f"Document with UUID {doc.id} already exists.")
                # Optionally, implement update logic or raise a specific exception
                raise ValueError(f"Document with UUID {doc.id} already exists.") from e
            else:
                log.error(f"SQLite integrity error adding document: {e}")
                raise
        except Exception as e:
            log.error(f"Unexpected error adding document: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
                log.info("DIAG (add): Cursor closed.")

    def query(self, query_text: str, k: int = 5, **kwargs) -> list[dict]:
        """Queries the memory store using FTS5 (simplified, returns dicts)."""
        if not query_text:
            return []
        
        sql = """
            SELECT m.uuid, m.text_content -- Select only essential fields
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.rowid 
            WHERE fts.text_content MATCH ?
            ORDER BY fts.rank 
            LIMIT ?
        """
        params = (query_text, k)
        log.info(f"DIAG (query): Executing FTS query: '{sql}' with params: {params}")
        print(f"DIAG (query): Executing FTS query for '{query_text}' (k={k})") # DIAG PRINT
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            raw_results = cursor.fetchall() # Fetch all results as Row objects
            print(f"DIAG (query): Raw results count: {len(raw_results)}") # DIAG PRINT
            
            # Convert Row objects to simple dictionaries
            results = [{'uuid': row['uuid'], 'text_content': row['text_content']} for row in raw_results]
            print(f"DIAG (query): Returning {len(results)} simplified dict results.") # DIAG PRINT
            return results
        except Exception as e:
            log.error(f"Error querying memory: {e}")
            print(f"DIAG (query): ERROR during query: {e}") # DIAG PRINT
            return []
        finally:
             if cursor:
                cursor.close()
                log.info("DIAG (query): Cursor closed.")

    def close(self):
        """Closes the SQLite database connection."""
        if self.conn:
            try:
                self.conn.close()
                log.info(f"Closed SQLite DB connection: {self.db_path}")
                self.conn = None
            except sqlite3.Error as e:
                log.error(f"Error closing SQLite DB connection: {e}")

    def __del__(self):
        """Ensure connection is closed when object is garbage collected."""
        # This might be redundant if close() is called explicitly,
        # but provides a fallback.
        if hasattr(self, 'conn') and self.conn:
            self.close()

# Example usage (for testing)
if __name__ == '__main__':
    import uuid
    import json # Make sure json is imported

    print("Testing HybridSQLiteAdapter...")
    adapter = HybridSQLiteAdapter(db_path_str=":memory:") # Use in-memory DB for test

    # Test Add
    doc1_id = str(uuid.uuid4())
    doc1 = MemoryDoc(
        id=doc1_id,
        text="This is the first test document about apples.",
        tags=["test", "fruit"],
        source_agent="test_script"
    )
    adapter.add(doc1)

    doc2_id = str(uuid.uuid4())
    doc2 = MemoryDoc(
        id=doc2_id,
        text="A second document, this one mentions oranges and apples.",
        tags=["test", "citrus", "fruit"],
        source_agent="test_script"
    )
    adapter.add(doc2)

    # Test Query
    print("\nQuerying for 'apples':")
    results_apple = adapter.query("apples", k=5)
    for doc in results_apple:
        print(f" - ID: {doc['uuid']}, Text: {doc['text_content'][:50]}...")
        assert "apples" in doc['text_content'].lower()

    print("\nQuerying for 'oranges':")
    results_orange = adapter.query("oranges", k=5)
    for doc in results_orange:
        print(f" - ID: {doc['uuid']}, Text: {doc['text_content'][:50]}...")
        assert "oranges" in doc['text_content'].lower()
        assert doc['uuid'] == doc2_id # Should only find doc2
    assert len(results_orange) == 1

    print("\nTesting complete.")
    adapter.close()
