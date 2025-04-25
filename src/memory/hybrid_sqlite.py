import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import uuid
from typing import List, Optional, Dict, Tuple

from .adapter import MemoryAdapter, MemoryDoc
from .embedding_manager import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class HybridSQLiteAdapter(MemoryAdapter):
    """Implements the MemoryAdapter interface using SQLite for metadata/FTS
       and delegates embedding handling to EmbeddingManager.
    """

    def __init__(self, 
                 db_path_str: str = ".memory_db/memory.db", 
                 embedding_dir_str: str = ".memory_db/embeddings",
                 embedding_model: 'SentenceTransformer | None' = None): 
        self.db_path = Path(db_path_str)
        self.embedding_manager = EmbeddingManager(embedding_model, embedding_dir_str)
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db_uri = f"file:{self.db_path}?mode=rwc" 
        log.info(f"Initializing HybridSQLiteAdapter. DB: {db_uri}, Embeddings via Manager: {self.embedding_manager.embedding_dir}")
        try:
            self.conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row 
            log.info("SQLite connection established.")
            self._setup_schema()
        except sqlite3.Error as e:
            log.error(f"Error connecting to or setting up SQLite DB at {self.db_path}: {e}")
            raise

    def _setup_schema(self):
        """Creates SQLite tables including FTS and embedding path."""
        try:
            log.info("Setting up database schema...") 
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    uuid TEXT UNIQUE NOT NULL,
                    text_content TEXT NOT NULL,
                    embedding_path TEXT, 
                    timestamp TEXT, 
                    source_agent TEXT, 
                    tags_json TEXT, 
                    metadata_json TEXT 
                )
            """)
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    text_content,
                    tokenize = 'porter',
                    content = memories, 
                    content_rowid = 'id' 
                )
            """)
            # Add triggers for FTS synchronization (although often automatic with 'content=')
            # It's safer to include them explicitly for robustness.
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts (rowid, text_content) VALUES (new.id, new.text_content);
                END;
            """)
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = old.id;
                END;
            """)
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    UPDATE memories_fts SET text_content = new.text_content WHERE rowid = old.id;
                END;
            """)
            self.conn.commit()
            log.info("SQLite schema setup complete (FTS triggers added).")
        except sqlite3.Error as e:
            log.error(f"Error setting up SQLite schema: {e}")
            raise

    def add(self, doc: MemoryDoc) -> str:
        """Adds a document to the memory store, generates embedding, saves and stores in DB."""
        if not doc.id:
            doc.id = str(uuid.uuid4())
            
        embedding_abs_path = self.embedding_manager.generate_and_save_embedding(doc.text, doc.id)
        
        sql = """
            INSERT INTO memories (uuid, text_content, embedding_path, timestamp, source_agent, tags_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            doc.id,
            doc.text,
            str(embedding_abs_path) if embedding_abs_path else None, 
            doc.timestamp.isoformat() if doc.timestamp else datetime.now(timezone.utc).isoformat(), 
            doc.source_agent,
            json.dumps(doc.tags or []),
            json.dumps(doc.metadata or {})
        )
        
        log.debug(f"Adding metadata to DB for UUID: {doc.id}")
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            self.conn.commit()
            log.info(f"Successfully added document UUID: {doc.id} to DB.")
            return doc.id
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: memories.uuid" in str(e):
                log.error(f"Document with UUID {doc.id} already exists in DB.")
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
                log.debug("Cursor closed after add operation.") 

    def _row_to_memory_doc(self, row: sqlite3.Row) -> MemoryDoc:
        """Converts a database row to a MemoryDoc."""
        return MemoryDoc(
            id=row['uuid'],
            text=row['text_content'],
            embedding_path=row['embedding_path'],
            timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else None,
            source_agent=row['source_agent'],
            tags=json.loads(row['tags_json']) if row['tags_json'] else [],
            metadata=json.loads(row['metadata_json']) if row['metadata_json'] else {},
            score=row['score'] if 'score' in row.keys() else None
        )

    def query(self, query_text: str, k: int = 10,
              filter_tags: list[str] | None = None, 
              filter_source_agents: list[str] | None = None) -> list[MemoryDoc]:
        """Queries the memory store using FTS5, applying optional filters."""
        if not query_text:
            return []
        
        # Base query joining FTS and main table
        sql_base = """
            SELECT m.uuid, m.text_content, m.embedding_path, m.timestamp, 
                   m.source_agent, m.tags_json, m.metadata_json, fts.rank as score 
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.rowid 
        """
        
        # Sanitize query_text for FTS5 phrase search: escape double quotes and wrap in double quotes
        fts_query_text = '"' + query_text.replace('"', '""') + '"'

        where_clauses = ["fts.text_content MATCH ?"] # Query text placeholder
        params = [fts_query_text] # Use the sanitized query text
        
        # Add tag filtering (requires ALL specified tags to be present)
        if filter_tags:
            # This assumes tags_json stores a JSON array like '["tag1", "tag2"]'
            # We use JSON_EACH to check for the presence of each tag.
            # A Common Table Expression (CTE) might be cleaner for complex tag logic,
            # but for simple AND logic, multiple JOINs or subqueries work.
            # Using json_each and checking count:
            # Add placeholder for each tag to params
            tag_placeholders = ','.join(['?'] * len(filter_tags))
            where_clauses.append(f"""
                m.id IN (
                    SELECT jt.id
                    FROM json_each(m.tags_json) AS je, memories AS jt
                    WHERE jt.id = m.id AND je.value IN ({tag_placeholders})
                    GROUP BY jt.id
                    HAVING COUNT(DISTINCT je.value) = ?
                )
            """)
            params.extend(filter_tags)
            params.append(len(filter_tags))
            
        # Add source agent filtering
        if filter_source_agents:
            agent_placeholders = ','.join(['?'] * len(filter_source_agents))
            where_clauses.append(f"m.source_agent IN ({agent_placeholders})")
            params.extend(filter_source_agents)

        # Combine WHERE clauses
        sql = sql_base + " WHERE " + " AND ".join(where_clauses)

        # Add ordering and limit
        sql += " ORDER BY fts.rank LIMIT ?" 
        params.append(k)

        log.info(f"Executing FTS query: '{query_text[:50]}...', k={k}, tags={filter_tags}, agents={filter_source_agents}")
        log.debug(f"Full FTS SQL: {sql} PARAMS: {params}") 
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            results = [self._row_to_memory_doc(row) for row in rows]
            for doc in results:
                if doc.score is not None:
                    doc.score = 1.0 / (1.0 + doc.score) 
            log.info(f"FTS query returned {len(results)} results.")
            return results
        except Exception as e:
            log.error(f"Error during FTS query: {e}", exc_info=True)
            return []
        finally:
            if cursor:
                cursor.close()
                log.debug("Cursor closed after FTS query.") 

    def semantic_query(self, query_text: str, k: int = 10,
                       filter_tags: list[str] | None = None,
                       filter_source_agents: list[str] | None = None) -> list[MemoryDoc]:
        """Performs semantic search using embeddings and cosine similarity via EmbeddingManager."""
        if not self.embedding_manager or not self.embedding_manager.embedding_model:
            log.error("Cannot perform semantic query: embedding model not available.")
            return []
        if not query_text:
            return []
            
        log.info(f"Starting semantic query: '{query_text[:50]}...', k={k}")

        query_embedding = self.embedding_manager.generate_embedding(query_text)
        if query_embedding is None:
            log.error("Failed to generate embedding for query text.")
            return []
        log.debug(f"Generated query embedding, shape: {query_embedding.shape}")

        sql = "SELECT uuid, text_content, embedding_path, timestamp, source_agent, tags_json, metadata_json FROM memories WHERE embedding_path IS NOT NULL"
        params = []

        candidates: list[MemoryDoc] = []
        cursor = None
        try:
            cursor = self.conn.cursor()
            log.debug(f"Fetching candidates for semantic search. SQL: {sql}")
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            candidates = [self._row_to_memory_doc(row) for row in rows]
            log.debug(f"Fetched {len(candidates)} candidates with embeddings from DB.")
        except Exception as e:
            log.error(f"Error fetching candidates for semantic search: {e}", exc_info=True)
            return []
        finally:
            if cursor:
                cursor.close()
                log.debug("Cursor closed after fetching semantic candidates.")

        results_with_scores: list[tuple[MemoryDoc, float]] = []
        for candidate_doc in candidates:
            passes_filter = True
            if filter_tags and not set(filter_tags).issubset(set(candidate_doc.tags or [])):
                passes_filter = False
            if filter_source_agents and candidate_doc.source_agent not in filter_source_agents:
                passes_filter = False

            if passes_filter and candidate_doc.embedding_path:
                doc_embedding = self.embedding_manager.load_embedding(candidate_doc.embedding_path)
                if doc_embedding is not None:
                    similarity = self.embedding_manager.calculate_similarity(query_embedding, doc_embedding)
                    results_with_scores.append((candidate_doc, similarity))
                else:
                    log.warning(f"Could not load embedding for doc {candidate_doc.id} from path {candidate_doc.embedding_path}")

        results_with_scores.sort(key=lambda item: item[1], reverse=True)
        top_k_results = results_with_scores[:k]

        final_results = []
        for doc, score in top_k_results:
            doc.score = score
            final_results.append(doc)

        log.info(f"Semantic query returned {len(final_results)} results.")
        return final_results

    def hybrid_query(
        self,
        query_text: str,
        k: int = 10,
        filter_tags: Optional[List[str]] = None,
        filter_source_agents: Optional[List[str]] = None,
        rrf_k: int = 60,  # RRF constant, common default is 60
    ) -> List[Tuple[MemoryDoc, float]]:
        """Perform a hybrid search combining FTS and semantic search using RRF.

        Args:
            query_text: The text to search for.
            k: The final number of results to return.
            filter_tags: Optional list of tags to filter results by.
            filter_source_agents: Optional list of source agents to filter results by.
            rrf_k: The constant used in the RRF calculation (default: 60).

        Returns:
            A list of tuples, each containing a MemoryDoc and its RRF score,
            sorted by score in descending order.
        """
        # 1. Perform FTS query
        # We fetch more results initially to give RRF a good pool to work with
        fts_results = self.query(
            query_text,
            k=k * 5,  # Fetch more for ranking
            filter_tags=filter_tags,
            filter_source_agents=filter_source_agents,
        )

        # 2. Perform Semantic query
        semantic_results = self.semantic_query(
            query_text,
            k=k * 5,  # Fetch more for ranking
            filter_tags=filter_tags,
            filter_source_agents=filter_source_agents,
        )

        # 3. Combine results using Reciprocal Rank Fusion (RRF)
        ranked_results: Dict[str, float] = {}

        # Process FTS results
        for rank, doc in enumerate(fts_results):
            doc_id = str(doc.id)
            if doc_id not in ranked_results:
                ranked_results[doc_id] = 0.0
            ranked_results[doc_id] += 1.0 / (rrf_k + rank)

        # Process Semantic results
        for rank, doc in enumerate(semantic_results):
            doc_id = str(doc.id)
            if doc_id not in ranked_results:
                ranked_results[doc_id] = 0.0
            ranked_results[doc_id] += 1.0 / (rrf_k + rank)

        # 4. Sort by RRF score
        sorted_doc_ids = sorted(ranked_results.keys(), key=lambda doc_id: ranked_results[doc_id], reverse=True)

        # 5. Retrieve full MemoryDoc objects for the top K results
        final_results: List[Tuple[MemoryDoc, float]] = []
        doc_map = {str(doc.id): doc for doc in fts_results + semantic_results}

        for doc_id in sorted_doc_ids[:k]:
            if doc_id in doc_map:
                # Add the doc and its score, ensuring no duplicates if a doc was only in one list initially
                if not any(res[0].id == doc_map[doc_id].id for res in final_results):
                     final_results.append((doc_map[doc_id], ranked_results[doc_id]))
            else:
                # This case might happen if filters were applied differently or
                # if a doc appeared very low in both lists and wasn't fetched.
                # Fetch the doc explicitly if needed, though less likely with k*5 fetch.
                # For simplicity here, we rely on the combined list. If a doc_id isn't
                # in doc_map, it means it wasn't in the top k*5 of either search.
                pass # Or log a warning

        # Ensure we don't exceed k due to potential rounding/tie issues (unlikely with float scores)
        return final_results[:k]

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
        if hasattr(self, 'conn') and self.conn:
            self.close()

# Example usage (for testing)
if __name__ == '__main__':
    import uuid
    import json 

    print("Testing HybridSQLiteAdapter...")
    adapter = HybridSQLiteAdapter(db_path_str=":memory:") 

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
        assert doc['uuid'] == doc2_id 
    assert len(results_orange) == 1

    print("\nTesting complete.")
    adapter.close()
