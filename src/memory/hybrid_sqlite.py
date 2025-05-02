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
# logging.basicConfig(level=logging.INFO) # Keep root config
log = logging.getLogger('HybridSQLiteAdapter') # Use specific logger name
log.setLevel(logging.DEBUG) # Set level for this logger

class HybridSQLiteAdapter(MemoryAdapter):
    """Implements the MemoryAdapter interface using SQLite for metadata/FTS
       and delegates embedding handling to EmbeddingManager.
    """

    def __init__(self, 
                 db_path_str: str = ".memory_db/memory.db", 
                 embedding_dir_str: str = ".memory_db/embeddings",
                 embedding_model: 'SentenceTransformer | None' = None): 
        log.debug(f"Initializing HybridSQLiteAdapter: DB='{db_path_str}', Embeddings='{embedding_dir_str}', Model provided={embedding_model is not None}")
        self.db_path = Path(db_path_str)
        self.embedding_manager = EmbeddingManager(embedding_model, embedding_dir_str)
        
        log.debug(f"Ensuring parent directory exists: {self.db_path.parent}")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db_uri = f"file:{self.db_path}?mode=rwc" 
        log.info(f"Initializing HybridSQLiteAdapter. DB URI: {db_uri}, Embeddings via Manager: {self.embedding_manager.embedding_dir}")
        self.conn = None # Initialize conn to None
        try:
            log.debug(f"Attempting to connect to SQLite DB: {db_uri}")
            self.conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row 
            log.info("SQLite connection established successfully.")
            self._setup_schema()
        except sqlite3.Error as e:
            log.error(f"Error connecting to or setting up SQLite DB at {self.db_path}: {e}", exc_info=True)
            if self.conn: # Attempt to close if partially opened
                try:
                    self.conn.close()
                    log.debug("Closed potentially partial connection after error.")
                except Exception as close_err:
                    log.error(f"Error closing connection after initial error: {close_err}")
            raise

    def _setup_schema(self):
        """Sets up the necessary SQLite tables and FTS virtual table."""
        # Use try-finally to ensure cursor is closed
        cursor = None
        try:
            cursor = self.conn.cursor()
            # Main table for metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    uuid TEXT UNIQUE NOT NULL,
                    text_content TEXT,
                    embedding_path TEXT, -- Path to the .npy file 
                    timestamp TEXT NOT NULL,
                    source_agent TEXT,
                    tags_json TEXT,     -- JSON array of strings
                    metadata_json TEXT  -- JSON object for other metadata
                );
            """)
            # Index on UUID for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_uuid ON memories (uuid);");
            # Index on timestamp
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp);");
            
            # FTS5 virtual table for text search
            # Link to the main table using content_rowid='id'
            # Revert back to 'porter unicode61' tokenizer
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    text_content, 
                    content='memories', 
                    content_rowid='id',
                    tokenize='porter unicode61' 
                );
            """)
            # Triggers to keep FTS table synchronized
            # AFTER INSERT
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts (rowid, text_content) VALUES (new.id, new.text_content);
                END;
            """)
            # AFTER DELETE
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts (content, rowid, text_content) VALUES ('delete', old.id, old.text_content);
                END;
            """)
            # AFTER UPDATE
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts (content, rowid, text_content) VALUES ('update', new.id, new.text_content);
                END;
            """)
            self.conn.commit()
            log.info("SQLite schema setup complete (FTS triggers added).")
        except sqlite3.Error as e:
            log.error(f"SQLite error during schema setup: {e}", exc_info=True)
            # Optional: rollback changes if needed, though table creation is often idempotent
            # if self.conn: self.conn.rollback()
            raise # Re-raise the exception to indicate failure
        finally:
            if cursor:
                cursor.close()
                log.debug("Schema setup cursor closed.")

    def add(self, doc: MemoryDoc) -> str | None: # Return None on failure
        """Adds a document to the memory store, generates embedding, saves and stores in DB."""
        log.debug(f"Entering add method for doc (ID provided: {doc.id is not None}, Text length: {len(doc.text)})")
        if not self.conn:
            log.error("Cannot add document: SQLite connection is not established.")
            return None

        if not doc.id:
            doc_uuid = str(uuid.uuid4())
            log.debug(f"No document ID provided, generated new UUID: {doc_uuid}")
            doc.id = doc_uuid # Assign back to doc object as well
        else:
            doc_uuid = doc.id
            log.debug(f"Using provided document ID (UUID): {doc_uuid}")
        
        log.debug(f"Calling embedding_manager.generate_and_save_embedding for UUID: {doc_uuid}")
        embedding_abs_path = self.embedding_manager.generate_and_save_embedding(doc.text, doc_uuid)
        if embedding_abs_path:
             log.debug(f"Embedding generated and saved to: {embedding_abs_path}")
        else:
             log.warning(f"Embedding generation/saving failed or was disabled for UUID: {doc_uuid}")
        
        sql = """
            INSERT INTO memories (uuid, text_content, embedding_path, timestamp, source_agent, tags_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        current_timestamp = doc.timestamp or datetime.now(timezone.utc)
        params = (
            doc_uuid,
            doc.text,
            str(embedding_abs_path) if embedding_abs_path else None, 
            current_timestamp.isoformat(), 
            doc.source_agent,
            json.dumps(doc.tags or []),
            json.dumps(doc.metadata or {})
        )
        
        log.debug(f"Preparing to insert metadata into DB for UUID: {doc_uuid}")
        log.debug(f"SQL: {sql}")
        log.debug(f"PARAMS: UUID={params[0]}, Text Len={len(params[1])}, Embed Path={params[2]}, Timestamp={params[3]}, Agent={params[4]}, Tags={params[5]}, Meta={params[6]}")
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            log.debug("Executing INSERT statement...")
            cursor.execute(sql, params)
            log.debug(f"INSERT executed. Row count: {cursor.rowcount}. Committing...")
            self.conn.commit()
            log.info(f"Successfully added document UUID: {doc_uuid} to DB.")
            return doc_uuid
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: memories.uuid" in str(e):
                log.error(f"Document with UUID {doc_uuid} already exists in DB.")
                # Consider if this should return the ID or raise a different error
                # For now, returning None to indicate add failure due to duplication
                return None 
            else:
                log.error(f"SQLite integrity error adding document UUID {doc_uuid}: {e}", exc_info=True)
                return None
        except Exception as e:
            log.error(f"Unexpected error adding document UUID {doc_uuid}: {e}", exc_info=True)
            return None # Return None on generic error
        finally:
            if cursor:
                cursor.close()
                log.debug(f"Cursor closed after add operation for UUID: {doc_uuid}.") 

    def _row_to_memory_doc(self, row: sqlite3.Row) -> MemoryDoc:
        """Converts a database row to a MemoryDoc."""
        # Add logging here if conversion issues arise
        # log.debug(f"Converting row to MemoryDoc: {dict(row)}")
        try:
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
        except Exception as e:
            log.error(f"Error converting row to MemoryDoc: {e}. Row data: {dict(row)}", exc_info=True)
            raise # Re-raise after logging

    def query(self, query_text: str, k: int = 10,
              filter_tags: list[str] | None = None, 
              filter_source_agents: list[str] | None = None) -> list[MemoryDoc]:
        """Queries the memory store using FTS5, applying optional filters."""
        log.debug(f"Entering query method: query='{query_text[:50]}...', k={k}, tags={filter_tags}, agents={filter_source_agents}")
        if not self.conn:
            log.error("Cannot query: SQLite connection is not established.")
            return []
            
        if not query_text:
            log.warning("Query text is empty. Returning empty list.")
            return []
        
        # Base query joining FTS and main table
        sql_base = """
            SELECT m.uuid, m.text_content, m.embedding_path, m.timestamp, 
                   m.source_agent, m.tags_json, m.metadata_json, fts.rank as score 
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.rowid 
        """
        
        # Sanitize query_text for FTS5: escape double quotes
        # Don't wrap in quotes here to allow FTS parsing
        fts_query_text = query_text.replace('"', '""') 
        
        # Append wildcard for prefix matching - helps with stemming variations
        if not fts_query_text.endswith('*'):
            fts_query_text += '*'
            
        log.debug(f"Sanitized FTS query text for MATCH: {fts_query_text}")

        where_clauses = ["fts.text_content MATCH ?"] # Query text placeholder
        params: list[str | int] = [fts_query_text] # Use the sanitized & wildcarded query text
        
        # Add tag filtering (requires ALL specified tags to be present)
        if filter_tags:
            log.debug(f"Adding tag filter: {filter_tags}")
            tag_placeholders = ','.join(['?'] * len(filter_tags))
            # Simplified check using json_extract and LIKE for individual tags
            # This is less efficient than json_each but simpler syntax for AND logic
            for tag in filter_tags:
                 # Check if tags_json contains the specific tag string
                 # Note: This is a basic substring check, assumes tags are simple strings
                 # and JSON format is consistent. Might need refinement for complex tags.
                 where_clauses.append("json_valid(m.tags_json) AND instr(m.tags_json, ?)")
                 params.append(f'"{tag}"') # Add quotes for JSON string comparison
            log.debug(f"WHERE clauses after tags: {where_clauses}, PARAMS: {params}")
            
        # Add source agent filtering
        if filter_source_agents:
            log.debug(f"Adding source agent filter: {filter_source_agents}")
            agent_placeholders = ','.join(['?'] * len(filter_source_agents))
            where_clauses.append(f"m.source_agent IN ({agent_placeholders})")
            params.extend(filter_source_agents)
            log.debug(f"WHERE clauses after agents: {where_clauses}, PARAMS: {params}")

        # Combine WHERE clauses
        sql = sql_base + " WHERE " + " AND ".join(where_clauses)

        # Add ordering and limit
        sql += " ORDER BY fts.rank LIMIT ?" 
        params.append(k)

        log.info(f"Executing FTS query: '{query_text[:50]}...', k={k}, tags={filter_tags}, agents={filter_source_agents}")
        log.debug(f"Full FTS SQL: {sql}")
        log.debug(f"PARAMS: {params}") 
        
        results = []
        cursor = None
        try:
            cursor = self.conn.cursor()
            log.debug("Executing FTS SELECT statement...")
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            log.info(f"FTS query executed. Fetched {len(rows)} rows.")
            for row in rows:
                try:
                    results.append(self._row_to_memory_doc(row))
                except Exception as convert_err:
                    log.error(f"Skipping row due to conversion error: {convert_err}. Row: {dict(row)}")
            log.debug(f"Converted {len(results)} rows to MemoryDoc objects.")
            return results
        except sqlite3.Error as e:
            log.error(f"SQLite error during FTS query execution: {e}", exc_info=True)
            return []
        except Exception as e:
             log.error(f"Unexpected error during FTS query: {e}", exc_info=True)
             return []
        finally:
            if cursor:
                cursor.close()
                log.debug("Cursor closed after FTS query operation.")

    def semantic_query(self, query_text: str, k: int = 10,
                       filter_tags: list[str] | None = None,
                       filter_source_agents: list[str] | None = None) -> list[MemoryDoc]:
        """Performs semantic search using embeddings if available.
           Applies pre-filtering based on tags/agents if provided.
        """
        log.debug(f"Entering semantic_query method: query='{query_text[:50]}...', k={k}, tags={filter_tags}, agents={filter_source_agents}")
        if not self.conn:
            log.error("Cannot semantic query: SQLite connection is not established.")
            return []
            
        if not self.embedding_manager.embedding_model:
            log.error("Cannot perform semantic query: embedding model not available.")
            return []

        # 1. Pre-filter potential candidates from DB based on tags/agents (if provided)
        candidate_uuids: Optional[List[str]] = None
        if filter_tags or filter_source_agents:
            log.debug("Performing pre-filtering based on tags/agents for semantic query...")
            sql_filter = "SELECT uuid FROM memories" 
            where_clauses = []
            params: list[str] = []
            if filter_tags:
                # Using basic instr check again for simplicity
                for tag in filter_tags:
                    where_clauses.append("json_valid(tags_json) AND instr(tags_json, ?)")
                    params.append(f'"{tag}"')
            if filter_source_agents:
                agent_placeholders = ','.join(['?'] * len(filter_source_agents))
                where_clauses.append(f"source_agent IN ({agent_placeholders})")
                params.extend(filter_source_agents)
            
            if where_clauses:
                sql_filter += " WHERE " + " AND ".join(where_clauses)
                log.debug(f"Pre-filter SQL: {sql_filter}, PARAMS: {params}")
                cursor = None
                try:
                    cursor = self.conn.cursor()
                    cursor.execute(sql_filter, params)
                    candidate_uuids = [row['uuid'] for row in cursor.fetchall()]
                    log.info(f"Pre-filtering yielded {len(candidate_uuids)} candidate UUIDs.")
                    if not candidate_uuids:
                        log.info("No candidates found after pre-filtering. Semantic search will yield no results.")
                        return [] # No need to proceed if no candidates
                except sqlite3.Error as e:
                    log.error(f"SQLite error during pre-filtering for semantic query: {e}", exc_info=True)
                    return [] # Return empty on error
                finally:
                    if cursor:
                        cursor.close()
                        log.debug("Cursor closed after pre-filtering.")
            else:
                 log.debug("No tags or agents provided for pre-filtering.")
        else:
            log.debug("No pre-filtering requested for semantic query.")

        # 2. Perform semantic search using EmbeddingManager
        log.debug(f"Calling embedding_manager.search for query='{query_text[:50]}...', k={k}, filter_ids={candidate_uuids is not None}")
        try:
            similar_docs_info = self.embedding_manager.search(query_text, k, filter_ids=candidate_uuids)
            log.info(f"EmbeddingManager search returned {len(similar_docs_info)} similar docs.")
        except Exception as search_err:
            log.error(f"Error during EmbeddingManager search: {search_err}", exc_info=True)
            return []

        if not similar_docs_info:
            return []

        # 3. Retrieve full MemoryDoc objects for the top results from DB
        top_k_uuids = [uuid_ for uuid_, score in similar_docs_info]
        scores_map = {uuid_: score for uuid_, score in similar_docs_info}
        log.debug(f"Retrieving full MemoryDocs for top {len(top_k_uuids)} UUIDs: {top_k_uuids}")

        if not top_k_uuids:
             return []
             
        sql_retrieve = f"SELECT * FROM memories WHERE uuid IN ({','.join(['?']*len(top_k_uuids))})"
        params_retrieve = top_k_uuids
        log.debug(f"Retrieve SQL: {sql_retrieve}, PARAMS: {params_retrieve}")
        
        results_map = {}
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql_retrieve, params_retrieve)
            rows = cursor.fetchall()
            log.debug(f"Retrieved {len(rows)} full docs from DB.")
            for row in rows:
                try:
                    doc = self._row_to_memory_doc(row)
                    doc.score = scores_map.get(doc.id) # Assign the semantic score
                    results_map[doc.id] = doc
                except Exception as convert_err:
                    log.error(f"Skipping row due to conversion error during semantic result retrieval: {convert_err}. Row: {dict(row)}")
        except sqlite3.Error as e:
            log.error(f"SQLite error retrieving full docs for semantic query: {e}", exc_info=True)
            return [] # Return empty list on DB error
        finally:
            if cursor:
                cursor.close()
                log.debug("Cursor closed after retrieving semantic results.")

        # Re-order results according to the semantic search score order
        ordered_results = [results_map[uuid_] for uuid_ in top_k_uuids if uuid_ in results_map]
        log.info(f"Semantic query finished. Returning {len(ordered_results)} MemoryDoc objects.")
        return ordered_results

    def hybrid_query(
        self,
        query_text: str,
        k: int = 10,
        filter_tags: Optional[List[str]] = None,
        filter_source_agents: Optional[List[str]] = None,
        rrf_k: int = 60,  # RRF constant, common default is 60
        alpha: float = 0.5 # Weight for FTS vs Semantic (0=FTS only, 1=Semantic only)
    ) -> List[Tuple[MemoryDoc, float]]:
        """Performs hybrid search (FTS + Semantic) using Reciprocal Rank Fusion (RRF)."""
        log.debug(f"Entering hybrid_query method: query='{query_text[:50]}...', k={k}, tags={filter_tags}, agents={filter_source_agents}, alpha={alpha}, rrf_k={rrf_k}")
        if not self.conn:
            log.error("Cannot hybrid query: SQLite connection is not established.")
            return []

        # Ensure alpha is within valid range
        alpha = max(0.0, min(1.0, alpha))
        log.debug(f"Using alpha (semantic weight): {alpha}")

        # Determine how many results to fetch from each method (fetch more for RRF)
        fetch_k = max(k * 2, 50) # Fetch more results for better ranking potential
        log.debug(f"Fetching top {fetch_k} results from FTS and Semantic searches.")

        # 1. FTS Query
        log.debug("Performing FTS query part of hybrid search...")
        fts_results_docs = self.query(query_text, fetch_k, filter_tags, filter_source_agents)
        fts_results = {doc.id: rank + 1 for rank, doc in enumerate(fts_results_docs)}
        log.info(f"Hybrid Query: FTS returned {len(fts_results)} results.")
        # Assign FTS scores (lower rank is better)
        # fts_scores = {doc.id: 1.0 / (rrf_k + rank + 1) for rank, doc in enumerate(fts_results)}

        # 2. Semantic Query
        semantic_results_docs = []
        semantic_results = {}
        if self.embedding_manager.embedding_model is not None and alpha > 0:
            log.debug("Performing Semantic query part of hybrid search...")
            semantic_results_docs = self.semantic_query(query_text, fetch_k, filter_tags, filter_source_agents)
            # semantic_scores = {doc.id: 1.0 / (rrf_k + rank + 1) for rank, doc in enumerate(semantic_results)}
            semantic_results = {doc.id: rank + 1 for rank, doc in enumerate(semantic_results_docs)}
            log.info(f"Hybrid Query: Semantic returned {len(semantic_results)} results.")
        elif alpha > 0:
            log.warning("Semantic query requested (alpha > 0) but model not available. Hybrid search will only use FTS.")
        else:
            log.info("Semantic query skipped (alpha == 0).")

        # 3. Combine results using Weighted RRF or Simple Weighting
        log.debug("Combining FTS and Semantic results...")
        combined_scores: Dict[str, float] = {}
        all_doc_ids = set(fts_results.keys()) | set(semantic_results.keys())
        log.debug(f"Total unique documents from both searches: {len(all_doc_ids)}")

        for doc_id in all_doc_ids:
            fts_rank = fts_results.get(doc_id)
            semantic_rank = semantic_results.get(doc_id)
            
            fts_score = 1.0 / (rrf_k + fts_rank) if fts_rank is not None else 0.0
            semantic_score = 1.0 / (rrf_k + semantic_rank) if semantic_rank is not None else 0.0
            
            # Weighted combination
            combined_scores[doc_id] = (1 - alpha) * fts_score + alpha * semantic_score
            # log.debug(f"Doc ID: {doc_id}, FTS Rank: {fts_rank}, Sem Rank: {semantic_rank}, FTS Score: {fts_score:.4f}, Sem Score: {semantic_score:.4f}, Combined: {combined_scores[doc_id]:.4f}")


        # 4. Sort by combined score and retrieve top K documents
        sorted_doc_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)
        top_k_ids = sorted_doc_ids[:k]
        log.info(f"Combined scores calculated for {len(combined_scores)} docs. Returning top {len(top_k_ids)}.")
        log.debug(f"Top {k} combined doc IDs: {top_k_ids}")

        if not top_k_ids:
            return []

        # Retrieve full documents for the top K IDs
        # We can reuse the docs we already fetched if available
        results_map = {doc.id: doc for doc in fts_results_docs}
        results_map.update({doc.id: doc for doc in semantic_results_docs}) # Overwrite with semantic if present, though content is same

        final_results: List[Tuple[MemoryDoc, float]] = []
        retrieved_count = 0
        missing_ids = []
        for doc_id in top_k_ids:
            if doc_id in results_map:
                 final_results.append((results_map[doc_id], combined_scores[doc_id]))
                 retrieved_count += 1
            else:
                missing_ids.append(doc_id)
                log.warning(f"Document ID {doc_id} from top hybrid results not found in initially fetched docs. This might indicate an issue.")
        
        # If any top IDs were missing, fetch them directly (should be rare)
        if missing_ids:
            log.warning(f"Fetching {len(missing_ids)} missing top-k documents directly from DB...")
            sql_retrieve_missing = f"SELECT * FROM memories WHERE uuid IN ({','.join(['?']*len(missing_ids))})"
            cursor = None
            try:
                cursor = self.conn.cursor()
                cursor.execute(sql_retrieve_missing, missing_ids)
                rows = cursor.fetchall()
                log.debug(f"Retrieved {len(rows)} missing docs.")
                for row in rows:
                    try:
                        doc = self._row_to_memory_doc(row)
                        final_results.append((doc, combined_scores[doc.id]))
                        retrieved_count += 1
                    except Exception as convert_err:
                        log.error(f"Skipping row due to conversion error during missing doc retrieval: {convert_err}. Row: {dict(row)}")
            except sqlite3.Error as e:
                log.error(f"SQLite error retrieving missing docs for hybrid query: {e}", exc_info=True)
            finally:
                if cursor:
                    cursor.close()
                    log.debug("Cursor closed after retrieving missing docs.")
            # Re-sort final results potentially including newly fetched ones
            final_results.sort(key=lambda item: item[1], reverse=True)

        log.info(f"Hybrid query finished. Returning {len(final_results)} documents.")
        return final_results[:k] # Ensure we only return k

    def close(self):
        """Closes the SQLite connection."""
        log.info("Attempting to close SQLite connection...")
        if self.conn:
            try:
                self.conn.close()
                log.info("SQLite connection closed successfully.")
                self.conn = None
            except sqlite3.Error as e:
                log.error(f"Error closing SQLite connection: {e}")
        else:
            log.warning("Close called but SQLite connection was not established or already closed.")

    def __del__(self):
        """Ensure connection is closed when the object is deleted."""
        # log.debug("HybridSQLiteAdapter __del__ called. Attempting to close connection.") # Can be noisy
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
