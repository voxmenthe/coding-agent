import sqlite3
import logging
from datetime import datetime, timezone
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)

# --- Custom Datetime Handling ---
def adapt_datetime_iso(val: datetime) -> str:
    """Adapt datetime.datetime to timezone-aware ISO 8601 string format (UTC)."""
    if val.tzinfo is None:
        # Assume naive datetime is UTC before formatting
        val = val.replace(tzinfo=timezone.utc)
    result = val.isoformat()
    return result

def convert_timestamp_iso(val: bytes) -> datetime:
    """Convert ISO 8601 string timestamp back to datetime.datetime object."""
    # The value from SQLite might be bytes
    dt_str = val.decode('utf-8')
    # Handle timezone offsets like +00:00 and Z
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    elif '+' not in dt_str and '-' not in dt_str[10:]: # Check if timezone info exists after date part
         # Assume UTC if no timezone info (or handle as naive if preferred)
         # For consistency with input, let's assume UTC if stored without offset
         pass # Let fromisoformat handle it, might raise error if format unexpected

    try:
        dt = datetime.fromisoformat(dt_str)
        # If parsing results in a naive datetime, assume it's UTC
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt # Already timezone-aware
    except ValueError as e:
        logger.error(f"Could not parse timestamp '{val!r}': {e}")
        raise # Re-raise the error to make parsing issues obvious

# Register the adapter and converter
sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp_iso)

# --- Database Initialization ---

def get_db_connection(db_path: Path) -> Optional[sqlite3.Connection]:
    """Establishes a connection to the SQLite database at the specified path."""
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        conn.execute("PRAGMA foreign_keys = ON;") # Recommended for data integrity
        logger.info(f"Database connection established to {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}", exc_info=True)
        return None

def close_db_connection(conn: Optional[sqlite3.Connection]):
    """Closes the database connection."""
    if conn:
        conn.close()
        logger.info("Database connection closed.")

def create_tables(conn: sqlite3.Connection):
    """Creates the necessary tables if they don't exist."""
    try:
        cursor = conn.cursor()
        # --- Papers Table --- V2 Schema
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_filename TEXT NOT NULL,         -- Original filename of the PDF
            arxiv_id TEXT UNIQUE,                  -- Optional: Extracted arXiv ID (e.g., '2310.06825v1')
            title TEXT,                            -- Optional: Extracted/User-provided title
            authors TEXT,                          -- Optional: Stored as JSON string list
            summary TEXT,                          -- Optional: Extracted abstract/summary
            blob_path TEXT,                        -- Optional: Path to the saved text blob (relative to BLOBS_DIR)
            source_pdf_url TEXT UNIQUE,            -- Optional: Direct link to arXiv PDF or other source
            genai_file_uri TEXT,                 -- Optional: URI of the file stored in GenAI File Service
            publication_date TIMESTAMP,            -- Optional: Extracted from metadata
            last_updated_date TIMESTAMP,           -- Optional: From arXiv metadata or manual update
            categories TEXT,                       -- Optional: Stored as JSON string list (e.g., from arXiv)
            status TEXT DEFAULT 'pending',         -- e.g., pending, processing, complete, error_fetch, error_process, error_blob
            notes TEXT,                            -- Optional: User notes
            processed_timestamp TIMESTAMP,         -- When the PDF was processed by the agent
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- When the record was added
            updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Tracks local record updates
        );
        """)
        # --- Indexes ---
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_arxiv_id ON papers (arxiv_id);");
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON papers (status);");
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_publication_date ON papers (publication_date);");
        conn.commit()
        logger.info("Table 'papers' checked/created successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error creating table 'papers': {e}", exc_info=True)
        conn.rollback()

def initialize_database(db_path: Path) -> Optional[sqlite3.Connection]:
    """Initializes the database: gets connection and creates tables."""
    conn = get_db_connection(db_path)
    if conn:
        create_tables(conn)
        # Set journal mode to WAL for potentially better concurrency
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            logger.info("Database journal mode set to WAL.")
        except sqlite3.Error as e:
            logger.warning(f"Could not set journal mode to WAL: {e}")
        return conn
    else:
        logger.error(f"Database initialization failed: Could not establish connection to {db_path}.") # Updated log
        return None

# --- CRUD Operations ---

def add_minimal_paper(conn: sqlite3.Connection, source_filename: str) -> Optional[int]:
    """Adds a new paper record with only the source filename and status='processing',
       and returns its ID."""
    if not source_filename:
        logger.error("Cannot add minimal paper without a source filename.")
        return None

    sql = "INSERT INTO papers (source_filename, status) VALUES (?, ?)"

    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, (source_filename, 'processing'))
            last_id = cursor.lastrowid
            logger.info(f"Minimal paper record added for '{source_filename}' with ID {last_id}.")
            return last_id
    except sqlite3.Error as e:
        logger.error(f"Database error adding minimal paper for '{source_filename}': {e}", exc_info=True)
        return None


def add_paper(conn: sqlite3.Connection, paper_data: Dict[str, Any]) -> Optional[int]:
    """Adds a new paper to the database. 
       ***Prefer add_minimal_paper + update_paper_field for standard flow.***
    """
    required_fields = ['source_filename'] # Only source_filename is truly required now
    if not all(field in paper_data and paper_data[field] is not None for field in required_fields):
        logger.error(f"Missing required field 'source_filename' in paper_data for add_paper.")
        return None

    # Work on a copy to avoid modifying the original dict
    data_copy = paper_data.copy()

    # Let's ensure updated_date is always set on add, like added_date
    data_copy['updated_date'] = datetime.now(timezone.utc)

    # Construct SQL query with explicit fields based on sample_paper_data fixtures for robustness
    fields = [
        'source_filename', 'arxiv_id', 'title', 'authors', 
        'summary', 'publication_date', 'last_updated_date', 'categories', 'notes', 'status', 'updated_date',
        'source_pdf_url'
    ] # Assuming these are the core fields in fixtures
    placeholders = ', '.join(['?' for _ in fields])
    # Ensure JSON fields are dumped and other fields are correctly retrieved

    values = [
        data_copy.get('source_filename'),
        data_copy.get('arxiv_id'),
        data_copy.get('title'),
        json.dumps(data_copy.get('authors', [])) if data_copy.get('authors') else None,
        data_copy.get('summary'),
        data_copy.get('publication_date'), # Pass original value from fixture
        data_copy.get('last_updated_date'),
        json.dumps(data_copy.get('categories', [])) if data_copy.get('categories') else None,
        data_copy.get('notes'),
        data_copy.get('status', 'pending'), # Default status if not provided
        data_copy.get('updated_date'), # Add updated_date
        data_copy.get('source_pdf_url'), # Added missing field value
    ]
    sql = f"INSERT INTO papers ({', '.join(fields)}) VALUES ({placeholders})"

    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, values)
            last_id = cursor.lastrowid
            logger.info(f"Paper '{data_copy['source_filename']}' added successfully with ID {last_id}.")
            return last_id
    except sqlite3.IntegrityError as e:
        logger.warning(f"Integrity error adding paper '{data_copy.get('source_filename', 'N/A')}': {e}")
        return None # Likely duplicate arxiv_id or source_pdf_url
    except sqlite3.Error as e:
        logger.error(f"Database error adding paper '{data_copy.get('source_filename', 'N/A')}': {e}", exc_info=True)
        return None

def _parse_paper_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a database row into a dictionary, handling JSON fields and timestamps."""
    paper_dict = dict(row)
    list_fields = ['authors', 'categories']
    for field in list_fields:
        if paper_dict.get(field):
            try:
                paper_dict[field] = json.loads(paper_dict[field])
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON for field '{field}' in paper ID {paper_dict.get('id')}. Setting to empty list.")
                paper_dict[field] = [] # Set to empty list on error

    # Convert timestamp fields (already handled by converters, but good to be explicit)
    # No explicit conversion needed here if converters are registered and used correctly.
    # Ensure all expected keys are present, potentially adding None if missing
    expected_keys = [
        'id', 'source_filename', 'arxiv_id', 'title', 'authors', 'summary',
        'blob_path', 'source_pdf_url', 'publication_date', 'last_updated_date',
        'categories', 'status', 'notes', 'processed_timestamp', 'added_date', 'updated_date',
        'genai_file_uri'
    ]
    for key in expected_keys:
        if key not in paper_dict:
            paper_dict[key] = None

    return paper_dict

def get_paper_by_id(conn: sqlite3.Connection, paper_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a paper by its primary key ID."""
    sql = "SELECT * FROM papers WHERE id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (paper_id,))
        row = cursor.fetchone()
        if row:
            return _parse_paper_row(row)
        else:
            logger.info(f"No paper found with ID {paper_id}.")
            return None
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving paper ID {paper_id}: {e}", exc_info=True)
        return None

def get_paper_by_arxiv_id(conn: sqlite3.Connection, arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a paper by its arXiv ID."""
    sql = "SELECT * FROM papers WHERE arxiv_id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (arxiv_id,))
        row = cursor.fetchone()
        if row:
            return _parse_paper_row(row)
        else:
            logger.info(f"No paper found with arXiv ID '{arxiv_id}'.")
            return None
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving paper by arXiv ID '{arxiv_id}': {e}", exc_info=True)
        return None

def get_all_papers(conn: sqlite3.Connection, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all papers, optionally filtered by status.
    
    Args:
        conn: Active SQLite database connection.
        status_filter: Optional status string to filter papers by.

    Returns:
        A list of paper dictionaries, ordered by descending added_date.
    """
    papers = []
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM papers"
        params = []
        if status_filter:
            query += " WHERE status = ?"
            params.append(status_filter)
        # Order by most recently added first
        query += " ORDER BY added_date DESC" 
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        papers = [_parse_paper_row(row) for row in rows]
        logger.info(f"Retrieved {len(papers)} papers" + (f" with status '{status_filter}'." if status_filter else "."))
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving papers: {e}", exc_info=True)
    return papers

def update_paper_field(conn: sqlite3.Connection, paper_id: int, field: str, value: Any) -> bool:
    """Updates a specific field for a given paper ID.
    
    Returns:
        bool: True if update was successful (row found and changed), False otherwise.
    """
    # Prevent updating immutable fields
    if field in ['id', 'source_filename', 'added_date', 'updated_date']: 
        logger.error(f"Attempted to update immutable field '{field}' for paper ID {paper_id}. Operation aborted.")
        return False

    # Special check for arxiv_id: allow update only if currently NULL
    if field == 'arxiv_id':
        try:
            cursor_check = conn.cursor()
            cursor_check.execute("SELECT arxiv_id FROM papers WHERE id = ?", (paper_id,))
            row = cursor_check.fetchone()
            if row and row['arxiv_id'] is not None:
                 logger.error(f"Attempted to update non-NULL arxiv_id for paper ID {paper_id}. Operation aborted.")
                 return False
            elif not row:
                 logger.warning(f"Paper ID {paper_id} not found for arxiv_id update check.")
                 return False # Paper doesn't exist, can't update
        except sqlite3.Error as e:
            logger.error(f"Database error checking arxiv_id for paper ID {paper_id}: {e}", exc_info=True)
            return False

    # Validate field exists (optional, protects against typos)
    # Get column names
    cursor = conn.execute(f"PRAGMA table_info(papers)")
    columns = [row[1] for row in cursor.fetchall()]
    if field not in columns:
        logger.error(f"Field '{field}' does not exist in 'papers' table. Update failed for paper ID {paper_id}.")
        return False

    # Handle list fields by converting to JSON string
    if field in ['authors', 'categories'] and isinstance(value, list):
        value = json.dumps(value)
    elif isinstance(value, list):
        # If it's a list but not a designated list field, log an error or handle appropriately
        logger.error(f"Attempting to save a list to non-list field '{field}' for paper ID {paper_id}. Value type: {type(value)}. Aborting update.")
        return False

    # Add updated_date automatically
    now_utc = datetime.now(timezone.utc)
    sql = f"UPDATE papers SET {field} = ?, updated_date = ? WHERE id = ?"

    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, (value, now_utc, paper_id))
            # Check if any row was actually updated
            success = cursor.rowcount > 0
            if success:
                 logger.info(f"Updated field '{field}' for paper ID {paper_id}.")
            else:
                 logger.warning(f"No paper found with ID {paper_id} to update field '{field}'.")
            return success 
    except sqlite3.Error as e:
        logger.error(f"Database error updating field '{field}' for paper ID {paper_id}: {e}", exc_info=True)
        return False

def delete_paper(conn: sqlite3.Connection, paper_id: int) -> bool:
    """Deletes a paper from the database by its ID.
    
    Returns:
        bool: True if deletion was successful (row found and deleted), False otherwise.
    """
    sql = "DELETE FROM papers WHERE id = ?"
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, (paper_id,))
            # Check if a row was actually deleted
            success = cursor.rowcount > 0
            if success:
                logger.info(f"Deleted paper with ID {paper_id}.")
            else:
                logger.warning(f"No paper found with ID {paper_id} to delete.") # Changed log level
            return success
    except sqlite3.Error as e:
        logger.error(f"Database error deleting paper with ID {paper_id}: {e}", exc_info=True)
        return False

# --- Helper/Utility Functions ---
# ...

if __name__ == "__main__":
    # Example usage (for testing purposes)
    logging.basicConfig(level=logging.INFO)
    
    # Use a temporary in-memory database or a test file for demonstration
    # test_db_path = Path("./test_papers.db") 
    # test_db_path.unlink(missing_ok=True) # Delete previous test DB
    
    # --- Use in-memory for basic tests --- 
    # conn = initialize_database(":memory:") # Does not work if Path object expected
    
    # --- Use file for persistent tests --- 
    test_db_path = Path("./test_papers.db") 
    test_db_path.unlink(missing_ok=True) # Clean up before test
    conn = initialize_database(test_db_path)

    if conn:
        # 1. Add minimal paper
        min_paper_id = add_minimal_paper(conn, "my_presentation.pdf")
        print(f"Minimal paper added with ID: {min_paper_id}")

        # 2. Get the minimal paper
        if min_paper_id:
            paper = get_paper_by_id(conn, min_paper_id)
            print(f"Retrieved minimal paper: {paper}")

            # 3. Update some fields
            update_paper_field(conn, min_paper_id, 'title', "My Awesome Presentation")
            update_paper_field(conn, min_paper_id, 'status', "complete")
            update_paper_field(conn, min_paper_id, 'blob_path', "blobs/my_presentation.txt")
            update_paper_field(conn, min_paper_id, 'arxiv_id', "local_001") # Example non-arXiv ID

            # 4. Get updated paper
            paper_updated = get_paper_by_id(conn, min_paper_id)
            print(f"Updated paper: {paper_updated}")

            # 5. Try updating an immutable field (should fail)
            update_paper_field(conn, min_paper_id, 'source_filename', "new_name.pdf")

            # 6. Try updating non-existent field (should fail)
            update_paper_field(conn, min_paper_id, 'non_existent_field', "some_value")
            
            # 7. Add another paper using the old add_paper (demonstrating flexibility)
            arxiv_paper_data = {
                'source_filename': '2310.06825v1.pdf',
                'arxiv_id': '2310.06825v1',
                'title': 'Large Language Models are Zero-Shot Reasoners',
                'authors': ['Takeshi Kojima', 'Shixiang Shane Gu', 'Machel Reid', 'Yutaka Matsuo', 'Yusuke Iwasawa'],
                'summary': 'Foundation models, which are trained on broad data at scale...', # Truncated
                'status': 'pending', # Example
                'source_pdf_url': 'https://arxiv.org/pdf/2310.06825v1.pdf'
            }
            arxiv_paper_id = add_paper(conn, arxiv_paper_data)
            print(f"arXiv paper added with ID: {arxiv_paper_id}")

            if arxiv_paper_id:
                arxiv_paper = get_paper_by_id(conn, arxiv_paper_id)
                print(f"Retrieved arXiv paper: {arxiv_paper}")
                # Test get by arxiv id
                arxiv_paper_by_aid = get_paper_by_arxiv_id(conn, '2310.06825v1')
                print(f"Retrieved by arxiv_id: {arxiv_paper_by_aid}")

            # 8. Delete the first paper
            delete_paper(conn, min_paper_id)
            print(f"Attempted to delete paper ID: {min_paper_id}")
            paper_after_delete = get_paper_by_id(conn, min_paper_id)
            print(f"Paper after delete attempt: {paper_after_delete}") # Should be None

        close_db_connection(conn)
        # Optional: Clean up test database file
        # test_db_path.unlink()
