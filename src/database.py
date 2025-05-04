import sqlite3
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import datetime
import json

logger = logging.getLogger(__name__)

DATABASE_NAME = "arxiv_papers.db"
DB_PATH = Path(__file__).parent.parent / DATABASE_NAME # Place DB in project root

# --- Custom Datetime Handling ---
def adapt_datetime_iso(val: datetime.datetime) -> str:
    """Adapt datetime.datetime to timezone-aware ISO 8601 string format."""
    return val.isoformat()

def convert_timestamp_iso(val: bytes) -> datetime.datetime:
    """Convert ISO 8601 string timestamp back to datetime.datetime object."""
    # The value from SQLite might be bytes
    dt_str = val.decode('utf-8')
    # Handle timezone offsets like +00:00 and Z
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    elif '+' not in dt_str and '-' not in dt_str[10:]: # Check if timezone info exists after date part
         # Assume UTC if no timezone info (or handle as naive if preferred)
         # For consistency with input, let's assume UTC if stored without offset
         # Or, better, rely on the input always having timezone
         pass # Let fromisoformat handle it, might raise error if format unexpected

    try:
        return datetime.datetime.fromisoformat(dt_str)
    except ValueError as e:
        logger.error(f"Could not parse timestamp '{val!r}': {e}")
        # Return a sensible default or re-raise, depending on desired strictness
        # Returning epoch might hide issues, maybe None is better or re-raise?
        raise # Re-raise the error to make parsing issues obvious

# Register the adapter and converter
sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp_iso)

# --- Database Initialization ---

def get_db_connection() -> Optional[sqlite3.Connection]:
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        logger.info(f"Database connection established to {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database {DB_PATH}: {e}", exc_info=True)
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
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT UNIQUE NOT NULL, -- e.g., '2310.06825v1'
            title TEXT NOT NULL,
            authors TEXT, -- Stored as JSON string list
            summary TEXT,
            source_pdf_url TEXT UNIQUE, -- Direct link to arXiv PDF
            download_path TEXT, -- Local path where PDF is saved
            publication_date TIMESTAMP,
            last_updated_date TIMESTAMP,
            categories TEXT, -- Stored as JSON string list
            status TEXT DEFAULT 'pending', -- e.g., pending, downloaded, processing, summarized, complete, error
            notes TEXT,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        # Add indexes for frequently queried columns
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_arxiv_id ON papers (arxiv_id);");
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON papers (status);");
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_publication_date ON papers (publication_date);");
        conn.commit()
        logger.info("Table 'papers' checked/created successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error creating table 'papers': {e}", exc_info=True)
        conn.rollback()

def initialize_database():
    """Initializes the database: connects and creates tables."""
    conn = get_db_connection()
    if conn:
        create_tables(conn)
        close_db_connection(conn)
    else:
        logger.error("Database initialization failed: Could not establish connection.")

# --- CRUD Operations ---

def add_paper(conn: sqlite3.Connection, paper_data: Dict[str, Any]) -> Optional[int]:
    """Adds a new paper to the database.

    Args:
        conn: The database connection.
        paper_data: A dictionary containing paper details. Must include 'arxiv_id' and 'title'.
                    Other keys can be: 'authors', 'summary', 'source_pdf_url',
                    'download_path', 'publication_date', 'last_updated_date', 'categories',
                    'status', 'notes'.

    Returns:
        The ID of the newly inserted row, or None if insertion fails.
    """
    required_fields = ['arxiv_id', 'title']
    if not all(field in paper_data for field in required_fields):
        logger.error(f"Missing required fields ({required_fields}) in paper_data.")
        return None

    # Work on a copy to avoid modifying the original dict
    data_copy = paper_data.copy()

    # Convert lists to JSON strings if present
    if 'authors' in data_copy and isinstance(data_copy['authors'], list):
        data_copy['authors'] = json.dumps(data_copy['authors'])
    if 'categories' in data_copy and isinstance(data_copy['categories'], list):
        data_copy['categories'] = json.dumps(data_copy['categories'])

    # Ensure dates are handled correctly (assuming they might be datetime objects)
    for date_field in ['publication_date', 'last_updated_date']:
        if date_field in data_copy and isinstance(data_copy[date_field], datetime.datetime):
            # The adapter now handles conversion, so this comment is less relevant
            pass

    # Construct SQL query dynamically based on provided keys
    fields = list(data_copy.keys())
    placeholders = ', '.join(['?' for _ in fields])
    sql = f"INSERT INTO papers ({', '.join(fields)}) VALUES ({placeholders})"

    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, [data_copy.get(field) for field in fields])
            last_id = cursor.lastrowid
            logger.info(f"Paper '{data_copy['arxiv_id']}' added successfully with ID {last_id}.")
            return last_id
    except sqlite3.IntegrityError as e:
        logger.warning(f"Integrity error adding paper '{data_copy.get('arxiv_id', 'N/A')}': {e}")
        return None # Likely duplicate arxiv_id or source_pdf_url
    except sqlite3.Error as e:
        logger.error(f"Database error adding paper '{data_copy.get('arxiv_id', 'N/A')}': {e}", exc_info=True)
        return None

def _parse_paper_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Helper to convert a Row object into a dict, parsing JSON fields."""
    paper = dict(row)
    for field in ['authors', 'categories']:
        if paper.get(field):
            try:
                parsed_data = json.loads(paper[field])
                # Ensure the parsed data is actually a list
                if isinstance(parsed_data, list):
                    paper[field] = parsed_data
                else:
                    logger.warning(f"Parsed '{field}' JSON for paper ID {paper.get('id')} is not a list: {paper.get(field)}")
                    paper[field] = [] # Default to empty list if not a list
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse '{field}' JSON for paper ID {paper.get('id')}: {paper.get(field)}")
                paper[field] = [] # Default to empty list on error
        else:
            paper[field] = [] # Default to empty list if None/empty string

    # Ensure date fields are datetime objects (converter should handle this)
    for date_field in ['publication_date', 'last_updated_date', 'added_date', 'updated_date']:
        if date_field in paper and not isinstance(paper[date_field], datetime.datetime):
            logger.warning(f"Field '{date_field}' (value: {paper[date_field]}) is not a datetime object after retrieval.")
            # Attempt conversion again or set to None if critical
            # This might indicate an issue with the converter registration or usage
            pass # For now, just log
    return paper

def get_paper_by_arxiv_id(conn: sqlite3.Connection, arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single paper by its arXiv ID."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        row = cursor.fetchone()
        if row:
            return _parse_paper_row(row)
        else:
            logger.debug(f"Paper with arxiv_id '{arxiv_id}' not found.")
            return None
    except sqlite3.Error as e:
        logger.error(f"Database error getting paper by arxiv_id '{arxiv_id}': {e}", exc_info=True)
        return None

def get_all_papers(conn: sqlite3.Connection, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all papers, optionally filtering by status."""
    papers = []
    try:
        cursor = conn.cursor()
        if status_filter:
            cursor.execute("SELECT * FROM papers WHERE status = ? ORDER BY added_date DESC, id DESC", (status_filter,))
        else:
            cursor.execute("SELECT * FROM papers ORDER BY added_date DESC, id DESC")

        rows = cursor.fetchall()
        papers = [_parse_paper_row(row) for row in rows]
        logger.debug(f"Retrieved {len(papers)} papers" + (f" with status '{status_filter}'" if status_filter else ""))
    except sqlite3.Error as e:
        logger.error(f"Database error getting all papers: {e}", exc_info=True)
    return papers

def update_paper_field(conn: sqlite3.Connection, arxiv_id: str, field: str, value: Any) -> bool:
    """Updates a specific field for a paper identified by arxiv_id.

    Handles JSON encoding for list fields ('authors', 'categories').
    Updates the 'updated_date' field automatically.

    Args:
        conn: The database connection.
        arxiv_id: The arXiv ID of the paper to update.
        field: The name of the database column to update.
        value: The new value for the field.

    Returns:
        True if the update was successful, False otherwise.
    """
    allowed_fields = [
        'title', 'authors', 'summary', 'source_pdf_url', 'download_path',
        'publication_date', 'last_updated_date', 'categories', 'status', 'notes'
    ]
    if field not in allowed_fields:
        logger.error(f"Attempted to update disallowed field '{field}' for paper '{arxiv_id}'.")
        return False

    # Convert lists to JSON strings if necessary
    if field in ['authors', 'categories'] and isinstance(value, list):
        value = json.dumps(value)

    # Ensure dates are handled correctly (assuming they might be datetime objects)
    if field in ['publication_date', 'last_updated_date'] and isinstance(value, datetime.datetime):
         # The adapter now handles conversion, so this comment is less relevant
         pass

    sql = f"UPDATE papers SET {field} = ?, updated_date = CURRENT_TIMESTAMP WHERE arxiv_id = ?"

    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, (value, arxiv_id))
            if cursor.rowcount > 0:
                logger.info(f"Paper '{arxiv_id}' field '{field}' updated successfully.")
                return True
            else:
                logger.warning(f"Paper '{arxiv_id}' not found for update or field value unchanged.")
                return False
    except sqlite3.IntegrityError as e:
        logger.warning(f"Integrity error updating paper '{arxiv_id}' field '{field}': {e}")
        return False # Likely duplicate source_pdf_url if that was updated
    except sqlite3.Error as e:
        logger.error(f"Database error updating paper '{arxiv_id}' field '{field}': {e}", exc_info=True)
        return False

def delete_paper(conn: sqlite3.Connection, arxiv_id: str) -> bool:
    """Deletes a paper from the database by its arXiv ID."""
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM papers WHERE arxiv_id = ?", (arxiv_id,))
            if cursor.rowcount > 0:
                logger.info(f"Paper '{arxiv_id}' deleted successfully.")
                return True
            else:
                logger.warning(f"Paper '{arxiv_id}' not found for deletion.")
                return False
    except sqlite3.Error as e:
        logger.error(f"Database error deleting paper '{arxiv_id}': {e}", exc_info=True)
        return False

# --- Main execution for testing ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing database...")
    initialize_database()
    logger.info("Database initialization complete.")

    # Example usage (optional, primarily for direct script run)
    # conn = get_db_connection()
    # if conn:
    #     # Perform some operations
    #     close_db_connection(conn)
