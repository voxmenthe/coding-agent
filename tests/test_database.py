import sqlite3
import pytest
import os
import sys
import json
from datetime import datetime, timezone, timedelta
import time

# Ensure the src directory is in the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import the database module
from src import database

# --- Test Fixtures ---

@pytest.fixture
def db_conn():
    """Pytest fixture to set up and tear down an in-memory database."""
    # Use :memory: for an in-memory database that's destroyed when the connection closes
    conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    # Create the table using the function from the database module
    database.create_tables(conn)
    yield conn
    conn.close()

@pytest.fixture
def sample_paper_data_1():
    return {
        "source_filename": "2310.00001v1.pdf",
        "arxiv_id": "2310.00001v1",
        "title": "Test Paper 1: Quantum Widgets",
        "authors": ["Alice", "Bob"],
        "summary": "A paper about testing widgets.",
        "source_pdf_url": "https://arxiv.org/pdf/2310.00001v1.pdf",
        "publication_date": datetime(2023, 10, 1, 12, 0, 0, tzinfo=timezone.utc),
        "last_updated_date": datetime(2023, 10, 2, 10, 0, 0, tzinfo=timezone.utc),
        "categories": ["cs.AI", "stat.ML"],
        "status": "pending",
        "notes": "Initial submission."
    }

@pytest.fixture
def sample_paper_data_2():
    return {
        "source_filename": "2401.00002v2.pdf",
        "arxiv_id": "2401.00002v2",
        "title": "Test Paper 2: Advanced Sprockets",
        "authors": ["Charlie"],
        "summary": "Further research into sprockets.",
        "source_pdf_url": "https://arxiv.org/pdf/2401.00002v2.pdf",
        "publication_date": datetime(2024, 1, 5, 9, 0, 0, tzinfo=timezone.utc),
        "last_updated_date": datetime(2024, 1, 6, 15, 30, 0, tzinfo=timezone.utc),
        "categories": ["physics.COMP"],
        "status": "downloaded",
        "notes": None
    }

@pytest.fixture
def sample_paper_data_tz_date():
    """Minimal data with a timezone-aware publication date."""
    return {
        "source_filename": "tz_date_test.pdf",
        "publication_date": datetime(2022, 11, 15, 10, 0, 0, tzinfo=timezone.utc)
    }

# --- Test Cases ---

def test_create_tables(db_conn):
    """Test if the papers table is created successfully."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers';")
    result = cursor.fetchone()
    assert result is not None
    assert result['name'] == 'papers'
    # Check if indexes were created (optional but good practice)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='papers';")
    indexes = {row['name'] for row in cursor.fetchall()}
    assert 'idx_arxiv_id' in indexes
    assert 'idx_status' in indexes
    assert 'idx_publication_date' in indexes

def test_add_paper_success(db_conn, sample_paper_data_1):
    """Test adding a new paper successfully."""
    paper_id = database.add_paper(db_conn, sample_paper_data_1)
    assert paper_id is not None
    assert isinstance(paper_id, int)

    # Verify the data was inserted correctly
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    row = cursor.fetchone()
    assert row is not None
    assert row['source_filename'] == sample_paper_data_1['source_filename']
    assert row['arxiv_id'] == sample_paper_data_1['arxiv_id']
    assert row['title'] == sample_paper_data_1['title']
    assert json.loads(row['authors']) == sample_paper_data_1['authors'] # Check JSON conversion
    assert row['summary'] == sample_paper_data_1['summary']
    assert row['source_pdf_url'] == sample_paper_data_1['source_pdf_url']
    # SQLite might store timezone info differently, compare aware vs aware
    assert row['publication_date'].isoformat() == sample_paper_data_1['publication_date'].isoformat()
    assert row['last_updated_date'].isoformat() == sample_paper_data_1['last_updated_date'].isoformat()
    assert json.loads(row['categories']) == sample_paper_data_1['categories']
    assert row['status'] == sample_paper_data_1['status']
    assert row['notes'] == sample_paper_data_1['notes']
    assert row['added_date'] is not None
    assert row['updated_date'] is not None

def test_add_paper_missing_required_fields(db_conn):
    """Test adding a paper with missing required fields fails."""
    incomplete_data = {"title": "Incomplete Paper"}
    paper_id = database.add_paper(db_conn, incomplete_data)
    assert paper_id is None

def test_add_paper_duplicate_arxiv_id(db_conn, sample_paper_data_1):
    """Test that adding a paper with a duplicate arxiv_id fails gracefully."""
    database.add_paper(db_conn, sample_paper_data_1) # Add first time
    paper_id_duplicate = database.add_paper(db_conn, sample_paper_data_1) # Try adding again
    assert paper_id_duplicate is None # Should fail due to UNIQUE constraint

    # Verify only one record exists
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers WHERE arxiv_id = ?", (sample_paper_data_1['arxiv_id'],))
    count = cursor.fetchone()[0]
    assert count == 1

def test_get_paper_by_arxiv_id_found(db_conn, sample_paper_data_1):
    """Test retrieving an existing paper by arxiv_id."""
    database.add_paper(db_conn, sample_paper_data_1)
    paper = database.get_paper_by_arxiv_id(db_conn, sample_paper_data_1['arxiv_id'])
    assert paper is not None
    assert paper['arxiv_id'] == sample_paper_data_1['arxiv_id']
    assert paper['title'] == sample_paper_data_1['title']
    # Check list conversion from JSON
    assert paper['authors'] == sample_paper_data_1['authors']
    assert paper['categories'] == sample_paper_data_1['categories']
    # Check datetime objects (should be timezone-aware)
    assert paper['publication_date'].tzinfo is not None
    assert paper['publication_date'] == sample_paper_data_1['publication_date']

def test_get_paper_by_arxiv_id_not_found(db_conn):
    """Test retrieving a non-existent paper by arxiv_id."""
    paper = database.get_paper_by_arxiv_id(db_conn, "0000.00000")
    assert paper is None

def test_get_all_papers_empty(db_conn):
    """Test retrieving papers when the database is empty."""
    papers = database.get_all_papers(db_conn)
    assert isinstance(papers, list)
    assert len(papers) == 0

def test_get_all_papers_multiple(db_conn, sample_paper_data_1, sample_paper_data_2):
    """Test retrieving multiple papers, ensuring correct order."""
    paper_id_1 = database.add_paper(db_conn, sample_paper_data_1)
    assert paper_id_1 is not None
    time.sleep(0.01) # Add a small delay to ensure distinct timestamps
    paper_id_2 = database.add_paper(db_conn, sample_paper_data_2)
    assert paper_id_2 is not None

    papers = database.get_all_papers(db_conn)
    assert len(papers) == 2
    # Check presence instead of order, as timestamp precision might cause flakes
    retrieved_arxiv_ids = {p['arxiv_id'] for p in papers}
    assert sample_paper_data_1['arxiv_id'] in retrieved_arxiv_ids
    assert sample_paper_data_2['arxiv_id'] in retrieved_arxiv_ids

def test_get_all_papers_with_status_filter(db_conn, sample_paper_data_1, sample_paper_data_2):
    """Test retrieving papers filtered by status."""
    database.add_paper(db_conn, sample_paper_data_1) # status='pending'
    database.add_paper(db_conn, sample_paper_data_2) # status='downloaded'

    pending_papers = database.get_all_papers(db_conn, status_filter='pending')
    assert len(pending_papers) == 1
    assert pending_papers[0]['arxiv_id'] == sample_paper_data_1['arxiv_id']

    downloaded_papers = database.get_all_papers(db_conn, status_filter='downloaded')
    assert len(downloaded_papers) == 1
    assert downloaded_papers[0]['arxiv_id'] == sample_paper_data_2['arxiv_id']

    unknown_status_papers = database.get_all_papers(db_conn, status_filter='unknown')
    assert len(unknown_status_papers) == 0

def test_update_paper_field_success(db_conn, sample_paper_data_1):
    """Test updating a field of an existing paper."""
    paper_id = database.add_paper(db_conn, sample_paper_data_1)
    assert paper_id is not None

    # Update status
    new_status = "downloaded"
    success = database.update_paper_field(db_conn, paper_id, 'status', new_status)
    assert success is True
    paper = database.get_paper_by_id(db_conn, paper_id)
    assert paper['status'] == new_status
    original_updated_date = paper['updated_date']

    # Update notes
    new_notes = "Paper downloaded successfully."
    # Need a slight delay to ensure updated_date changes
    import time; time.sleep(0.01)
    success = database.update_paper_field(db_conn, paper_id, 'notes', new_notes)
    assert success is True
    paper = database.get_paper_by_id(db_conn, paper_id)
    assert paper['notes'] == new_notes
    assert paper['updated_date'] >= original_updated_date

    # Update categories (list -> JSON string)
    new_categories = ["cs.LG"]
    success = database.update_paper_field(db_conn, paper_id, 'categories', new_categories)
    assert success is True
    paper = database.get_paper_by_id(db_conn, paper_id)
    assert paper['categories'] == new_categories

def test_update_paper_field_not_found(db_conn):
    """Test updating a non-existent paper fails."""
    success = database.update_paper_field(db_conn, 99999, 'status', 'failed')
    assert success is False

def test_update_paper_field_disallowed_field(db_conn, sample_paper_data_1):
    """Test attempting to update a disallowed field (like id or arxiv_id)."""
    paper_id = database.add_paper(db_conn, sample_paper_data_1)
    success = database.update_paper_field(db_conn, paper_id, 'arxiv_id', 'new_id')
    assert success is False
    success = database.update_paper_field(db_conn, paper_id, 'id', 999)
    assert success is False
    success = database.update_paper_field(db_conn, paper_id, 'added_date', datetime.now(timezone.utc))
    assert success is False
    success = database.update_paper_field(db_conn, paper_id, 'updated_date', datetime.now(timezone.utc))
    assert success is False # updated_date is handled internally

def test_delete_paper_success(db_conn, sample_paper_data_1):
    """Test deleting an existing paper."""
    paper_id = database.add_paper(db_conn, sample_paper_data_1)
    assert paper_id is not None

    # Verify it exists
    paper = database.get_paper_by_id(db_conn, paper_id)
    assert paper is not None

    # Delete it
    deleted = database.delete_paper(db_conn, paper_id)
    assert deleted is True

    # Verify it's gone
    paper = database.get_paper_by_id(db_conn, paper_id)
    assert paper is None

def test_delete_paper_not_found(db_conn):
    """Test deleting a paper that does not exist."""
    deleted = database.delete_paper(db_conn, 99999)
    assert deleted is False

def test_parse_paper_row_bad_json(db_conn):
    """Test that _parse_paper_row handles invalid JSON gracefully."""
    # Manually insert a row with bad JSON in 'authors'
    bad_json_authors = '["Author One", "Author Two"' # Missing closing bracket
    source_file = "bad_json_test.pdf" # Need source_filename
    insert_sql = "INSERT INTO papers (source_filename, authors, categories) VALUES (?, ?, ?)"
    cursor = db_conn.cursor()
    # Also insert valid categories JSON to test multiple fields
    cursor.execute(insert_sql, (source_file, bad_json_authors, json.dumps(['cs.AI'])))
    paper_id = cursor.lastrowid
    db_conn.commit()

    # Fetch the row
    cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    row = cursor.fetchone()
    paper = database._parse_paper_row(row)
    assert paper is not None
    assert paper['source_filename'] == source_file
    assert paper['authors'] == [] # Should default to empty list on parse failure
    assert paper['categories'] == ['cs.AI']

def test_add_minimal_paper_success(db_conn):
    """Test adding a minimal paper record successfully."""
    source_filename = "test_minimal_paper.pdf"
    paper_id = database.add_minimal_paper(db_conn, source_filename)
    assert paper_id is not None
    assert isinstance(paper_id, int)

    # Verify the data was inserted correctly
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    row = cursor.fetchone()
    assert row is not None
    assert row['source_filename'] == source_filename
    assert row['status'] == 'processing' # Default status
    assert row['title'] is None # Should be null initially
    assert row['arxiv_id'] is None
    assert row['blob_path'] is None
    assert row['processed_timestamp'] is None
    assert row['added_date'] is not None
    assert row['updated_date'] is not None

def test_add_minimal_paper_invalid_filename(db_conn):
    """Test adding a minimal paper with invalid filename (e.g., None)."""
    paper_id = database.add_minimal_paper(db_conn, None)
    assert paper_id is None # Should fail gracefully
    paper_id = database.add_minimal_paper(db_conn, "") # Empty string
    assert paper_id is None # Should fail gracefully

def test_get_paper_by_id_found(db_conn, sample_paper_data_1):
    """Test retrieving an existing paper by its integer ID."""
    paper_id = database.add_paper(db_conn, sample_paper_data_1)
    assert paper_id is not None

    paper = database.get_paper_by_id(db_conn, paper_id)
    assert paper is not None
    assert paper['id'] == paper_id
    assert paper['arxiv_id'] == sample_paper_data_1['arxiv_id']
    assert paper['title'] == sample_paper_data_1['title']
    assert paper['authors'] == sample_paper_data_1['authors']

def test_get_paper_by_id_not_found(db_conn):
    """Test retrieving a non-existent paper by ID."""
    paper = database.get_paper_by_id(db_conn, 99999)
    assert paper is None

def test_update_paper_field_by_id_success(db_conn):
    """Test updating fields of an existing paper using its integer ID."""
    source_filename = "update_test.pdf"
    paper_id = database.add_minimal_paper(db_conn, source_filename)
    assert paper_id is not None

    paper_before = database.get_paper_by_id(db_conn, paper_id)
    assert paper_before['status'] == 'processing'
    original_updated_date = paper_before['updated_date']

    # Update status
    new_status = "complete"
    success = database.update_paper_field(db_conn, paper_id, 'status', new_status)
    assert success is True
    paper_after_status = database.get_paper_by_id(db_conn, paper_id)
    assert paper_after_status['status'] == new_status
    assert paper_after_status['updated_date'] >= original_updated_date

    # Update blob_path
    new_blob_path = f"paper_{paper_id}_text.txt"
    # Need a slight delay to ensure updated_date changes demonstrably if resolution is low
    import time; time.sleep(0.01) 
    success = database.update_paper_field(db_conn, paper_id, 'blob_path', new_blob_path)
    assert success is True
    paper_after_blob = database.get_paper_by_id(db_conn, paper_id)
    assert paper_after_blob['blob_path'] == new_blob_path
    assert paper_after_blob['updated_date'] > paper_after_status['updated_date']

    # Update processed_timestamp
    process_time = datetime.now(timezone.utc)
    # Need a slight delay to ensure updated_date changes
    import time; time.sleep(0.01) 
    success = database.update_paper_field(db_conn, paper_id, 'processed_timestamp', process_time)
    assert success is True
    paper_after_time = database.get_paper_by_id(db_conn, paper_id)
    assert paper_after_time['processed_timestamp'] is not None # Make sure it's not None
    # Ensure both are timezone-aware UTC before comparing
    expected_timestamp_utc = process_time # Already UTC
    fetched_timestamp = paper_after_time['processed_timestamp']
    # Ensure fetched timestamp is timezone-aware (it should be, due to converter)
    if fetched_timestamp.tzinfo is None:
         # If somehow it's naive, assume UTC based on storage logic
         fetched_timestamp_utc = fetched_timestamp.replace(tzinfo=timezone.utc)
    else:
        fetched_timestamp_utc = fetched_timestamp.astimezone(timezone.utc)
    # Allow a small delta for processing time differences
    assert abs(fetched_timestamp_utc - expected_timestamp_utc) < timedelta(seconds=1)

    # Update arxiv_id (which was initially None)
    new_arxiv_id = "2405.00001"
    success = database.update_paper_field(db_conn, paper_id, 'arxiv_id', new_arxiv_id)
    assert success is True
    paper_after_arxiv = database.get_paper_by_id(db_conn, paper_id)
    assert paper_after_arxiv['arxiv_id'] == new_arxiv_id

def test_update_paper_field_by_id_not_found(db_conn):
    """Test updating a field for a non-existent paper ID."""
    success = database.update_paper_field(db_conn, 99999, 'status', 'failed')
    assert success is False

def test_delete_paper_by_id_success(db_conn):
    """Test deleting an existing paper using its integer ID."""
    paper_id = database.add_minimal_paper(db_conn, "delete_me.pdf")
    assert paper_id is not None

    deleted = database.delete_paper(db_conn, paper_id)
    assert deleted is True

    # Verify the paper is gone
    paper = database.get_paper_by_id(db_conn, paper_id)
    assert paper is None

def test_delete_paper_by_id_not_found(db_conn):
    """Test deleting a paper by ID that does not exist."""
    deleted = database.delete_paper(db_conn, 99999)
    assert deleted is False

def test_add_paper_tz_date_success(db_conn, sample_paper_data_tz_date):
    """Test adding a paper with only source_filename and a tz-aware pub date."""
    paper_id = database.add_paper(db_conn, sample_paper_data_tz_date)
    assert paper_id is not None

    # Verify data
    cursor = db_conn.cursor()
    cursor.execute("SELECT source_filename, publication_date FROM papers WHERE id = ?", (paper_id,))
    row = cursor.fetchone()
    assert row is not None
    assert row['source_filename'] == sample_paper_data_tz_date['source_filename']
    
    retrieved_date = row['publication_date']
    assert retrieved_date is not None
    # Ensure comparison is between aware datetimes
    expected_date = sample_paper_data_tz_date['publication_date'] # Already aware
    if retrieved_date.tzinfo is None:
        retrieved_date = retrieved_date.replace(tzinfo=timezone.utc)
    else:
        retrieved_date = retrieved_date.astimezone(timezone.utc)
        
    assert abs(retrieved_date - expected_date) < timedelta(seconds=1)

# --- Test Helpers --- 

def test_parse_paper_row_bad_json(db_conn):
    """Test that _parse_paper_row handles invalid JSON gracefully."""
    # Manually insert a row with bad JSON in 'authors'
    bad_json_authors = '["Author One", "Author Two"' # Missing closing bracket
    source_file = "bad_json_test.pdf" # Need source_filename
    insert_sql = "INSERT INTO papers (source_filename, authors, categories) VALUES (?, ?, ?)"
    cursor = db_conn.cursor()
    # Also insert valid categories JSON to test multiple fields
    cursor.execute(insert_sql, (source_file, bad_json_authors, json.dumps(['cs.AI'])))
    paper_id = cursor.lastrowid
    db_conn.commit()

    # Fetch the row
    cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    row = cursor.fetchone()
    paper = database._parse_paper_row(row)
    assert paper is not None
    assert paper['source_filename'] == source_file
    assert paper['authors'] == [] # Should default to empty list on parse failure
    assert paper['categories'] == ['cs.AI']

# Example of how to run tests from the command line:
# Navigate to the project root directory (coding-agent)
# Ensure pytest is installed: pip install pytest
# Run: pytest tests/test_database.py
