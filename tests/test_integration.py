import pytest
import pypdf
import sqlite3
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Assuming your project structure allows these imports
from src.main import CodeAgent, load_config
from src import database

# Constants for the test PDF
TEST_PDF_FILENAME = 'AProbabilisticInferenceApproachtoInference-TimeScalingofLLMsusingParticle-BasedMonteCarloMethods2502.01618v3.pdf'
TEST_PDF_FILENAME_NON_ARXIV = 'PhysicsofLanguageModelsPart4.1,ArchitectureDesignandtheMagicofCanonLayers.pdf'
MOCKED_EXTRACTED_TEXT = "This is the mocked extracted text from the PDF."

@pytest.fixture
def test_environment(tmp_path: Path):
    """Creates a temporary environment for integration tests."""
    # Define temporary paths
    temp_pdf_dir = tmp_path / "PDFS"
    temp_blob_dir = tmp_path / "paper_blobs"
    temp_db_path = tmp_path / "test_paper_database.db"

    # Create directories
    temp_pdf_dir.mkdir()
    temp_blob_dir.mkdir()

    # Copy all PDFs from the test PDF directory to the temporary PDF directory
    source_pdf_dir = Path(__file__).parent.parent / "PDFS"
    for pdf_file in source_pdf_dir.glob("*.pdf"):
        shutil.copy(pdf_file, temp_pdf_dir / pdf_file.name)

    # Initialize the database schema
    conn = database.get_db_connection(temp_db_path)
    assert conn is not None
    database.create_tables(conn)
    conn.close()

    # Configure and create the agent instance
    # Mock load_config if necessary, or ensure defaults point to tmp paths
    # For simplicity, we'll instantiate directly with temp paths
    # NOTE: Assumes CodeAgent doesn't strictly require a real API key for this test path
    agent = CodeAgent(
        model_name="gemini-test-model", # Use a dummy model name
        verbose=False,
        api_key="DUMMY_API_KEY", # Provide a dummy key
        default_thinking_budget=100,
        pdf_dir=str(temp_pdf_dir),
        db_path=str(temp_db_path),
        blob_dir=str(temp_blob_dir)
    )
    # Prevent actual client configuration if not needed
    agent.client = MagicMock()
    agent.chat = MagicMock()

    return agent, temp_pdf_dir, temp_blob_dir, temp_db_path


# --- Test Cases --- #

@patch('pypdf.PdfReader') # Correct patch target for the library
def test_handle_pdf_basic(mock_pdf_reader, test_environment):
    """Test processing a PDF via _handle_pdf_command without arxiv_id."""
    agent, temp_pdf_dir, temp_blob_dir, temp_db_path = test_environment

    # Configure the mock PdfReader
    mock_page = MagicMock()
    mock_page.extract_text.return_value = MOCKED_EXTRACTED_TEXT
    mock_instance = MagicMock()
    mock_instance.pages = [mock_page]
    mock_pdf_reader.return_value = mock_instance

    # --- Act ---
    test_pdf_path = temp_pdf_dir / TEST_PDF_FILENAME # Still need the path for setup
    paper_id = agent._handle_pdf_command([TEST_PDF_FILENAME])

    # --- Assert --- 
    assert paper_id is not None, "_handle_pdf_command should return a paper_id on success"
    assert isinstance(paper_id, int)

    # Check database
    conn = database.get_db_connection(temp_db_path)
    assert conn is not None
    paper = database.get_paper_by_id(conn, paper_id)
    conn.close()

    assert paper is not None
    assert paper['id'] == paper_id # Should be the first paper added
    assert paper['source_filename'] == TEST_PDF_FILENAME
    # The current logic sets status to 'pending' after blob save - NO, it sets to 'complete'
    # assert paper['status'] == 'pending'
    assert paper['status'] == 'complete' # Expect 'complete' after successful run
    assert paper['arxiv_id'] is None # No arxiv_id provided
    assert paper['blob_path'] == f"paper_{paper_id}_text.txt" # Use captured paper_id

    # Check blob file
    blob_file_path = temp_blob_dir / f"paper_{paper_id}_text.txt" # Use captured paper_id
    assert blob_file_path.is_file()
    assert blob_file_path.read_text() == MOCKED_EXTRACTED_TEXT + "\n" # Expect newline due to extraction loop


@patch('pypdf.PdfReader') # Correct patch target for the library
def test_handle_pdf_non_arxiv_filename(mock_pdf_reader, test_environment):
    """Test processing a non-arXiv-named PDF via _handle_pdf_command without arxiv_id."""
    agent, temp_pdf_dir, temp_blob_dir, temp_db_path = test_environment

    # Configure the mock PdfReader
    mock_page = MagicMock()
    mock_page.extract_text.return_value = MOCKED_EXTRACTED_TEXT
    mock_instance = MagicMock()
    mock_instance.pages = [mock_page]
    mock_pdf_reader.return_value = mock_instance

    # --- Act ---
    test_pdf_path = temp_pdf_dir / TEST_PDF_FILENAME_NON_ARXIV # Use non-arXiv filename
    paper_id = agent._handle_pdf_command([TEST_PDF_FILENAME_NON_ARXIV]) # Pass non-arXiv filename

    # --- Assert ---
    assert paper_id is not None, "_handle_pdf_command should return a paper_id on success"
    assert isinstance(paper_id, int)

    # Check database
    conn = database.get_db_connection(temp_db_path)
    assert conn is not None
    paper = database.get_paper_by_id(conn, paper_id)
    conn.close()

    assert paper is not None
    assert paper['id'] == paper_id
    assert paper['source_filename'] == TEST_PDF_FILENAME_NON_ARXIV # Check correct filename stored
    assert paper['status'] == 'complete'
    assert paper['arxiv_id'] is None # No arxiv_id provided or inferred
    assert paper['blob_path'] == f"paper_{paper_id}_text.txt"

    # Check blob file
    blob_file_path = temp_blob_dir / f"paper_{paper_id}_text.txt"
    assert blob_file_path.is_file()
    assert blob_file_path.read_text() == MOCKED_EXTRACTED_TEXT + "\n" # Expect newline


@patch('pypdf.PdfReader') # Correct patch target for the library
def test_handle_pdf_with_arxiv_id(mock_pdf_reader, test_environment):
    """Test processing a PDF via _handle_pdf_command with arxiv_id."""
    agent, temp_pdf_dir, temp_blob_dir, temp_db_path = test_environment
    test_arxiv_id = "2502.01618v3" # Extracted from filename for realism

    # Configure the mock PdfReader (same as basic test)
    mock_page = MagicMock()
    mock_page.extract_text.return_value = MOCKED_EXTRACTED_TEXT
    mock_instance = MagicMock()
    mock_instance.pages = [mock_page]
    mock_pdf_reader.return_value = mock_instance

    # --- Act ---
    test_pdf_path = temp_pdf_dir / TEST_PDF_FILENAME # Still need the path for setup
    paper_id = agent._handle_pdf_command([TEST_PDF_FILENAME, test_arxiv_id])

    # --- Assert ---
    assert paper_id is not None, "_handle_pdf_command should return a paper_id on success"
    assert isinstance(paper_id, int)
    
    # Check database
    conn = database.get_db_connection(temp_db_path)
    assert conn is not None
    paper = database.get_paper_by_id(conn, paper_id) # Use get_paper_by_id
    conn.close()

    assert paper is not None
    assert paper['id'] == paper_id # Should be the first paper added
    assert paper['source_filename'] == TEST_PDF_FILENAME
    # assert paper['status'] == 'pending'
    assert paper['status'] == 'complete' # Expect 'complete' after successful run
    assert paper['arxiv_id'] == test_arxiv_id # Verify provided arxiv_id was saved
    assert paper['blob_path'] == f"paper_{paper_id}_text.txt" # Use captured paper_id

    # Check blob file
    blob_file_path = temp_blob_dir / f"paper_{paper_id}_text.txt" # Use captured paper_id
    assert blob_file_path.is_file()
    assert blob_file_path.read_text() == MOCKED_EXTRACTED_TEXT + "\n" # Expect newline due to extraction loop
