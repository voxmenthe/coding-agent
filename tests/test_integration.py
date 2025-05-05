import pytest
import sqlite3
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import os
import logging

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


@pytest.fixture
def manual_test_environment(tmp_path: Path):
    """Creates a temporary environment for MANUAL integration tests requiring REAL API access."""
    # Check for API key first
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY environment variable not set, skipping manual test.")

    # Define temporary paths
    temp_pdf_dir = tmp_path / "PDFS_manual"
    temp_blob_dir = tmp_path / "paper_blobs_manual"
    temp_db_path = tmp_path / "test_paper_database_manual.db"

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

    # Configure and create the agent instance with REAL API Key
    agent = CodeAgent(
        model_name="gemini-1.5-flash", # Use a real model capable of file processing
        verbose=True, # Enable verbose for manual test debugging
        api_key=api_key, # Pass the real API key
        default_thinking_budget=100,
        pdf_dir=str(temp_pdf_dir),
        db_path=str(temp_db_path),
        blob_dir=str(temp_blob_dir)
    )
    # DO NOT mock agent.client or agent.chat here

    return agent, temp_pdf_dir, temp_blob_dir, temp_db_path


# --- Test Cases --- #

@patch('src.tools.extract_text_from_pdf_gemini') # Mock the new tool function
def test_handle_pdf_basic(mock_extract_text, test_environment): # Add mock_extract_text arg
    """Test processing an arXiv-named PDF via _handle_pdf_command without arxiv_id."""
    agent, temp_pdf_dir, temp_blob_dir, temp_db_path = test_environment

    # Configure the mock for the Gemini tool
    mock_extract_text.return_value = MOCKED_EXTRACTED_TEXT

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
    assert paper['id'] == paper_id
    assert paper['source_filename'] == TEST_PDF_FILENAME
    assert paper['status'] == 'complete' # Expect 'complete' after successful run
    assert paper['arxiv_id'] is None # No arxiv_id provided
    assert paper['blob_path'] == f"paper_{paper_id}_text.txt"

    # Check blob file
    blob_file_path = temp_blob_dir / f"paper_{paper_id}_text.txt"
    assert blob_file_path.is_file()
    assert blob_file_path.read_text() == MOCKED_EXTRACTED_TEXT # Expect exact match

@patch('src.tools.extract_text_from_pdf_gemini') # Mock the new tool function
def test_handle_pdf_non_arxiv_filename(mock_extract_text, test_environment): # Add mock_extract_text arg
    """Test processing a non-arXiv-named PDF via _handle_pdf_command without arxiv_id."""
    agent, temp_pdf_dir, temp_blob_dir, temp_db_path = test_environment

    # Configure the mock for the Gemini tool
    mock_extract_text.return_value = MOCKED_EXTRACTED_TEXT

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
    assert blob_file_path.read_text() == MOCKED_EXTRACTED_TEXT # Expect exact match

@patch('src.tools.extract_text_from_pdf_gemini') # Mock the new tool function
def test_handle_pdf_with_arxiv_id(mock_extract_text, test_environment): # Add mock_extract_text arg
    """Test processing a PDF via _handle_pdf_command with arxiv_id."""
    agent, temp_pdf_dir, temp_blob_dir, temp_db_path = test_environment
    test_arxiv_id = "2502.01618v3" # Extracted from filename for realism

    # Configure the mock for the Gemini tool
    mock_extract_text.return_value = MOCKED_EXTRACTED_TEXT

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
    assert paper['id'] == paper_id
    assert paper['source_filename'] == TEST_PDF_FILENAME
    assert paper['status'] == 'complete' # Expect 'complete' after successful run
    assert paper['arxiv_id'] == test_arxiv_id # Verify provided arxiv_id was saved
    assert paper['blob_path'] == f"paper_{paper_id}_text.txt"

    # Check blob file
    blob_file_path = temp_blob_dir / f"paper_{paper_id}_text.txt"
    assert blob_file_path.is_file()
    assert blob_file_path.read_text() == MOCKED_EXTRACTED_TEXT # Expect exact match


# --- Manual Test for Real API Call --- #

# NOTE: This function IS prefixed with 'test_' but marked with 'manual'.
#       It will NOT be run by default pytest runs unless explicitly selected with '-m manual'.
#       It requires a valid GOOGLE_API_KEY environment variable to be set.
#       Run manually using: pytest -m manual -v -s
@pytest.mark.manual
def test_manual_integration_gemini_pdf_processing(manual_test_environment):
    """Manually test the full PDF processing pipeline with a real Gemini API call."""
    agent, temp_pdf_dir, temp_blob_dir, temp_db_path = manual_test_environment

    print("\n\n>>> Running MANUAL Gemini PDF Test <<<")
    print(f"Using PDF: {TEST_PDF_FILENAME}")
    print(f"DB Path: {temp_db_path}")
    print(f"Blob Dir: {temp_blob_dir}")

    # --- Act ---
    # Ensure the agent uses the Gemini method for this test
    agent.pdf_processing_method = "gemini"
    logger = logging.getLogger(__name__)
    logger.info(f"Set agent pdf_processing_method to: {agent.pdf_processing_method}")

    # Call the command without mocking the extraction tool
    paper_id = None
    try:
        paper_id = agent._handle_pdf_command([TEST_PDF_FILENAME])
        print(f"\n>>> _handle_pdf_command returned paper_id: {paper_id}")
    except Exception as e:
        print(f"\n>>> ERROR during _handle_pdf_command: {e}")
        pytest.fail(f"Error during real Gemini API call: {e}")

    # --- Assert --- #
    assert paper_id is not None, "_handle_pdf_command should return a paper_id on success"
    assert isinstance(paper_id, int)

    # Check database
    conn = database.get_db_connection(temp_db_path)
    assert conn is not None
    paper = database.get_paper_by_id(conn, paper_id)
    conn.close()
    print(f"\n>>> Database record for paper_id {paper_id}: {paper}")

    assert paper is not None, f"Paper with ID {paper_id} not found in database."
    assert paper['id'] == paper_id
    assert paper['source_filename'] == TEST_PDF_FILENAME
    assert paper['status'] == 'complete', f"Paper status is '{paper['status']}', expected 'complete'."
    assert paper['blob_path'] == f"paper_{paper_id}_text.txt"

    # Check blob file
    blob_file_path = temp_blob_dir / f"paper_{paper_id}_text.txt"
    print(f"\n>>> Checking blob file: {blob_file_path}")
    assert blob_file_path.is_file(), f"Blob file not found at {blob_file_path}"

    blob_content = blob_file_path.read_text()
    print(f"\n>>> Blob file content (first 500 chars):\n{blob_content[:500]}...")
    assert blob_content is not None, "Blob file content is None."
    assert len(blob_content.strip()) > 0, "Blob file content is empty or whitespace only."

    print("\n>>> MANUAL Gemini PDF Test Completed Successfully <<<")
