import pytest
import anyio
import os
from pathlib import Path
import shutil
import uuid
# No longer need fitz here as we copy a static PDF
# import fitz # pymupdf 
# No longer need CliRunner
# from click.testing import CliRunner 
import functools # Import functools
import logging

# Import the NEW top-level async function and the adapter
from src.cli import _run_pipeline_async # Import the async function
from src.memory.hybrid_sqlite import HybridSQLiteAdapter

# --- Test Setup --- 

# Mark all tests in this file as anyio tests
pytestmark = pytest.mark.anyio

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("TestMinimalPipeline")

# Check if Gemini API key is available, skip tests if not
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
requires_gemini_api = pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason="Requires GEMINI_API_KEY environment variable to be set"
)

@pytest.fixture(scope="function") # Use function scope for clean state per test
def temp_pipeline_env(tmp_path: Path) -> dict:
    """Creates a temporary environment for a pipeline run.

    Includes: temp dir, db path, embedding dir, dummy PDF paths.
    Cleans up afterwards.
    """
    log.info(f"Creating temp env in: {tmp_path}")
    db_path = tmp_path / "test_memory.db"
    embedding_dir = tmp_path / "test_embeddings"
    pdf_dir = tmp_path / "test_pdfs"
    
    pdf_dir.mkdir()
    embedding_dir.mkdir()
    # Ensure db parent dir exists (it will be tmp_path)

    # --- Use a static sample PDF instead of generating dummies ---
    source_pdf_path = Path("tests/assets/sample.pdf") 
    if not source_pdf_path.exists():
        pytest.fail(f"Source test PDF not found at: {source_pdf_path}")
        
    target_pdf_path = pdf_dir / source_pdf_path.name
    log.info(f"Copying {source_pdf_path} to {target_pdf_path}")
    shutil.copy(source_pdf_path, target_pdf_path)
    
    pdf_paths = [str(target_pdf_path)]
    pdf_stems = [target_pdf_path.stem] # Should be "sample"
    # -----------------------------------------------------------
        
    yield {
        "tmp_path": tmp_path,
        "db_path": str(db_path),
        "embedding_dir": str(embedding_dir),
        "pdf_paths": pdf_paths, # List containing the single sample PDF path
        "pdf_stems": pdf_stems  # List containing the single stem "sample"
    }
    
    # Cleanup happens automatically with tmp_path fixture scope
    log.info(f"Cleaning up temp env: {tmp_path}")

# --- Test Case --- 

@requires_gemini_api # Skip this test if API key is not set
async def test_minimal_pipeline_end_to_end(temp_pipeline_env: dict):
    """Tests the minimal pipeline (Ingest->Summarize->Synthesize) via the CLI.
    
    Verifies that expected documents (chunks, summaries, synthesis) are created
    in the memory database.
    """
    db_path = temp_pipeline_env["db_path"]
    embedding_dir = temp_pipeline_env["embedding_dir"]
    pdf_paths = temp_pipeline_env["pdf_paths"]
    pdf_stems = temp_pipeline_env["pdf_stems"]
    max_concurrent = 1 # Run sequentially for test
    gemini_model = "gemini-1.5-flash-latest" # Use a specific model for test
    sample_pdf_path = pdf_paths[0] # Get the path for logging stem
    pdf_stem = Path(sample_pdf_path).stem
    
    log.info(f"Starting test for PDF: {sample_pdf_path}, DB: {db_path}")
    
    verifier_adapter = None # Define outside try for finally block
    try:
        # --- Run the pipeline --- 
        log.info(f"Running async pipeline function...")
        await _run_pipeline_async(
            pdf_files=tuple(pdf_paths), 
            db_path=db_path, 
            embedding_dir=embedding_dir, 
            max_concurrent=max_concurrent, 
            gemini_model=gemini_model, 
            api_key=GEMINI_API_KEY
        )
        log.info("Async pipeline function finished.")

        # --- Verification --- 
        # Give some time for file operations/DB writes to settle (optional, but can help)
        await anyio.sleep(0.5) 
        
        log.info("Verifying results in the database...")
        # Connect to the database AFTER the pipeline has run
        verifier_adapter = HybridSQLiteAdapter(
            db_path_str=db_path, 
            embedding_dir_str=embedding_dir
        )
        log.info("Verifier adapter connected.")
        
        # --- Verify chunks exist (without FTS) ---
        log.info("Performing non-FTS query to verify chunk insertion...")
        verify_tags = ["chunk", "pymupdf", pdf_stem]
        verify_func = functools.partial(
            verifier_adapter.query, 
            query_text="", # Empty query text bypasses FTS
            k=20, # Get up to 20 chunks
            filter_tags=verify_tags
        )
        verify_chunks = await anyio.to_thread.run_sync(verify_func)
        log.info(f"Non-FTS query found {len(verify_chunks)} chunks.")
        assert len(verify_chunks) > 0, f"Non-FTS query failed to find any chunks with tags: {verify_tags}"
        if verify_chunks:
            log.info(f"First chunk text (first 200 chars): {verify_chunks[0].text[:200]}...")
            # Check if 'sample' is in the first chunk text (case-insensitive)
            if "sample".lower() in verify_chunks[0].text.lower():
                log.info("'sample' FOUND in the first chunk text.")
            else:
                log.warning("'sample' NOT FOUND in the first chunk text.")
        
        # --- Check for chunks using FTS ---
        log.info(f"Querying for chunks with pdf_stem: {pdf_stem}")
        chunk_query_tags = ["chunk", "pymupdf", pdf_stem]
        
        # *** Perform diagnostic query ***
        log.info(f"Performing diagnostic FTS query for 'the' with tags: {chunk_query_tags}")
        query_func_the = functools.partial(
            verifier_adapter.query, 
            query_text="the", 
            k=10,
            filter_tags=chunk_query_tags
        )
        simple_fts_chunks = await anyio.to_thread.run_sync(query_func_the)
        log.info(f"Diagnostic FTS query for 'the' found {len(simple_fts_chunks)} chunks.")
        assert len(simple_fts_chunks) > 0, f"Diagnostic FTS query for 'the' failed for PDF stem: {pdf_stem}"

        # *** Perform original query for the PDF stem ('sample') ***
        # known_word = "Transformer" # Keep commented out
        log.info(f"Performing main query for '{pdf_stem}' with tags: {chunk_query_tags}")
        query_func_stem = functools.partial(
            verifier_adapter.query,
            query_text=pdf_stem, # Use pdf_stem ('sample') again
            k=10,
            filter_tags=chunk_query_tags
        )
        chunks = await anyio.to_thread.run_sync(query_func_stem)
        log.info(f"Main query for '{pdf_stem}' found {len(chunks)} chunks.")
        assert len(chunks) > 0, f"No chunks found for PDF stem: {pdf_stem}"
        log.info(f"Found {len(chunks)} chunks for '{pdf_stem}'. First chunk text: {chunks[0].text[:100]}...")

        # --- Check for summary ---
        log.info(f"Querying for summary for target source: {pdf_stem}")
        summary_query_tags = ["summary", pdf_stem] 
        # Use functools.partial
        query_func_summary = functools.partial(
            verifier_adapter.hybrid_query, # Use hybrid query here
            query_text=f"summary of {pdf_stem}", 
            k=1, 
            filter_tags=summary_query_tags
        )
        summaries = await anyio.to_thread.run_sync(query_func_summary)
        log.info(f"Hybrid query for summary found {len(summaries)} results.")
        assert len(summaries) == 1, f"Expected 1 summary for PDF stem: {pdf_stem}, found {len(summaries)}"
        assert summaries[0].text is not None and len(summaries[0].text) > 10, f"Summary text seems empty or too short for {pdf_stem}"
        log.info(f"Found summary for {pdf_stem}. Text: {summaries[0].text[:100]}...")
        
        # --- Check for synthesis ---
        log.info("Querying for synthesis document...")
        # Use functools.partial
        query_func_synthesis = functools.partial(
            verifier_adapter.hybrid_query, # Use hybrid query here
            query_text="overall synthesis", # More descriptive query
            k=1, 
            filter_tags=["synthesis"]
        )
        synthesis_docs = await anyio.to_thread.run_sync(query_func_synthesis)
        log.info(f"Hybrid query for synthesis found {len(synthesis_docs)} results.")
        assert len(synthesis_docs) == 1, f"Expected 1 synthesis document, found {len(synthesis_docs)}"
        assert synthesis_docs[0].text is not None and len(synthesis_docs[0].text) > 20, "Synthesis text seems empty or too short"
        log.info(f"Found synthesis document. Text: {synthesis_docs[0].text[:100]}...")

    except Exception as e:
        # Log the exception before failing
        log.error(f"Test failed with exception: {e}", exc_info=True)
        pytest.fail(f"Test failed: {e}")

    finally:
        # Ensure DB connection is closed even if test fails
        if verifier_adapter and verifier_adapter.conn:
             log.info("Closing verifier_adapter connection in finally block...")
             verifier_adapter.close()
        else:
            log.info("Verifier adapter connection not found or already closed in finally block.")
        
        # Clean up temporary files/dirs using pytest fixture's teardown implicitly
        log.info("Integration test finished cleanup.")

    log.info("Integration test passed.") 