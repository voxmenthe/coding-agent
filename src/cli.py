import click
import anyio
import logging
import time
import os
from pathlib import Path

from src.core.scheduler import TaskScheduler
from src.core.agents.ingestor import IngestorAgent
from src.core.agents.summarizer import SummarizerAgent
from src.core.agents.synthesizer import SynthesizerAgent
from src.memory.hybrid_sqlite import HybridSQLiteAdapter

# Configure basic logging for the CLI
# Increased level for CLI diagnostics
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("CLI") # More specific logger name

# Default paths - adjust as needed
DEFAULT_DB_PATH = "./data/memory.db"
DEFAULT_EMBEDDING_DIR = "./data/embeddings"
DEFAULT_MODEL = "gemini-1.5-flash-latest"

# --- Top-level Async Pipeline Function ---
async def _run_pipeline_async(pdf_files: tuple[str], db_path: str, embedding_dir: str, max_concurrent: int, gemini_model: str, api_key: str):
    """Asynchronous core logic for the minimal pipeline."""
    log.info("Entering _run_pipeline_async") # DEBUG -> INFO
    log.info(f"Received {len(pdf_files)} PDF files: {pdf_files}")
    log.info(f"DB Path: {db_path}, Embedding Dir: {embedding_dir}")
    log.info(f"Max Concurrency: {max_concurrent}, Gemini Model: {gemini_model}")

    # --- Ensure directories exist ---
    log.debug("Ensuring parent directories exist...")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(embedding_dir).mkdir(parents=True, exist_ok=True)
    log.debug("Directories ensured.")

    # --- Instantiate Shared Adapter ---
    log.debug("Instantiating shared HybridSQLiteAdapter...")
    shared_adapter = HybridSQLiteAdapter(
        db_path_str=db_path, 
        embedding_dir_str=embedding_dir
        # Note: We are not passing an embedding model here for the minimal pipeline
    )
    log.info("Shared HybridSQLiteAdapter instantiated.") # DEBUG -> INFO

    # --- Instantiate Scheduler ---
    log.debug(f"Instantiating TaskScheduler with max_concurrent_tasks={max_concurrent}...")
    scheduler = TaskScheduler(max_concurrent_tasks=max_concurrent)
    log.info("TaskScheduler instantiated.") # DEBUG -> INFO

    ingestor_agents = [] # Changed name for clarity
    pdf_stems = [] # Store stems for targeting summarizers

    # --- Create and Schedule Ingestor Agents ---
    log.info("--- Phase 1: Ingestion --- Creating Ingestor Agents...")
    for i, pdf_path_str in enumerate(pdf_files):
        pdf_path = Path(pdf_path_str)
        pdf_stem = pdf_path.stem # Use filename stem as source identifier
        pdf_stems.append(pdf_stem)
        log.debug(f"Creating IngestorAgent for PDF: {pdf_path_str} (stem: {pdf_stem})")
        
        ingestor_config = {
            "pdf_paths": [str(pdf_path)] # Pass the single PDF path
        }
        agent = IngestorAgent(
            agent_id=f"ingestor_{pdf_stem}",
            adapter=shared_adapter, # Use the shared adapter instance
            ingestion_strategy="pymupdf", # Use default minimal strategy
            **ingestor_config 
        )
        ingestor_agents.append(agent) # Add to ingestor list
        log.debug(f"IngestorAgent {agent.agent_id} created.")

    log.info(f"Adding {len(ingestor_agents)} IngestorAgent tasks to scheduler...")
    for agent in ingestor_agents:
         log.debug(f"Adding task: {agent.agent_id}")
         scheduler.add_task(agent)
    log.info("IngestorAgent tasks added.")
    
    # Run Ingestion Phase
    log.info("Running Ingestion Phase via scheduler...")
    await scheduler.run() 
    log.info("--- Ingestion phase complete. --- SCHEDULER FINISHED")
    
    # --- Create and Schedule Summarizer Agents ---
    log.info("--- Phase 2: Summarization --- Creating Summarizer Agents...")
    summarizer_agents = []
    for pdf_stem in pdf_stems:
        log.debug(f"Creating SummarizerAgent for stem: {pdf_stem}")
        summarizer_config = {
            "target_source": pdf_stem # Tell summarizer which PDF's chunks to target
        }
        agent = SummarizerAgent(
            agent_id=f"summarizer_{pdf_stem}",
            adapter=shared_adapter, # Use the shared adapter instance
            gemini_api_key=api_key,
            gemini_model=gemini_model,
            **summarizer_config 
        )
        summarizer_agents.append(agent)
        log.debug(f"SummarizerAgent {agent.agent_id} created.")

    log.info(f"Clearing scheduler and adding {len(summarizer_agents)} SummarizerAgent tasks...")
    scheduler.tasks = [] # Reset tasks in the scheduler
    for agent in summarizer_agents:
        log.debug(f"Adding task: {agent.agent_id}")
        scheduler.add_task(agent)
    log.info("SummarizerAgent tasks added.")

    # Run Summarization Phase
    log.info("Running Summarization Phase via scheduler...")
    await scheduler.run()
    log.info("--- Summarization phase complete. --- SCHEDULER FINISHED")
    
    # --- Create and Schedule Synthesizer Agent ---
    log.info("--- Phase 3: Synthesis --- Creating Synthesizer Agent...")
    log.debug("Creating SynthesizerAgent...")
    synthesizer_agent = SynthesizerAgent(
        agent_id="synthesizer_01",
        adapter=shared_adapter, # Use the shared adapter instance
        gemini_api_key=api_key,
        gemini_model=gemini_model
        # No specific config needed for minimal version
    )
    log.debug(f"SynthesizerAgent {synthesizer_agent.agent_id} created.")

    log.info("Clearing scheduler and adding SynthesizerAgent task...")
    scheduler.tasks = []
    log.debug(f"Adding task: {synthesizer_agent.agent_id}")
    scheduler.add_task(synthesizer_agent)
    log.info("SynthesizerAgent task added.")
    
    # Run Synthesis Phase
    log.info("Running Synthesis Phase via scheduler...")
    await scheduler.run()
    log.info("--- Synthesis phase complete. --- SCHEDULER FINISHED")

    # --- Close shared adapter ---
    log.info("Attempting to close shared database adapter...")
    try:
        shared_adapter.close()
        log.info("Shared database adapter closed successfully.")
    except Exception as e:
        log.warning(f"Error closing shared database adapter: {e}")
    
    log.info("Exiting _run_pipeline_async")

# --- CLI Command Definition ---

@click.group()
def cli():
    """Minimal Multi-Agent Research Synthesizer CLI (v2)."""
    log.debug("CLI group function invoked.")
    pass

@cli.command()
@click.argument('pdf_files', nargs=-1, type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option('--db-path', default=DEFAULT_DB_PATH, type=click.Path(), help='Path to the SQLite database file.')
@click.option('--embedding-dir', default=DEFAULT_EMBEDDING_DIR, type=click.Path(), help='Directory to store embedding files.')
@click.option('--max-concurrent', default=3, type=int, help='Maximum number of agents to run concurrently.')
@click.option('--gemini-model', default=DEFAULT_MODEL, type=str, help='Gemini model name for Summarizer/Synthesizer.')
@click.option('--api-key', envvar='GEMINI_API_KEY', help='Gemini API Key (can also be set via GEMINI_API_KEY env var).')
def run_minimal_pipeline(pdf_files: tuple[str], db_path: str, embedding_dir: str, max_concurrent: int, gemini_model: str, api_key: str):
    """Runs the minimal agent pipeline (Ingestor -> Summarizer -> Synthesizer) on PDF files.

    Requires GEMINI_API_KEY environment variable or --api-key option.

    Example:
        python -m src.cli run-minimal-pipeline path/to/doc1.pdf path/to/doc2.pdf --max-concurrent 2
    """
    log.info(f"run_minimal_pipeline command invoked with {len(pdf_files)} files.")
    log.debug(f"Args: pdf_files={pdf_files}, db_path='{db_path}', embedding_dir='{embedding_dir}', max_concurrent={max_concurrent}, gemini_model='{gemini_model}', api_key_present={bool(api_key)}")

    if not pdf_files:
        log.error("No PDF files provided. Exiting.")
        return

    if not api_key:
        log.error("Gemini API Key not provided. Exiting.")
        return
    
    log.info("Starting pipeline execution...")
    start_time = time.monotonic()
    
    # --- Run the async pipeline --- 
    try:
        log.debug("Calling anyio.run on _run_pipeline_async...")
        anyio.run(
            _run_pipeline_async, 
            pdf_files, 
            db_path, 
            embedding_dir, 
            max_concurrent, 
            gemini_model, 
            api_key
        )
        log.info("anyio.run completed successfully.")
    except Exception as e:
        log.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)
    finally:
        # Ensure the shared adapter connection is closed after pipeline execution
        if shared_adapter: # Check if it exists
            log.info("Closing shared adapter connection in pipeline finally block...")
            shared_adapter.close()
            log.info("Shared adapter connection closed.")
        else:
             log.warning("Shared adapter was not initialized, cannot close.")
    
    end_time = time.monotonic()
    log.info(f"Minimal pipeline finished in {end_time - start_time:.2f} seconds.")
    log.info("run_minimal_pipeline command finished.")

if __name__ == '__main__':
    log.debug("CLI script execution started (__name__ == '__main__').")
    cli()
    log.debug("CLI script execution finished.") 