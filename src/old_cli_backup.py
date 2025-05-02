import click
import anyio
import logging
import time

from src.core.scheduler import TaskScheduler
from src.core.agents.ingestor import IngestorAgent
# Import other agents as they are created
# from src.core.agents.summarizer import SummarizerAgent
# from src.core.agents.synthesizer import SynthesizerAgent
# from src.core.agents.critic import CriticAgent

# Configure basic logging for the CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

@click.group()
def cli():
    """Multi-Agent Research Synthesizer CLI."""
    pass

@cli.command()
@click.argument('pdf_files', nargs=-1, type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option('--max-concurrent', default=3, type=int, help='Maximum number of agents to run concurrently.')
@click.option('--timeout', default=120.0, type=float, help='Timeout in seconds for each agent step.')
def run_pipeline(pdf_files: tuple[str], max_concurrent: int, timeout: float):
    """Runs the full agent pipeline on the specified PDF files.
    
    Example: python -m src.cli run-pipeline path/to/doc1.pdf path/to/doc2.pdf
    """
    if not pdf_files:
        log.error("No PDF files provided. Use --help for usage.")
        return

    log.info(f"Starting agent pipeline for {len(pdf_files)} PDF files...")
    log.info(f"Max Concurrency: {max_concurrent}, Step Timeout: {timeout}s")
    
    start_time = time.monotonic()

    # --- Instantiate Agents --- 
    # For now, only the Ingestor is defined
    # Create one ingestor agent per PDF file
    ingestor_agents = [
        IngestorAgent(agent_id=f"ingestor_{i}", chunk_size=1500, chunk_overlap=150)
        for i in range(len(pdf_files))
    ]
    
    # Add other agents once implemented
    # summarizer_agents = [SummarizerAgent(agent_id=f"summarizer_{i}") for i in ...]
    # synthesizer_agent = SynthesizerAgent(agent_id="synthesizer_01")
    # critic_agent = CriticAgent(agent_id="critic_01")

    # Combine all agents into a list for the scheduler
    # The pipeline structure (which agents run when) needs refinement.
    # For now, just run ingestors concurrently.
    all_agents = ingestor_agents # Add other agents here later

    # --- Instantiate Scheduler --- 
    scheduler = TaskScheduler(max_concurrent=max_concurrent, timeout_s=timeout)

    # --- Define the async function to run the pipeline --- 
    async def pipeline_task():
        # Map ingestor agents to their respective files
        # This simple approach assumes one ingestor per file.
        # More complex pipelines might need different scheduling logic.
        tasks = []
        async with anyio.create_task_group() as tg:
            for i, agent in enumerate(ingestor_agents):
                pdf_file = pdf_files[i]
                # Using start_soon to run the agent's run_step
                # We assume run_step is synchronous and needs to be wrapped by _run_agent_step
                log.info(f"Scheduling {agent.name} [{agent.agent_id}] for file: {pdf_file}")
                # Pass the PDF path to the agent's run_step method via the scheduler
                # Modifying scheduler to accept args/kwargs for the agent step might be cleaner.
                # For now, let's modify the agent list passed to scheduler.run_pipeline if needed
                # Or, adjust _run_agent_step to accept args.
                
                # Simplest approach for now: run_pipeline expects agent instances. 
                # We need a way to tell _run_agent_step *which* file to process.
                # Hacky approach: set an attribute on the agent instance right before scheduling?
                # agent.target_pdf = pdf_file # Not ideal
                
                # Let's assume run_step can take the file path directly.
                # The current TaskScheduler._run_agent_step calls agent.run_step() with no args.
                # We need to adjust TaskScheduler or how agents are called. 
                
                # --- TEMPORARY WORKAROUND --- 
                # Modify agent's run_step to accept the file path from args/kwargs
                # passed through anyio.to_thread.run_sync
                # Let's assume TaskScheduler._run_agent_step is updated to handle this:
                # e.g., await anyio.to_thread.run_sync(agent.run_step, pdf_file)
                
                # For now, just schedule the agent - the actual call needs adjustment
                tg.start_soon(scheduler._run_agent_step, agent, pdf_file) # Pass pdf_file as arg

        # After all ingestors finish, could schedule next steps (summarizers etc.)
        log.info("Ingestion phase complete.")
        # Add calls for summarizers, synthesizer, critic here
        
        # Example (if other agents were ready):
        # log.info("Starting summarization phase...")
        # await scheduler.run_pipeline(summarizer_agents)
        # log.info("Summarization complete. Starting synthesis...")
        # await scheduler.run_pipeline([synthesizer_agent])
        # ... and so on

    # --- Run the async pipeline --- 
    try:
        anyio.run(pipeline_task)
    except Exception as e:
        log.error(f"An error occurred during the pipeline execution: {e}", exc_info=True)
    finally:
        # Ensure memory service adapter is closed
        from src.memory.service import MemoryService
        MemoryService().close_adapter()
        log.info("Memory service adapter closed.")

    end_time = time.monotonic()
    log.info(f"Pipeline finished in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    cli()
