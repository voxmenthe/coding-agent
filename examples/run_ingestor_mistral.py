import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Adjust the import path based on your project structure
from src.core.agents.ingestor import IngestorAgent
from src.memory.service import MemoryService  # Assuming MemoryService is used

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Load environment variables from .env file if it exists
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Assumes the script is run from the root of the project
PDF_PATH = Path("tests/assets/sample.pdf")
AGENT_ID = "mistral_example_agent"


async def main():
    """Runs the IngestorAgent with the Mistral OCR strategy."""
    if not MISTRAL_API_KEY:
        logging.error("MISTRAL_API_KEY environment variable not set. Cannot run Mistral example.")
        logging.info("Please set the MISTRAL_API_KEY in your environment or a .env file.")
        return

    if not PDF_PATH.is_file():
        logging.error(f"PDF file not found at: {PDF_PATH.resolve()}")
        return

    logging.info(f"Initializing IngestorAgent (Mistral OCR) for: {PDF_PATH}")

    # Initialize MemoryService (using default in-memory for this example)
    memory_service = MemoryService()

    agent = IngestorAgent(
        agent_id=AGENT_ID,
        ingestion_strategy="mistral_ocr",
        pdf_paths=[str(PDF_PATH)],
        memory_service=memory_service,  # Pass the memory service instance
        mistral_api_key=MISTRAL_API_KEY # Pass the API key
    )

    logging.info("Starting agent run...")
    await agent.run()
    logging.info("Agent run finished.")

    # Optional: Query memory to see what was added
    try:
        # Use the correct synchronous method name: query_memory
        results = memory_service.query_memory("What is the document about?", k=1)
        if results:
            logging.info(f"Query result snippet: {results[0].text[:200]}...")
            logging.info(f"Metadata: {results[0].metadata}")
        else:
            logging.info("No results found in memory for the query.")
    except Exception as e:
        logging.error(f"Error querying memory: {e}")

if __name__ == "__main__":
    # Ensure the script can find the 'src' directory if run from 'examples'
    # This adds the project root to the Python path
    project_root = Path(__file__).parent.parent
    import sys
    sys.path.insert(0, str(project_root))

    asyncio.run(main())
