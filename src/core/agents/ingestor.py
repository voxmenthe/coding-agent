import logging
import fitz
from pathlib import Path
import uuid
import os
import asyncio
import aiofiles
# Use unified client
from mistralai import Mistral

import pymupdf # fitz

# Local imports
from src.memory.service import MemoryService
from src.memory.adapter import MemoryDoc
from src.config import config
from .base import BaseAgent
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_INGESTION_STRATEGY = "pymupdf"

MISTRAL_OCR_DEFAULT_MODEL = "mistral-ocr-latest"

logger = logging.getLogger(__name__)

class IngestorAgent(BaseAgent):
    """Agent responsible for reading documents (e.g., PDFs), extracting text
       using a configured strategy (e.g., Mistral OCR, pymupdf), chunking it,
       and adding it to the memory system.
    """

    def __init__(self,
                 agent_id: str,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 ingestion_strategy: str = DEFAULT_INGESTION_STRATEGY,
                 **kwargs):
        """Initializes the Ingestor Agent.

        Args:
            agent_id: Unique ID for this agent instance.
            chunk_size: Target size for text chunks (in characters).
            chunk_overlap: Character overlap between consecutive chunks.
            ingestion_strategy: Strategy for document ingestion (e.g., "mistral_ocr" or "pymupdf").
            **kwargs: Additional configuration passed to BaseAgent (e.g., pdf_paths).
        """
        super().__init__(agent_id=agent_id, **kwargs)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.memory_service = MemoryService() # Get the singleton instance
        self.ingestion_strategy = ingestion_strategy
        self.mistral_client: Optional[Mistral] = None # Use unified client type
        self.mistral_model = MISTRAL_OCR_DEFAULT_MODEL
        # Initialize clients based on strategy
        if self.ingestion_strategy == "mistral_ocr":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                log.error("MISTRAL_API_KEY environment variable not set. Mistral OCR will fail.")
                # No need for else block, keep mistral_client as None
            else:
                try:
                    # Initialize the client using the API key
                    # Initialize unified client
                    self.mistral_client = Mistral(api_key=api_key)
                    log.info(f"Mistral client initialized successfully (model={self.mistral_model}).") 
                except Exception as e:
                    log.error(f"Failed to initialize Mistral client: {e}", exc_info=True)
                    self.mistral_client = None # Ensure it's None on failure
        elif self.ingestion_strategy == "pymupdf":
            log.info("Using pymupdf strategy. No specific client initialization needed.")
        # Add elif blocks here for other strategies like 'gemini_native' later
        else:
            log.warning(f"Unknown ingestion strategy '{self.ingestion_strategy}' specified during init.")

        log.info(f"IngestorAgent [{self.agent_id}] initialized (strategy={self.ingestion_strategy}, chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

    def _chunk_text(self, text: str, source_ref: str) -> list[str]:
        """Simple text chunking based on character count with overlap."""
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            next_start = end - self.chunk_overlap
            # Ensure progress if overlap is large or chunk_size is small
            if next_start <= start:
                 # Move to the position right after the current chunk if no progress would be made
                start = end
            else:
                start = next_start
            # Break if we've reached the end
            if end == len(text):
                break
        log.debug(f"Chunked text from '{source_ref}' into {len(chunks)} chunks.")
        return chunks

    async def _process_pdf_mistral(self, pdf_path: Path) -> Tuple[str, int]:
        """
        Processes a single PDF using the Mistral OCR API (asynchronous approach).

        Uploads file, gets signed URL, calls ocr.process, deletes file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A tuple containing the extracted text (str) and the page count (int, currently hardcoded to 0).
        """
        if not self.mistral_client:
            logger.error("Mistral client not initialized. Cannot process with mistral_ocr.")
            return "", 0

        file_id = None
        uploaded_file_obj = None # Keep track of the uploaded file object
        try:
            # 1. Upload the file asynchronously using the async client
            logger.info(f"Uploading file {pdf_path.name} to Mistral...")
            # Use upload_async with correct dictionary format and synchronous open()
            with open(pdf_path, mode='rb') as pdf_file:
                uploaded_file_obj = await self.mistral_client.files.upload_async(
                    file={
                        "file_name": pdf_path.name, 
                        "content": pdf_file # Pass synchronous file handle
                    },
                    purpose='ocr' # Use string literal 'ocr'
                )
            file_id = uploaded_file_obj.id
            logger.info(f"File {pdf_path.name} uploaded successfully with ID: {file_id}")

            # 2. Get Signed URL - Use direct await
            logger.info(f"Getting signed URL for file ID: {file_id}...")
            # Use get_signed_url_async
            signed_url_obj = await self.mistral_client.files.get_signed_url_async(
                file_id=file_id
            )
            signed_url = signed_url_obj.url
            logger.info(f"Obtained signed URL for file ID: {file_id}")

            # 3. Process OCR using the signed URL - Use direct await
            logger.info(f"Processing OCR for signed URL (file ID: {file_id})...")
            # Use process_async with correct 'document' dictionary structure
            ocr_result = await self.mistral_client.ocr.process_async(
                model=self.mistral_model, 
                document={
                    "type": "document_url",
                    "document_url": signed_url
                }
            ) 
            logger.info(f"OCR Result type: {type(ocr_result)}")
            logger.info(f"OCR Result dir: {dir(ocr_result)}")
            # Iterate through pages and join markdown content
            full_text = "\n".join([page.markdown for page in ocr_result.pages])
            # Calculate page count based on the number of pages in the response
            page_count = len(ocr_result.pages)
            logger.info(f"Successfully extracted text from {page_count} pages using Mistral OCR.")
        except Exception as e: # Catch generic exception
            logger.error(
                f"Error during Mistral API call for {pdf_path.name}"
                f"{' (File ID: ' + uploaded_file_obj.id + ')' if uploaded_file_obj else ''}: {e}",
                exc_info=True # Add full traceback to log for debugging
            )
            full_text = ""
            page_count = 0
        except Exception as e:
            # Catch any other unexpected errors during processing
            logger.error(f"Unexpected error during Mistral processing for {pdf_path.name}: {e}", exc_info=True)
            full_text = ""
            page_count = 0

        finally:
            # 6. Delete the uploaded file - Use direct await
            if file_id:
                logger.info(f"Attempting to delete uploaded Mistral file {file_id}...")
                try:
                    # Use delete_async
                    await self.mistral_client.files.delete_async(file_id=file_id) # Use keyword arg
                    logger.info(f"Successfully deleted Mistral file {file_id}.")
                except Exception as e: # Catch generic exception for deletion failure
                    logger.error(f"Failed to delete temporary Mistral file {file_id} ({type(e).__name__}): {e}")

        return full_text, page_count

    def _process_pdf_pymupdf(self, pdf_path: Path) -> tuple[str, int]:
        """Extracts text from a PDF using the pymupdf library (synchronous)."""
        log.info(f"Processing '{pdf_path.name}' using pymupdf...")
        full_text = ""
        page_count = 0
        try:
            # Use fitz.open directly as it handles closing via context manager
            with fitz.open(pdf_path) as doc:
                page_count = len(doc)
                log.debug(f"Opened PDF '{pdf_path.name}' with {page_count} pages.")
                for i, page in enumerate(doc):
                    page_text = page.get_text("text", sort=True) # Use "text" for better reading order
                    if page_text:
                        # Add page breaks for clarity
                        full_text += f"\n\n--- Page {i+1} ---\n{page_text.strip()}"
            # Remove leading/trailing whitespace
            full_text = full_text.strip()
            log.info(f"Extracted text from {page_count} pages of '{pdf_path.name}' via pymupdf. Total length: {len(full_text)}")
            return full_text, page_count
        except Exception as e:
            log.error(f"Error reading PDF '{pdf_path.name}' with pymupdf: {e}", exc_info=True)
            return "", 0

    async def run(self, **kwargs) -> None:
        """The main execution logic for the Ingestor agent.

        Retrieves PDF paths from configuration and processes each one based
        on the selected ingestion strategy.
        """
        log.info(f"[{self.agent_id}] Running IngestorAgent...")
        # Access config passed during init directly from self.config
        pdf_paths = self.config.get("pdf_paths", [])
        if not pdf_paths:
            log.warning(f"[IngestorAgent / {self.agent_id}] No 'pdf_paths' found in configuration. Nothing to ingest.")
            return

        total_chunks_added = 0
        processed_files_count = 0
        failed_files_count = 0

        for pdf_path_str in pdf_paths:
            if not pdf_path_str or not isinstance(pdf_path_str, str):
                log.warning(f"Invalid PDF path entry: {pdf_path_str}. Skipping.")
                continue

            pdf_path = Path(pdf_path_str)
            if not pdf_path.is_file():
                log.error(f"PDF file not found: {pdf_path}. Skipping.")
                failed_files_count += 1
                continue

            log.info(f"[{self.agent_id}] Processing file: {pdf_path.name} using strategy: {self.ingestion_strategy}")
            full_text = ""
            page_count = 0
            processing_successful = False

            try:
                if self.ingestion_strategy == "mistral_ocr":
                    # Ensure Mistral client is available before calling
                    if self.mistral_client:
                        full_text, page_count = await self._process_pdf_mistral(pdf_path)
                    else:
                        raise ValueError("Mistral client not initialized for mistral_ocr strategy.")
                    processing_successful = bool(full_text) # Consider success if text extracted
                elif self.ingestion_strategy == "pymupdf":
                    # Run synchronous pymupdf processing in a thread
                    full_text, page_count = await asyncio.to_thread(
                        self._process_pdf_pymupdf, pdf_path
                    )
                    processing_successful = bool(full_text) # Consider success if text extracted
                # Add elif for other strategies here...
                else:
                    log.error(f"Unsupported ingestion strategy '{self.ingestion_strategy}' for file {pdf_path.name}. Skipping.")
                    failed_files_count += 1
                    continue # Skip chunking and adding to memory for this file

                if not processing_successful:
                    log.error(f"Failed to extract text from {pdf_path.name} using {self.ingestion_strategy}.")
                    failed_files_count += 1
                    continue # Skip chunking and adding to memory

                # Chunk the extracted text
                chunks = self._chunk_text(full_text, source_ref=str(pdf_path))
                if not chunks:
                    log.warning(f"No chunks generated from {pdf_path.name} (text length: {len(full_text)}).")
                    processed_files_count += 1 # Still counts as processed if text extraction worked but was empty
                    continue

                log.info(f"Generated {len(chunks)} chunks from {pdf_path.name}.")

                # Create MemoryDoc objects for each chunk
                memory_docs = []
                for i, chunk_text in enumerate(chunks):
                    doc_id = f"doc_{uuid.uuid4()}" # Unique ID for each chunk memory
                    metadata = {
                        "agent_id": self.agent_id,
                        "source_type": "pdf",
                        "source_path": str(pdf_path),
                        "source_filename": pdf_path.name,
                        "ingestion_strategy": self.ingestion_strategy,
                        "chunk_number": i + 1,
                        "total_chunks": len(chunks),
                        "page_count": page_count, # Total pages in the original PDF (might be 0 for Mistral)
                        "mistral_model": self.mistral_model if self.ingestion_strategy == "mistral_ocr" else None,
                    }
                    # Filter out None values from metadata
                    filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
                    memory_docs.append(MemoryDoc(
                         id=doc_id, # Use keyword argument 'id'
                          text=chunk_text,
                          metadata=filtered_metadata,
                    ))

                # Add chunks to memory individually
                for doc in memory_docs:
                    self.memory_service.add_memory(doc) # Removed await
                # Original line commented out for clarity:
                # await self.memory_service.add_memory(memory_docs) 
                chunks_added_for_file = len(memory_docs)
                total_chunks_added += chunks_added_for_file
                processed_files_count += 1
                log.info(f"Successfully added {chunks_added_for_file} chunks from {pdf_path.name} to memory.")

            except Exception as e:
                log.error(f"Unhandled exception during processing of {pdf_path.name}: {e}", exc_info=True)
                failed_files_count += 1

        log.info(f"[{self.agent_id}] IngestorAgent run finished. Processed {processed_files_count} files, Failed {failed_files_count} files. Total chunks added: {total_chunks_added}.")


# --- Example Usage (for standalone testing) ---
async def run_ingestion_example():
    """Example demonstrating IngestorAgent usage."""
    log.info("--- Running Ingestion Example ---")

    # Ensure MISTRAL_API_KEY is set if testing Mistral OCR
    # export MISTRAL_API_KEY='your_key_here'

    example_dir = Path("./example_pdfs")
    example_dir.mkdir(exist_ok=True)
    pdf1_path = example_dir / "dummy1.pdf"
    pdf2_path = example_dir / "dummy2_long.pdf"

    # Create simple PDFs using pymupdf (if not existing)
    if not pdf1_path.exists():
        with fitz.open() as doc1:
            page1 = doc1.new_page()
            page1.insert_text((72, 72), "This is the first page of dummy PDF 1.")
            page2 = doc1.new_page()
            page2.insert_text((72, 72), "This is the second page.")
            doc1.save(pdf1_path)
        log.info(f"Created dummy PDF: {pdf1_path}")

    if not pdf2_path.exists():
        with fitz.open() as doc2:
            for i in range(5): # Create a slightly longer PDF
                page = doc2.new_page()
                page.insert_text((72, 72), f"Content for page {i+1} of the long PDF. " * 20)
            doc2.save(pdf2_path)
        log.info(f"Created dummy PDF: {pdf2_path}")

    pdf_paths_to_process = [str(pdf1_path.resolve()), str(pdf2_path.resolve())]
    invalid_path = "/path/to/nonexistent.pdf"
    pdf_paths_with_invalid = pdf_paths_to_process + [invalid_path]

    # --- Test pymupdf Strategy ---
    log.info("\n--- Testing pymupdf Strategy ---")
    pymupdf_agent_config = {
        "pdf_paths": pdf_paths_to_process,
    }
    pymupdf_agent = IngestorAgent(
        agent_id="ingestor_pymupdf_01",
        chunk_size=200,
        chunk_overlap=20,
        ingestion_strategy="pymupdf",
        config=pymupdf_agent_config
    )
    await pymupdf_agent.run()

    # --- Test Mistral OCR Strategy ---
    log.info("\n--- Testing Mistral OCR Strategy ---")
    if os.getenv("MISTRAL_API_KEY"):
        mistral_agent_config = {
            "pdf_paths": pdf_paths_with_invalid,
        }
        mistral_agent = IngestorAgent(
            agent_id="ingestor_mistral_01",
            chunk_size=500,
            chunk_overlap=50,
            ingestion_strategy="mistral_ocr",
            config=mistral_agent_config
        )
        await mistral_agent.run()
    else:
        log.warning("Skipping Mistral OCR test because MISTRAL_API_KEY is not set.")

    log.info("--- Ingestion Example Finished ---")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(run_ingestion_example())