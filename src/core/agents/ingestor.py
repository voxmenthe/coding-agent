import logging
import fitz
from pathlib import Path
import uuid
import os
import asyncio
import aiofiles
# Use unified client
from mistralai import Mistral

import pymupdf

# Local imports
# Remove MemoryService import
# from src.memory.service import MemoryService 
from src.memory.adapter import MemoryDoc
from src.memory.hybrid_sqlite import HybridSQLiteAdapter # Add adapter import
from src.config import config
from .base import BaseAgent
from typing import Dict, List, Optional, Tuple

# Configure logger specifically for this agent
# logging.basicConfig(level=logging.INFO) # Keep root config if needed, but use specific logger
log = logging.getLogger('IngestorAgent') # Use specific logger name
log.setLevel(logging.DEBUG) # Set level for this logger

DEFAULT_INGESTION_STRATEGY = "pymupdf"

MISTRAL_OCR_DEFAULT_MODEL = "mistral-ocr-latest"

# logger = logging.getLogger(__name__) # Replaced with specific logger

class IngestorAgent(BaseAgent):
    """Agent responsible for reading documents (e.g., PDFs), extracting text
       using a configured strategy (e.g., Mistral OCR, pymupdf), chunking it,
       and adding it to the memory system.
    """

    def __init__(self,
                 agent_id: str,
                 adapter: HybridSQLiteAdapter, # Add adapter parameter
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
        log.debug(f"[{agent_id}] Initializing IngestorAgent...")
        super().__init__(agent_id=agent_id, **kwargs)
        if chunk_size <= 0:
            log.error(f"[{self.agent_id}] Invalid chunk_size: {chunk_size}")
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            log.error(f"[{self.agent_id}] Invalid chunk_overlap: {chunk_overlap} (chunk_size: {chunk_size})")
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Remove MemoryService instantiation
        # self.memory_service = MemoryService() # Get the singleton instance 
        self.adapter = adapter # Store the passed adapter instance
        self.ingestion_strategy = ingestion_strategy
        self.mistral_client: Optional[Mistral] = None # Use unified client type
        self.mistral_model = MISTRAL_OCR_DEFAULT_MODEL
        log.debug(f"[{self.agent_id}] Adapter instance received: {type(adapter)}")

        # Initialize clients based on strategy
        if self.ingestion_strategy == "mistral_ocr":
            log.debug(f"[{self.agent_id}] Initializing Mistral client...")
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                log.error(f"[{self.agent_id}] MISTRAL_API_KEY environment variable not set. Mistral OCR will fail.")
                # No need for else block, keep mistral_client as None
            else:
                try:
                    # Initialize the client using the API key
                    # Initialize unified client
                    self.mistral_client = Mistral(api_key=api_key)
                    log.info(f"[{self.agent_id}] Mistral client initialized successfully (model={self.mistral_model}).") 
                except Exception as e:
                    log.error(f"[{self.agent_id}] Failed to initialize Mistral client: {e}", exc_info=True)
                    self.mistral_client = None # Ensure it's None on failure
        elif self.ingestion_strategy == "pymupdf":
            log.info(f"[{self.agent_id}] Using pymupdf strategy. No specific client initialization needed.")
        # Add elif blocks here for other strategies like 'gemini_native' later
        else:
            log.warning(f"[{self.agent_id}] Unknown ingestion strategy '{self.ingestion_strategy}' specified during init.")

        log.info(f"IngestorAgent [{self.agent_id}] initialized (strategy={self.ingestion_strategy}, chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

    def _chunk_text(self, text: str, source_ref: str) -> list[str]:
        """Simple text chunking based on character count with overlap."""
        log.debug(f"[{self.agent_id}] Entering _chunk_text for source '{source_ref}' (text length: {len(text)})")
        if not text:
            log.warning(f"[{self.agent_id}] _chunk_text received empty text for '{source_ref}'. Returning empty list.")
            return []

        chunks = []
        start = 0
        chunk_num = 1
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_content = text[start:end]
            chunks.append(chunk_content)
            log.debug(f"[{self.agent_id}] Created chunk {chunk_num} ({start}-{end}, length {len(chunk_content)}) for '{source_ref}'")
            chunk_num += 1

            next_start = end - self.chunk_overlap
            # Ensure progress if overlap is large or chunk_size is small
            if next_start <= start:
                 log.debug(f"[{self.agent_id}] Chunk overlap ({self.chunk_overlap}) does not allow progress (next_start {next_start} <= start {start}). Moving start to end ({end}).")
                 # Move to the position right after the current chunk if no progress would be made
                 start = end
            else:
                start = next_start
            # Break if we've reached the end
            if end == len(text):
                log.debug(f"[{self.agent_id}] Reached end of text during chunking for '{source_ref}'.")
                break
        log.info(f"[{self.agent_id}] Chunked text from '{source_ref}' into {len(chunks)} chunks.")
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
        log.debug(f"[{self.agent_id}] Entering _process_pdf_mistral for {pdf_path.name}")
        if not self.mistral_client:
            log.error(f"[{self.agent_id}] Mistral client not initialized. Cannot process {pdf_path.name} with mistral_ocr.")
            return "", 0

        file_id = None
        uploaded_file_obj = None # Keep track of the uploaded file object
        full_text = ""
        page_count = 0
        try:
            # 1. Upload the file asynchronously using the async client
            log.info(f"[{self.agent_id}] Uploading file {pdf_path.name} to Mistral...")
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
            log.info(f"[{self.agent_id}] File {pdf_path.name} uploaded successfully with ID: {file_id}")

            # 2. Get Signed URL - Use direct await
            log.info(f"[{self.agent_id}] Getting signed URL for file ID: {file_id}...")
            # Use get_signed_url_async
            signed_url_obj = await self.mistral_client.files.get_signed_url_async(
                file_id=file_id
            )
            signed_url = signed_url_obj.url
            log.info(f"[{self.agent_id}] Obtained signed URL for file ID: {file_id}")

            # 3. Process OCR using the signed URL - Use direct await
            log.info(f"[{self.agent_id}] Processing OCR for signed URL (file ID: {file_id})...")
            # Use process_async with correct 'document' dictionary structure
            ocr_result = await self.mistral_client.ocr.process_async(
                model=self.mistral_model, 
                document={
                    "type": "document_url",
                    "document_url": signed_url
                }
            ) 
            # logger.info(f"OCR Result type: {type(ocr_result)}") # Old logger
            # logger.info(f"OCR Result dir: {dir(ocr_result)}")
            log.debug(f"[{self.agent_id}] OCR Result type: {type(ocr_result)}, Dir: {dir(ocr_result)}") # New logger
            # Iterate through pages and join markdown content
            full_text = "\n".join([page.markdown for page in ocr_result.pages])
            # Calculate page count based on the number of pages in the response
            page_count = len(ocr_result.pages)
            log.info(f"[{self.agent_id}] Successfully extracted text (length {len(full_text)}) from {page_count} pages using Mistral OCR for {pdf_path.name}.")
        except Exception as e: # Catch generic exception
            log.error(
                f"[{self.agent_id}] Error during Mistral API call for {pdf_path.name}"
                f"{' (File ID: ' + uploaded_file_obj.id + ')' if uploaded_file_obj else ''}: {e}",
                exc_info=True # Add full traceback to log for debugging
            )
            full_text = ""
            page_count = 0
        # except Exception as e: # Duplicate catch block removed
        #     # Catch any other unexpected errors during processing
        #     logger.error(f"Unexpected error during Mistral processing for {pdf_path.name}: {e}", exc_info=True)
        #     full_text = ""
        #     page_count = 0

        finally:
            # 6. Delete the uploaded file - Use direct await
            if file_id:
                log.info(f"[{self.agent_id}] Attempting to delete uploaded Mistral file {file_id}...")
                try:
                    # Use delete_async
                    await self.mistral_client.files.delete_async(file_id=file_id) # Use keyword arg
                    log.info(f"[{self.agent_id}] Successfully deleted Mistral file {file_id}.")
                except Exception as e: # Catch generic exception for deletion failure
                    log.error(f"[{self.agent_id}] Failed to delete temporary Mistral file {file_id} ({type(e).__name__}): {e}")

        log.debug(f"[{self.agent_id}] Exiting _process_pdf_mistral for {pdf_path.name}. Returning text length {len(full_text)}, page count {page_count}.")
        return full_text, page_count

    def _process_pdf_pymupdf(self, pdf_path: Path) -> tuple[str, int]:
        """Extracts text from a PDF using the pymupdf library (synchronous)."""
        log.debug(f"[{self.agent_id}] Entering _process_pdf_pymupdf for '{pdf_path.name}'...")
        full_text = ""
        page_count = 0
        try:
            log.debug(f"[{self.agent_id}] Opening PDF '{pdf_path.name}' with fitz.open()...")
            # Use fitz.open directly as it handles closing via context manager
            with fitz.open(pdf_path) as doc:
                page_count = len(doc)
                log.info(f"[{self.agent_id}] Opened PDF '{pdf_path.name}' with {page_count} pages.")
                for i, page in enumerate(doc):
                    log.debug(f"[{self.agent_id}] Extracting text from page {i+1}/{page_count}...")
                    page_text = page.get_text("text", sort=True) # Use "text" for better reading order
                    if page_text:
                        log.debug(f"[{self.agent_id}] Page {i+1} text length: {len(page_text)}")
                        # Add page breaks for clarity
                        full_text += f"\n\n--- Page {i+1} ---\n{page_text.strip()}"
                    else:
                        log.debug(f"[{self.agent_id}] Page {i+1} has no text content.")
            # Remove leading/trailing whitespace
            full_text = full_text.strip()
            log.info(f"[{self.agent_id}] Successfully extracted text from {page_count} pages of '{pdf_path.name}' via pymupdf. Total length: {len(full_text)}")
        except Exception as e:
            log.error(f"[{self.agent_id}] Error reading PDF '{pdf_path.name}' with pymupdf: {e}", exc_info=True)
            full_text = "" # Ensure empty on error
            page_count = 0 # Ensure zero on error
        
        log.debug(f"[{self.agent_id}] Exiting _process_pdf_pymupdf for '{pdf_path.name}'. Returning text length {len(full_text)}, page count {page_count}.")
        return full_text, page_count

    async def run(self, **kwargs) -> None:
        """The main execution logic for the Ingestor agent.

        Retrieves PDF paths from configuration and processes each one based
        on the selected ingestion strategy.
        """
        log.info(f"[{self.agent_id}] >>> Starting IngestorAgent run() method...")
        # Access config passed during init directly from self.config
        # *** Add diagnostic log ***
        log.debug(f"[{self.agent_id}] Config dictionary at start of run: {self.config}")
        pdf_paths = self.config.get("pdf_paths", [])
        if not pdf_paths:
            log.warning(f"[{self.agent_id}] No 'pdf_paths' found in configuration. Nothing to ingest. Exiting run().")
            return

        log.info(f"[{self.agent_id}] Found {len(pdf_paths)} PDF paths to process: {pdf_paths}")
        total_chunks_added = 0
        processed_files_count = 0
        failed_files_count = 0

        for pdf_path_str in pdf_paths:
            log.info(f"[{self.agent_id}] --- Processing PDF: {pdf_path_str} ---")
            if not pdf_path_str or not isinstance(pdf_path_str, str):
                log.warning(f"[{self.agent_id}] Invalid PDF path entry: {pdf_path_str}. Skipping.")
                continue

            pdf_path = Path(pdf_path_str)
            if not pdf_path.is_file():
                log.error(f"[{self.agent_id}] PDF file not found: {pdf_path}. Skipping.")
                failed_files_count += 1
                continue

            log.info(f"[{self.agent_id}] Processing file: {pdf_path.name} using strategy: {self.ingestion_strategy}")
            full_text = ""
            page_count = 0
            processing_successful = False

            try:
                # --- PDF Processing --- 
                log.debug(f"[{self.agent_id}] Entering PDF processing block for {pdf_path.name}...")
                if self.ingestion_strategy == "mistral_ocr":
                    if self.mistral_client:
                        log.debug(f"[{self.agent_id}] Calling _process_pdf_mistral for {pdf_path.name}...")
                        full_text, page_count = await self._process_pdf_mistral(pdf_path) 
                        log.debug(f"[{self.agent_id}] _process_pdf_mistral returned. Text length: {len(full_text)}, Pages: {page_count}")
                    else:
                        log.error(f"[{self.agent_id}] Mistral client not initialized for {pdf_path.name}. Skipping processing.")
                        full_text, page_count = "", 0
                elif self.ingestion_strategy == "pymupdf":
                    log.debug(f"[{self.agent_id}] Calling anyio.to_thread.run_sync for _process_pdf_pymupdf({pdf_path.name})...")
                    import anyio
                    full_text, page_count = await anyio.to_thread.run_sync(self._process_pdf_pymupdf, pdf_path)
                    log.debug(f"[{self.agent_id}] anyio.to_thread.run_sync(_process_pdf_pymupdf) returned. Text length: {len(full_text)}, Pages: {page_count}")
                    # *** Add diagnostic log *** # Already present
                    # log.info(f"[{self.agent_id}] Text extracted for {pdf_path.name}. Length: {len(full_text)}, Page count: {page_count}")
                # Add elif blocks here for other strategies later
                else:
                    log.warning(f"[{self.agent_id}] Unsupported ingestion strategy '{self.ingestion_strategy}' for file {pdf_path.name}. Skipping processing.")
                    full_text, page_count = "", 0
                log.debug(f"[{self.agent_id}] PDF processing block finished for {pdf_path.name}.")

                # --- Chunking and Storing --- 
                log.debug(f"[{self.agent_id}] Entering Chunking and Storing block for {pdf_path.name}. Text length: {len(full_text)}")
                if full_text:
                    # *** Add diagnostic log *** # Already present
                    # log.info(f"[{self.agent_id}] Entering chunking block for {pdf_path.name}. Text length: {len(full_text)}") 
                    log.info(f"[{self.agent_id}] Chunking text from {pdf_path.name} (length: {len(full_text)})...")
                    try:
                        log.debug(f"[{self.agent_id}] Calling _chunk_text...")
                        chunks = self._chunk_text(full_text, pdf_path.name)
                        log.info(f"[{self.agent_id}] _chunk_text returned {len(chunks)} chunks. Adding to memory...")
                        
                        chunks_added_for_file = 0
                        for i, chunk_text in enumerate(chunks):
                            chunk_num = i + 1
                            log.debug(f"[{self.agent_id}] Preparing MemoryDoc for chunk {chunk_num}/{len(chunks)} (length {len(chunk_text)}) for {pdf_path.name}...")
                            memory_doc = MemoryDoc(
                                id=None, # Rely on adapter.add to generate UUID
                                text=chunk_text,
                                source_agent=self.agent_id,
                                tags=["chunk", self.ingestion_strategy, pdf_path.stem], # Add file stem as tag
                                metadata={
                                    "source_pdf": str(pdf_path),
                                    "chunk_num": chunk_num, # 1-based index
                                    "total_chunks": len(chunks),
                                    "page_count": page_count, # Page count of the source PDF
                                    "strategy": self.ingestion_strategy
                                }
                            )
                            log.debug(f"[{self.agent_id}] MemoryDoc prepared: Tags={memory_doc.tags}, Meta={memory_doc.metadata}")
                            try:
                                log.debug(f"[{self.agent_id}] Calling anyio.to_thread.run_sync for adapter.add() for chunk {chunk_num}...")
                                # Adapter add is synchronous, run in thread
                                doc_id = await anyio.to_thread.run_sync(self.adapter.add, memory_doc)
                                if doc_id:
                                    log.info(f"[{self.agent_id}] Successfully added chunk {chunk_num}/{len(chunks)} (DB ID: {doc_id}) for {pdf_path.name}") # Changed log level to INFO
                                    chunks_added_for_file += 1
                                else:
                                     log.error(f"[{self.agent_id}] adapter.add() returned None or empty for chunk {chunk_num} of {pdf_path.name}. This should not happen.")
                            except Exception as add_err:
                                log.error(f"[{self.agent_id}] Error adding chunk {chunk_num} for {pdf_path.name} to memory: {add_err}", exc_info=True)
                                # Decide whether to continue adding other chunks or break
                                log.warning(f"[{self.agent_id}] Stopping chunk processing for {pdf_path.name} due to add error.")
                                break # Stop processing chunks for this file on error

                        log.debug(f"[{self.agent_id}] Finished loop adding chunks for {pdf_path.name}. Added: {chunks_added_for_file}/{len(chunks)}")
                        total_chunks_added += chunks_added_for_file
                        if chunks_added_for_file == len(chunks) and len(chunks) > 0: # Ensure chunks were actually generated
                             processing_successful = True # Mark successful only if all chunks added
                             log.info(f"[{self.agent_id}] Successfully processed and added all {chunks_added_for_file} chunks for {pdf_path.name}.")
                        elif len(chunks) == 0 and len(full_text) > 0: # Check if text existed but no chunks created
                            log.warning(f"[{self.agent_id}] No chunks were generated for {pdf_path.name}, but text was present (length {len(full_text)}). Setting processing_successful=False.")
                            processing_successful = False
                        elif len(chunks) == 0 and len(full_text) == 0: # No text, no chunks is expected
                             log.info(f"[{self.agent_id}] No text extracted, so no chunks generated for {pdf_path.name}. Marked successful=False earlier.")
                             processing_successful = False # Keep as false if no text
                        else: # Chunks generated but not all added
                             log.warning(f"[{self.agent_id}] Only added {chunks_added_for_file}/{len(chunks)} chunks for {pdf_path.name} due to errors. Setting processing_successful=False.")
                             processing_successful = False # Ensure failure if not all chunks added
                    
                    except Exception as chunk_err:
                         log.error(f"[{self.agent_id}] Error during chunking or the add loop for {pdf_path.name}: {chunk_err}", exc_info=True)
                         processing_successful = False

                else:
                    log.warning(f"[{self.agent_id}] No text extracted from {pdf_path.name}. Cannot chunk or add to memory. Setting processing_successful=False.")
                    processing_successful = False 
                log.debug(f"[{self.agent_id}] Exiting Chunking and Storing block for {pdf_path.name}. Success: {processing_successful}")

            except Exception as process_err:
                log.error(f"[{self.agent_id}] Unhandled exception during processing of {pdf_path.name}: {process_err}", exc_info=True)
                processing_successful = False

            # --- Update Counters --- 
            log.debug(f"[{self.agent_id}] Updating counters for {pdf_path.name}. Success: {processing_successful}")
            if processing_successful:
                processed_files_count += 1
                log.info(f"[{self.agent_id}] * Successfully completed processing file: {pdf_path.name}")
            else:
                failed_files_count += 1
                log.warning(f"[{self.agent_id}] * Failed to complete processing file: {pdf_path.name}")
            log.info(f"[{self.agent_id}] --- Finished processing PDF: {pdf_path_str} ---")

        log.info(f"[{self.agent_id}] <<< Finished IngestorAgent run() method.")
        log.info(f"[{self.agent_id}] Summary: Processed {processed_files_count} files successfully.")
        if failed_files_count > 0:
            log.warning(f"[{self.agent_id}] {failed_files_count} files failed processing.")
        log.info(f"[{self.agent_id}] Total chunks added to memory across all files: {total_chunks_added}")


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
    log.info("--- Testing pymupdf Strategy ---")
    pymupdf_agent_config = {
        "pdf_paths": pdf_paths_to_process,
    }
    # Need an adapter for the example
    example_adapter_pymupdf = HybridSQLiteAdapter(db_path_str="./example_pymupdf.db", embedding_dir_str="./example_pymupdf_embeddings")
    pymupdf_agent = IngestorAgent(
        agent_id="ingestor_pymupdf_example",
        adapter=example_adapter_pymupdf, # Pass adapter instance
        chunk_size=200,
        chunk_overlap=20,
        ingestion_strategy="pymupdf",
        config=pymupdf_agent_config
    )
    await pymupdf_agent.run()
    example_adapter_pymupdf.close() # Close connection

    # --- Test Mistral OCR Strategy ---
    log.info("--- Testing Mistral OCR Strategy ---")
    if os.getenv("MISTRAL_API_KEY"):
        mistral_agent_config = {
            "pdf_paths": pdf_paths_with_invalid,
        }
        # Need an adapter for the example
        example_adapter_mistral = HybridSQLiteAdapter(db_path_str="./example_mistral.db", embedding_dir_str="./example_mistral_embeddings")
        mistral_agent = IngestorAgent(
            agent_id="ingestor_mistral_example",
            adapter=example_adapter_mistral, # Pass adapter instance
            chunk_size=500,
            chunk_overlap=50,
            ingestion_strategy="mistral_ocr",
            config=mistral_agent_config
        )
        await mistral_agent.run()
        example_adapter_mistral.close() # Close connection
    else:
        log.warning("Skipping Mistral OCR test because MISTRAL_API_KEY is not set.")

    log.info("--- Ingestion Example Finished ---")


if __name__ == '__main__':
    # Ensure root logger also shows DEBUG if needed for other modules
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # If standalone, ensure CLI logger isn't interfering or use root logger
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    log.info("Running IngestorAgent standalone example...")
    asyncio.run(run_ingestion_example())
    log.info("IngestorAgent standalone example finished.")