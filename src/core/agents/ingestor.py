import logging
import fitz
from pathlib import Path
import uuid
import os
import base64
import asyncio
import aiofiles
from mistralai import Mistral

# Local imports
from src.memory.service import MemoryService
from src.memory.adapter import MemoryDoc
from src.config import config
from .base import BaseAgent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_INGESTION_STRATEGY = "pymupdf"

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
        self.mistral_client = None
        self.mistral_model = None
        self.mistral_page_limit = None

        # Initialize clients based on strategy
        if self.ingestion_strategy == "mistral_ocr":
            self.mistral_model = config.get("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
            self.mistral_page_limit = config.get("MISTRAL_OCR_PAGE_LIMIT", None)
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                log.error("MISTRAL_API_KEY environment variable not set. Mistral OCR will fail.")
            else:
                try:
                    self.mistral_client = Mistral()
                    log.info(f"Mistral client initialized successfully (model={self.mistral_model}, page_limit={self.mistral_page_limit}).")
                except Exception as e:
                    log.error(f"Failed to initialize Mistral client: {e}", exc_info=True)
                    self.mistral_client = None
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
            if next_start <= start:
                start = end
            else:
                start = next_start
            if end == len(text):
                break
        log.debug(f"Chunked text from '{source_ref}' into {len(chunks)} chunks.")
        return chunks

    async def _process_pdf_mistral(self, pdf_path: Path) -> tuple[str, int]:
        if not self.mistral_client:
            log.error(f"Mistral client not available. Cannot process '{pdf_path.name}'.")
            return "", 0

        log.info(f"Processing '{pdf_path.name}' using Mistral OCR (model={self.mistral_model}, page_limit={self.mistral_page_limit})...")
        try:
            async with aiofiles.open(pdf_path, mode='rb') as f:
                pdf_bytes = await f.read()

            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            data_url = f"data:application/pdf;base64,{base64_pdf}"

            ocr_params = {
                "model": self.mistral_model,
                "files": [data_url]
            }
            pages_to_process = None
            if isinstance(self.mistral_page_limit, int) and self.mistral_page_limit > 0:
                pages_to_process = list(range(self.mistral_page_limit))
                ocr_params["pages"] = pages_to_process
                log.debug(f"Limiting Mistral OCR to first {self.mistral_page_limit} pages.")

            response = await asyncio.to_thread(self.mistral_client.ocr, **ocr_params)

            # --- TODO: Verify Mistral SDK Response Structure --- 
            # The following assumes the response object has a 'files' attribute
            # which is a list, and the first element has a 'content' attribute
            # containing the extracted markdown/text. Adjust as needed based on the
            # actual Mistral SDK version and documentation.
            # Also, confirm how to accurately get the number of pages processed if needed.
            full_text = ""
            page_count_processed = 0
            if response and hasattr(response, 'files') and len(response.files) > 0:
                if hasattr(response.files[0], 'content'):
                    full_text = response.files[0].content
                    page_count_processed = len(pages_to_process) if pages_to_process else 0
                    log.info(f"Extracted text from '{pdf_path.name}' via Mistral OCR. Length: {len(full_text)}. Processed pages estimate: {page_count_processed if pages_to_process else 'all'}")

            return full_text, page_count_processed

        except Exception as e:
            log.error(f"Error during Mistral OCR processing for '{pdf_path.name}': {e}", exc_info=True)
            return "", 0

    def _process_pdf_pymupdf(self, pdf_path: Path) -> tuple[str, int]:
        """Extracts text from a PDF using the pymupdf library (synchronous)."""
        log.info(f"Processing '{pdf_path.name}' using pymupdf...")
        full_text = ""
        page_count = 0
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            log.debug(f"Opened PDF '{pdf_path.name}' with {page_count} pages.")
            for i, page in enumerate(doc):
                page_text = page.get_text()
                if page_text:
                    full_text += f"\n\n--- Page {i+1} ---\n{page_text}"
            doc.close()
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
        # BaseAgent stores the whole kwargs dict in self.config.
        # The config dict passed during init is nested under the 'config' key.
        pdf_paths = self.config.get("config", {}).get("pdf_paths", [])
        if not pdf_paths:
            log.warning(f"[IngestorAgent / {self.agent_id}] No 'pdf_paths' found in configuration. Nothing to ingest.")
            return

        total_chunks_added = 0
        for pdf_path_str in pdf_paths:
            if not pdf_path_str or not isinstance(pdf_path_str, str):
                log.warning(f"Invalid PDF path entry: {pdf_path_str}. Skipping.")
                continue

            pdf_path = Path(pdf_path_str)
            if not pdf_path.is_file():
                log.error(f"PDF file not found: {pdf_path}. Skipping.")
                continue

            log.info(f"[{self.agent_id}] Processing file: {pdf_path.name} using strategy: {self.ingestion_strategy}")
            full_text = ""
            page_count = 0
            chunks_added_for_file = 0
            actual_page_limit = None

            if self.ingestion_strategy == "mistral_ocr":
                full_text, page_count = await self._process_pdf_mistral(pdf_path)
                actual_page_limit = self.mistral_page_limit
            elif self.ingestion_strategy == "pymupdf":
                full_text, page_count = await asyncio.to_thread(
                    self._process_pdf_pymupdf, pdf_path
                )
            else:
                log.error(f"Unsupported ingestion strategy: '{self.ingestion_strategy}' for file {pdf_path.name}")
                continue

            if not full_text:
                log.warning(f"No text extracted from '{pdf_path.name}' using strategy '{self.ingestion_strategy}'. Skipping chunking.")
                continue

            chunks = self._chunk_text(full_text, source_ref=str(pdf_path))
            if not chunks:
                log.warning(f"No text chunks generated for PDF: {pdf_path.name}")
                continue

            log.info(f"Adding {len(chunks)} chunks from '{pdf_path.name}' to memory...")
            for i, chunk in enumerate(chunks):
                doc_id = str(uuid.uuid4())
                metadata = {
                    "source_document": pdf_path.name,
                    "total_pages": page_count,
                    "chunk_index": i + 1,
                    "total_chunks": len(chunks),
                    "ingestion_strategy": self.ingestion_strategy,
                    "mistral_model": self.mistral_model if self.ingestion_strategy == "mistral_ocr" else None,
                    "page_limit_used": actual_page_limit if self.ingestion_strategy == "mistral_ocr" else None
                }
                tags = [self.ingestion_strategy, "pdf_ingestion", pdf_path.name]
                self.memory_service.add_memory(
                    MemoryDoc(
                        id=doc_id,
                        text=chunk,
                        source_agent=self.agent_id,
                        tags=tags,
                        metadata=metadata
                    )
                )
                chunks_added_for_file += 1
                log.debug(f"Added chunk {i+1}/{len(chunks)} (ID: {doc_id}) from {pdf_path.name}.")

            log.info(f"Finished processing '{pdf_path.name}'. Added {chunks_added_for_file}/{len(chunks)} chunks.")
            total_chunks_added += chunks_added_for_file

        log.info(f"[{self.agent_id}] Run cycle finished. Processed {len(pdf_paths)} files, added {total_chunks_added} total chunks.")

async def run_ingestion_example():
    print("--- Testing IngestorAgent (Async) ---")
    ingestor_config = {
        'pdf_paths': ["./dummy_ingest_test.pdf"]
    }
    ingestor = IngestorAgent(
        agent_id="ingestor_async_test_01",
        chunk_size=500,
        chunk_overlap=50,
        ingestion_strategy="pymupdf",
        **ingestor_config
    )

    try:
        await ingestor.run()
        print("Ingestion run completed.")

        service = MemoryService()
        results = service.query_memory("dummy_ingest_test")
        print(f"\nQuerying memory for tag 'dummy_ingest_test' found {len(results)} results.")
        for res in results:
            print(f" - ID: {res.id}, Text: {res.text[:80]}...", res.metadata)

        service.close_adapter()
    except Exception as e:
        print(f"An error occurred during the ingestion example: {e}")
        import traceback
        traceback.print_exc()

    print("--- IngestorAgent Async Test Finished ---")

    print("\n--- Testing Mistral OCR Strategy ---")
    ingestor_config_mistral = {
        'pdf_paths': ["./dummy_ingest_test.pdf"]
    }
    ingestor_mistral = IngestorAgent(
        agent_id="ingestor_mistral_test_01",
        chunk_size=500,
        chunk_overlap=50,
        ingestion_strategy="mistral_ocr",
        **ingestor_config_mistral
    )
    try:
        await ingestor_mistral.run()
        print("Mistral OCR ingestion run completed.")
    except Exception as e:
        print(f"An error occurred during the Mistral OCR ingestion example: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing pymupdf Strategy ---")
    ingestor_config_pymupdf = {
        'pdf_paths': ["./dummy_ingest_test.pdf"]
    }
    ingestor_pymupdf = IngestorAgent(
        agent_id="ingestor_pymupdf_test_01",
        chunk_size=500,
        chunk_overlap=50,
        ingestion_strategy="pymupdf",
        **ingestor_config_pymupdf
    )
    try:
        await ingestor_pymupdf.run()
        print("pymupdf ingestion run completed.")
    except Exception as e:
        print(f"An error occurred during the pymupdf ingestion example: {e}")
        import traceback
        traceback.print_exc()

    try:
        service = MemoryService()
        results_mistral = service.query_memory("mistral_ocr")
        results_pymupdf = service.query_memory("pymupdf")
        print(f"\nQuerying memory found {len(results_mistral)} mistral_ocr chunks and {len(results_pymupdf)} pymupdf chunks.")
        service.close_adapter()
    except Exception as e:
        print(f"Error querying memory service: {e}")

    print("--- IngestorAgent Async Test Finished ---")

if __name__ == '__main__':
    asyncio.run(run_ingestion_example())