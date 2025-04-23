import logging
import pdfplumber # Required for PDF processing
from pathlib import Path
import uuid

from .base import BaseAgent
from src.memory.service import MemoryService # To add memories
from src.memory.adapter import MemoryDoc # To create memory docs

# Configure logging for the ingestor agent
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class IngestorAgent(BaseAgent):
    """Agent responsible for reading documents (e.g., PDFs), extracting text,
       chunking it, and adding it to the memory system.
    """

    def __init__(self, agent_id: str, chunk_size: int = 1000, chunk_overlap: int = 100, **kwargs):
        """Initializes the Ingestor Agent.

        Args:
            agent_id: Unique ID for this agent instance.
            chunk_size: Target size for text chunks (in characters).
            chunk_overlap: Character overlap between consecutive chunks.
            **kwargs: Additional configuration passed to BaseAgent.
        """
        super().__init__(agent_id=agent_id, **kwargs)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.memory_service = MemoryService() # Get the singleton instance
        log.info(f"IngestorAgent [{self.agent_id}] initialized (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

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
            if next_start <= start: # Prevent infinite loop on very small overlaps/chunks
                start = end
            else:
                start = next_start
            if end == len(text):
                 break # Reached the end
        log.debug(f"Chunked text from '{source_ref}' into {len(chunks)} chunks.")
        return chunks

    def run_step(self, pdf_path_str: str) -> tuple[int, int]:
        """Processes a single PDF file.

        Reads the PDF, extracts text, chunks it, and adds each chunk to memory.
        This method is expected to be run synchronously by the TaskScheduler.

        Args:
            pdf_path_str: The path to the PDF file to ingest.

        Returns:
            A tuple containing: (number_of_pages_processed, number_of_chunks_added).
            Returns (0, 0) if the file cannot be processed.
        """
        pdf_path = Path(pdf_path_str)
        log.info(f"[{self.name} / {self.agent_id}] Running step: Ingesting PDF '{pdf_path.name}'")

        if not pdf_path.is_file() or pdf_path.suffix.lower() != '.pdf':
            log.error(f"Invalid PDF path or file type: {pdf_path_str}")
            return (0, 0)

        full_text = ""
        page_count = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                log.debug(f"Opened PDF '{pdf_path.name}' with {page_count} pages.")
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n\n--- Page {i+1} ---\n{page_text}"
            log.info(f"Extracted text from {page_count} pages of '{pdf_path.name}'. Total length: {len(full_text)}")
        except Exception as e:
            log.error(f"Error reading PDF '{pdf_path.name}': {e}", exc_info=True)
            return (0, 0) # Return zero counts on error

        # Chunk the extracted text
        chunks = self._chunk_text(full_text, source_ref=pdf_path.name)
        chunks_added = 0

        # Add chunks to memory
        if not chunks:
            log.warning(f"No text chunks generated for PDF: {pdf_path.name}")
            return (page_count, 0)
            
        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            metadata = {
                "source_document": pdf_path.name,
                "total_pages": page_count,
                "chunk_index": i + 1,
                "total_chunks": len(chunks)
            }
            memory_doc = MemoryDoc(
                id=doc_id,
                text=chunk,
                source_agent=self.agent_id,
                tags=["pdf_ingest", pdf_path.stem], # Add filename stem as a tag
                metadata=metadata
                # timestamp is handled by default factory
            )
            try:
                self.memory_service.add_memory(memory_doc)
                chunks_added += 1
                log.debug(f"Added chunk {i+1}/{len(chunks)} (ID: {doc_id}) to memory.")
            except Exception as e:
                log.error(f"Failed to add chunk {i+1} (ID: {doc_id}) to memory: {e}", exc_info=True)
                # Decide whether to continue or stop on error

        log.info(f"[{self.name} / {self.agent_id}] Finished step for '{pdf_path.name}'. Added {chunks_added}/{len(chunks)} chunks.")
        return (page_count, chunks_added)

# Example Usage
# Needs a PDF file (e.g., create a dummy one or use a real one)
# if __name__ == '__main__':
#     print("--- Testing IngestorAgent ---")
#     # Create a dummy PDF file for testing if needed
#     dummy_pdf_path = "./dummy_ingest_test.pdf"
#     # pdf = FPDF()
#     # pdf.add_page()
#     # pdf.set_font("Arial", size = 12)
#     # pdf.cell(200, 10, txt = "This is page 1 of the dummy PDF.", ln = 1, align = 'C')
#     # pdf.add_page()
#     # pdf.cell(200, 10, txt = "This is page 2. It contains some text.", ln = 1, align = 'C') 
#     # pdf.output(dummy_pdf_path, "F")
#     # print(f"Created dummy PDF: {dummy_pdf_path}")

#     if not Path(dummy_pdf_path).exists():
#          print(f"Error: Test PDF '{dummy_pdf_path}' not found. Cannot run example.")
#     else:
#         ingestor = IngestorAgent(agent_id="ingestor_test_01", chunk_size=200, chunk_overlap=20)
#         pages, chunks = ingestor.run_step(dummy_pdf_path)
#         print(f"Ingestion result: Processed {pages} pages, added {chunks} chunks.")

#         # Optionally, query memory service to verify
#         service = MemoryService()
#         results = service.query_memory("dummy PDF")
#         print(f"\nQuerying memory for 'dummy PDF' found {len(results)} results.")
#         for res in results:
#             print(f" - ID: {res.id}, Text: {res.text[:60]}...")

#         service.close_adapter()
#         # os.remove(dummy_pdf_path) # Clean up dummy file
#         # os.remove(".memory_db/memory.db") # Clean up db
#         # shutil.rmtree(".memory_db/embeddings") # Clean up embeddings
#         print("--- IngestorAgent Test Finished ---")

