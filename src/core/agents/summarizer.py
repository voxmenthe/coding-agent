import logging
import os
import anyio
from google import genai
from google.genai import types
from src.core.agents.base import BaseAgent
from src.memory.hybrid_sqlite import HybridSQLiteAdapter
from src.memory.adapter import MemoryDoc
import functools # Import functools

# Configure logging
# logging.basicConfig(level=logging.INFO) # Keep root config
log = logging.getLogger('SummarizerAgent') # Use specific logger name
log.setLevel(logging.DEBUG) # Set level for this logger

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-preview-0514"

class SummarizerAgent(BaseAgent):
    """Agent responsible for summarizing text chunks retrieved from memory."""

    def __init__(self, 
                 agent_id: str, 
                 adapter: HybridSQLiteAdapter, 
                 gemini_api_key: str | None = None,
                 gemini_model: str = DEFAULT_GEMINI_MODEL,
                 **kwargs):
        """Initializes the Summarizer Agent.

        Args:
            agent_id: Unique ID for this agent instance.
            adapter: Instance of HybridSQLiteAdapter for memory access.
            gemini_api_key: API key for Google Gemini. If None, tries OS env GEMINI_API_KEY.
            gemini_model: The Gemini model to use for summarization.
            **kwargs: Additional configuration passed to BaseAgent (e.g., target_source).
        """
        log.debug(f"[{agent_id}] Initializing SummarizerAgent...")
        super().__init__(agent_id=agent_id, **kwargs)
        self.adapter = adapter
        self.model_name = gemini_model # Store the model name
        log.debug(f"[{self.agent_id}] Adapter instance received: {type(adapter)}")
        log.debug(f"[{self.agent_id}] Target source from config: {kwargs.get('target_source')}")

        _api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not _api_key:
            log.error(f"[{self.agent_id}] Gemini API key not provided or found in environment (GEMINI_API_KEY). Summarization will fail.")
            self.client = None
        else:
            log.debug(f"[{self.agent_id}] Configuring Gemini client for model {self.model_name}...")
            try:
                # *** NEW SDK CLIENT INITIALIZATION ***
                # No longer use genai.configure
                # genai.configure(api_key=_api_key)
                self.client = genai.Client(api_key=_api_key)
                # *** END NEW SDK CLIENT INITIALIZATION ***
                log.info(f"[{self.agent_id}] Gemini client configured successfully for model {self.model_name}.")
            except Exception as e:
                log.error(f"[{self.agent_id}] Error configuring Gemini client: {e}", exc_info=True)
                self.client = None
        
        # Extract target source identifier (e.g., PDF filename stem) from config
        self.target_source = self.config.get("target_source")
        if not self.target_source:
            log.warning(f"[{self.agent_id}] 'target_source' not found in configuration. Agent may not find relevant chunks.")
        else:
            log.info(f"[{self.agent_id}] Target source set to: '{self.target_source}'")

        log.info(f"SummarizerAgent [{self.agent_id}] initialized.")


    async def _generate_summary(self, text_to_summarize: str) -> str | None:
        """Generates a summary using the Gemini API (synchronous) via the NEW SDK."""
        log.debug(f"[{self.agent_id}] Entering _generate_summary (text length: {len(text_to_summarize)})...")
        if not self.client:
            log.error(f"[{self.agent_id}] Gemini client not available. Cannot generate summary.")
            return None
        
        prompt = f"Summarize the following text concisely:\n\n---\n{text_to_summarize}\n---\n\nSummary:"
        log.debug(f"[{self.agent_id}] Prompt created (first 100 chars): {prompt[:100]}...")
        
        try:
            log.info(f"[{self.agent_id}] Generating summary for text (length: {len(text_to_summarize)}) using model {self.model_name}...")
            # *** NEW SDK generate_content CALL ***
            response = self.client.models.generate_content(
                model=self.model_name, # Pass model name directly (no 'models/' prefix needed)
                contents=prompt, # Pass prompt string in 'contents'
                # generation_config=genai.types.GenerationConfig(...)
                # safety_settings=... 
            )
            # Access text directly from response
            summary = response.text 
            # *** END NEW SDK generate_content CALL ***
            log.info(f"[{self.agent_id}] Summary generated successfully (length: {len(summary)})." )
            log.debug(f"[{self.agent_id}] Generated summary (first 100 chars): {summary[:100]}...")
            return summary
        except Exception as e:
            log.error(f"[{self.agent_id}] Error during Gemini API call: {e}", exc_info=True)
            return None

    async def run(self) -> None:
        """The main execution logic for the Summarizer agent."""
        log.info(f"[{self.agent_id}] >>> Starting SummarizerAgent run() method for target source: {self.target_source}")

        if not self.target_source:
            log.error(f"[{self.agent_id}] Cannot run without a 'target_source' configured. Exiting run().")
            return
            
        if not self.client:
             log.error(f"[{self.agent_id}] Cannot run because Gemini client is not configured. Exiting run().")
             return

        query_results = [] # Initialize to avoid potential UnboundLocalError
        combined_text = "" # Initialize
        # 1. Retrieve relevant chunks from memory
        try:
            log.info(f"[{self.agent_id}] Querying memory for chunks related to '{self.target_source}'...")
            # Query using the target_source which should match the tag added by ingestor
            # Use hybrid_query; adjust k as needed
            # Adapter query methods are synchronous, run in thread
            # *** Use functools.partial for keyword arguments ***
            query_text = f"content related to {self.target_source}" # Example query text
            k_val = 20 # Retrieve more chunks for summary context
            filter_tags_val = [self.target_source, "chunk"] # Filter by source and type
            log.debug(f"[{self.agent_id}] Preparing hybrid_query: query='{query_text}', k={k_val}, tags={filter_tags_val}")
            query_func = functools.partial(
                self.adapter.hybrid_query, 
                query_text=query_text,
                k=k_val, 
                filter_tags=filter_tags_val
            )
            log.debug(f"[{self.agent_id}] Calling anyio.to_thread.run_sync for adapter.hybrid_query()...")
            query_results = await anyio.to_thread.run_sync(query_func)
            log.debug(f"[{self.agent_id}] adapter.hybrid_query returned {len(query_results)} results.")
            
            if not query_results:
                log.warning(f"[{self.agent_id}] No chunks found in memory for target source: {self.target_source}. Cannot generate summary. Exiting run().")
                return

            log.info(f"[{self.agent_id}] Retrieved {len(query_results)} chunks for summarization.")
            
            # Combine text from chunks (results are tuples: (MemoryDoc, score))
            # Sort by chunk number if available in metadata
            log.debug(f"[{self.agent_id}] Sorting {len(query_results)} retrieved docs by chunk_num...")
            sorted_docs = sorted(
                [doc for doc, score in query_results if doc.text], # Filter out docs without text
                key=lambda d: d.metadata.get("chunk_num", float('inf')) if d.metadata else float('inf')
            )
            log.debug(f"[{self.agent_id}] Found {len(sorted_docs)} docs with text after filtering.")
            if not sorted_docs:
                 log.warning(f"[{self.agent_id}] No documents with text found after filtering query results for '{self.target_source}'. Cannot generate summary. Exiting run().")
                 return

            combined_text = "\n\n".join([doc.text for doc in sorted_docs])
            log.info(f"[{self.agent_id}] Combined text from {len(sorted_docs)} chunks. Total length: {len(combined_text)}.")

        except Exception as query_err:
            log.error(f"[{self.agent_id}] Error querying memory or processing chunks: {query_err}", exc_info=True)
            return # Exit run on error

        # 2. Generate summary using LLM (run synchronous _generate_summary in thread)
        log.debug(f"[{self.agent_id}] Calling anyio.to_thread.run_sync for _generate_summary()...")
        summary_text = await anyio.to_thread.run_sync(self._generate_summary, combined_text)
        log.debug(f"[{self.agent_id}] anyio.to_thread.run_sync(_generate_summary) returned.")

        if not summary_text:
            log.error(f"[{self.agent_id}] Failed to generate summary for {self.target_source}. Exiting run().")
            return
        
        # 3. Store the summary in memory
        try:
            log.info(f"[{self.agent_id}] Preparing to store summary (length: {len(summary_text)}) for {self.target_source}...")
            summary_doc = MemoryDoc(
                text=summary_text,
                source_agent=self.agent_id,
                tags=["summary", self.target_source], # Tag as summary and link to source
                metadata={
                    "original_source": self.target_source,
                    "summary_model": self.model_name,
                    "chunk_count": len(query_results) # Use original query_results count here
                }
            )
            log.debug(f"[{self.agent_id}] Summary MemoryDoc prepared: Tags={summary_doc.tags}, Meta={summary_doc.metadata}")
            log.debug(f"[{self.agent_id}] Calling anyio.to_thread.run_sync for adapter.add() for summary...")
            # Adapter add is synchronous, run in thread
            summary_id = await anyio.to_thread.run_sync(self.adapter.add, summary_doc)
            if summary_id:
                 log.info(f"[{self.agent_id}] Successfully stored summary (DB ID: {summary_id}) for {self.target_source}.")
            else:
                 log.error(f"[{self.agent_id}] adapter.add() returned None or empty for summary of {self.target_source}. This should not happen.")

        except Exception as add_err:
            log.error(f"[{self.agent_id}] Error storing summary to memory: {add_err}", exc_info=True)
            # Continue to finish log even if add fails

        log.info(f"[{self.agent_id}] <<< Finished SummarizerAgent run() method for target source: {self.target_source}") 