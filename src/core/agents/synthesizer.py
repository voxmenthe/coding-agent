import logging
import os
import anyio
from google import genai
from google.genai import types # May need for config later
import functools # Import functools

from src.core.agents.base import BaseAgent
from src.memory.hybrid_sqlite import HybridSQLiteAdapter
from src.memory.adapter import MemoryDoc

# Configure logging
# logging.basicConfig(level=logging.INFO)
log = logging.getLogger('SynthesizerAgent') # Use specific logger name
log.setLevel(logging.DEBUG) # Set level for this logger

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest"

class SynthesizerAgent(BaseAgent):
    """Agent responsible for synthesizing information from summaries retrieved from memory."""

    def __init__(self, 
                 agent_id: str, 
                 adapter: HybridSQLiteAdapter, 
                 gemini_api_key: str | None = None,
                 gemini_model: str = DEFAULT_GEMINI_MODEL,
                 **kwargs):
        """Initializes the Synthesizer Agent.

        Args:
            agent_id: Unique ID for this agent instance.
            adapter: Instance of HybridSQLiteAdapter for memory access.
            gemini_api_key: API key for Google Gemini. If None, tries OS env GEMINI_API_KEY.
            gemini_model: The Gemini model to use for synthesis.
            **kwargs: Additional configuration passed to BaseAgent.
        """
        log.debug(f"[{agent_id}] Initializing SynthesizerAgent...")
        super().__init__(agent_id=agent_id, **kwargs)
        self.adapter = adapter
        self.model_name = gemini_model # Store the model name
        log.debug(f"[{self.agent_id}] Adapter instance received: {type(adapter)}")

        _api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not _api_key:
            log.error(f"[{self.agent_id}] Gemini API key not provided or found in environment (GEMINI_API_KEY). Synthesis will fail.")
            self.client = None
        else:
            log.debug(f"[{self.agent_id}] Configuring Gemini client for model {self.model_name}...")
            try:
                # Use the new SDK client initialization
                self.client = genai.Client(api_key=_api_key)
                log.info(f"[{self.agent_id}] Gemini client configured successfully for model {self.model_name}.")
            except Exception as e:
                log.error(f"[{self.agent_id}] Error configuring Gemini client: {e}", exc_info=True)
                self.client = None
        
        # Potential future config: Specific sources/topics to synthesize
        # self.synthesis_topic = self.config.get("synthesis_topic")
        log.info(f"SynthesizerAgent [{self.agent_id}] initialized.")

    async def _generate_synthesis(self, text_to_synthesize: str) -> str | None:
        """Generates a synthesis using the Gemini API (synchronous) via the NEW SDK."""
        log.debug(f"[{self.agent_id}] Entering _generate_synthesis (text length: {len(text_to_synthesize)})...")
        if not self.client:
            log.error(f"[{self.agent_id}] Gemini client not available. Cannot generate synthesis.")
            return None
        
        # Simple synthesis prompt - can be refined significantly
        prompt = (
            f"Synthesize the key insights, connections, and potential research directions from the following summaries of different documents:\n\n" 
            f"---\n{text_to_synthesize}\n---\n\n" 
            f"Synthesis:"
        )
        log.debug(f"[{self.agent_id}] Prompt created (first 100 chars): {prompt[:100]}...")
        
        try:
            log.info(f"[{self.agent_id}] Generating synthesis for text (length: {len(text_to_synthesize)}) using model {self.model_name}...")
            # Use the new SDK's generate_content method
            response = self.client.models.generate_content(
                model=self.model_name, 
                contents=prompt,
                # generation_config=... 
                # safety_settings=...
            )
            synthesis = response.text
            log.info(f"[{self.agent_id}] Synthesis generated successfully (length: {len(synthesis)})." )
            log.debug(f"[{self.agent_id}] Generated synthesis (first 100 chars): {synthesis[:100]}...")
            return synthesis
        except Exception as e:
            log.error(f"[{self.agent_id}] Error during Gemini API call for synthesis: {e}", exc_info=True)
            return None

    async def run(self) -> None:
        """The main execution logic for the Synthesizer agent."""
        log.info(f"[{self.agent_id}] >>> Starting SynthesizerAgent run() method...")

        if not self.client:
             log.error(f"[{self.agent_id}] Cannot run because Gemini client is not configured. Exiting run().")
             return

        query_results = [] # Initialize
        combined_text = "" # Initialize
        # 1. Retrieve relevant summaries (or potentially chunks) from memory
        try:
            log.info(f"[{self.agent_id}] Querying memory for summaries...")
            # Query specifically for documents tagged as 'summary'
            # For minimal viability, retrieve all summaries. Refine later if needed.
            # Adapter query methods are synchronous, run in thread
            query_text_val = "document summaries" # General query for summaries
            k_val = 50 # Retrieve a decent number of summaries for synthesis
            filter_tags_val=["summary"] # Filter specifically for summaries
            log.debug(f"[{self.agent_id}] Preparing hybrid_query: query='{query_text_val}', k={k_val}, tags={filter_tags_val}")
            query_func = functools.partial(
                self.adapter.hybrid_query, 
                query_text=query_text_val,
                k=k_val, 
                filter_tags=filter_tags_val
            )
            log.debug(f"[{self.agent_id}] Calling anyio.to_thread.run_sync for adapter.hybrid_query()...")
            query_results = await anyio.to_thread.run_sync(query_func)
            log.debug(f"[{self.agent_id}] adapter.hybrid_query returned {len(query_results)} results.")
            
            if not query_results:
                log.warning(f"[{self.agent_id}] No summaries found in memory to synthesize. Exiting run().")
                return

            log.info(f"[{self.agent_id}] Retrieved {len(query_results)} summaries for synthesis.")
            
            # Combine text from summaries (results are tuples: (MemoryDoc, score))
            # Simple concatenation for now. Could add source info later.
            log.debug(f"[{self.agent_id}] Combining text from {len(query_results)} summaries...")
            combined_text = "\n\n---\n\n".join([
                f"Summary from source '{doc.metadata.get('original_source', 'unknown')}':\n{doc.text}" 
                for doc, score in query_results if doc.text and doc.metadata
            ])
            log.info(f"[{self.agent_id}] Combined text from summaries. Total length: {len(combined_text)}.")

        except Exception as query_err:
            log.error(f"[{self.agent_id}] Error querying memory for summaries: {query_err}", exc_info=True)
            return # Exit run on error
        
        if not combined_text:
            log.warning(f"[{self.agent_id}] Combined text from summaries is empty after processing. Skipping synthesis. Exiting run().")
            return

        # 2. Generate synthesis using LLM (run synchronous _generate_synthesis in thread)
        log.debug(f"[{self.agent_id}] Calling anyio.to_thread.run_sync for _generate_synthesis()...")
        synthesis_text = await anyio.to_thread.run_sync(self._generate_synthesis, combined_text)
        log.debug(f"[{self.agent_id}] anyio.to_thread.run_sync(_generate_synthesis) returned.")

        if not synthesis_text:
            log.error(f"[{self.agent_id}] Failed to generate synthesis. Exiting run().")
            return
        
        # 3. Store the synthesis in memory
        try:
            log.info(f"[{self.agent_id}] Preparing to store synthesis (length: {len(synthesis_text)})...")
            synthesis_doc = MemoryDoc(
                text=synthesis_text,
                source_agent=self.agent_id,
                tags=["synthesis"], # Tag as synthesis
                metadata={
                    "synthesis_model": self.model_name,
                    "summaries_count": len(query_results) # Number of summaries synthesized
                    # Potential: Add IDs of summarized docs
                }
            )
            log.debug(f"[{self.agent_id}] Synthesis MemoryDoc prepared: Tags={synthesis_doc.tags}, Meta={synthesis_doc.metadata}")
            log.debug(f"[{self.agent_id}] Calling anyio.to_thread.run_sync for adapter.add() for synthesis...")
            # Adapter add is synchronous, run in thread
            synthesis_id = await anyio.to_thread.run_sync(self.adapter.add, synthesis_doc)
            if synthesis_id:
                log.info(f"[{self.agent_id}] Successfully stored synthesis (DB ID: {synthesis_id}).")
            else:
                log.error(f"[{self.agent_id}] adapter.add() returned None or empty for synthesis. This should not happen.")

        except Exception as add_err:
            log.error(f"[{self.agent_id}] Error storing synthesis to memory: {add_err}", exc_info=True)
            # Continue to finish log even if add fails

        log.info(f"[{self.agent_id}] <<< Finished SynthesizerAgent run() method.") 