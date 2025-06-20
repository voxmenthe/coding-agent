import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import functools # Added for _make_verbose_tool

import google.generativeai as genai
# types will be accessed via genai.types

# Assuming tools.py might be needed later, add a placeholder import
# from . import tools

logger = logging.getLogger(__name__)

# Fallback if not in config, matches CodeAgent's previous default
DEFAULT_THINKING_BUDGET_FALLBACK = 256

class LLMAgentCore:
    def __init__(self, config: dict, api_key: Optional[str], model_name: str):
        self.config = config
        self.api_key = api_key
        self.model_name = model_name # Ensure this is passed and used

        self.client: Optional[genai.client.Client] = None
        # Async client part is derived from the main client
        self.async_client: Optional[genai.client.Client.AsyncClientPart] = None
        self.chat: Optional[genai.ChatSession] = None # Changed from genai.generative_models.ChatSession

        self.tool_functions: List[Callable] = []
        # Use model_name from args, and thinking_budget from config or fallback
        self.thinking_budget: int = config.get('default_thinking_budget', DEFAULT_THINKING_BUDGET_FALLBACK)
        # Initialize generation_config here, replacing thinking_config
        self.generation_config: Optional[genai.types.GenerationConfig] = genai.types.GenerationConfig()

        self.conversation_history: List[genai.types.Content] = []
        self.current_token_count: int = 0

        if self.api_key:
            self._configure_client()
            if self.client: # Only initialize chat if client configuration was successful
                self._initialize_chat()
        else:
            logger.warning("LLMAgentCore: GEMINI_API_KEY not found. LLM client not initialized.")

    def _configure_client(self):
        """Configures the Google Generative AI client."""
        logger.info("LLMAgentCore: Configuring GenAI client...")
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.async_client = self.client.aio # Initialize async_client
            logger.info("LLMAgentCore: GenAI client configured successfully.")
        except Exception as e:
            logger.error(f"LLMAgentCore: Error configuring GenAI client: {e}", exc_info=True)
            self.client = None # Ensure client is None on failure
            self.async_client = None

    def _initialize_chat(self):
        """Initializes the chat session."""
        if not self.client:
            logger.error("LLMAgentCore: Cannot initialize chat without a configured client.")
            return
        logger.info(f"LLMAgentCore: Initializing chat session with model {self.model_name}...")
        try:
            # Use the current self.conversation_history when creating a new chat session
            self.chat = self.client.chats.create(model=self.model_name, history=self.conversation_history)
            logger.info(f"LLMAgentCore: Chat session initialized/re-initialized with {len(self.conversation_history)} history entries.")
        except Exception as e:
            logger.error(f"LLMAgentCore: Error initializing/re-initializing chat session: {e}", exc_info=True)
            self.chat = None # Ensure chat is None on failure

    def _make_verbose_tool(self, func: Callable) -> Callable:
        """Wrap tool function to log verbose info when called."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"LLMAgentCore: 🔧 Tool called: {func.__name__}, args: {args}, kwargs: {kwargs}")
            result = func(*args, **kwargs)
            # Consider truncating long results for logging if necessary
            result_display = str(result)
            if len(result_display) > 200: # Example length limit
                result_display = result_display[:200] + "..."
            logger.info(f"LLMAgentCore: ▶️ Tool result ({func.__name__}): {result_display}")
            return result
        return wrapper

    def register_tools(self, tools_list: List[Callable], verbose: bool = False):
        """Registers tool functions, optionally wrapping them for verbose logging."""
        if verbose:
            self.tool_functions = [self._make_verbose_tool(f) for f in tools_list]
        else:
            self.tool_functions = tools_list
        logger.info(f"LLMAgentCore: Registered {len(self.tool_functions)} tools. Verbose: {verbose}")
        # self.generation_config can be updated here if needed, but basic init is in __init__
        # For now, removing direct update related to thinking_budget as its equivalent in GenerationConfig is unclear.

    def send_message(self, user_message_text: str,
                     active_files: Optional[List[genai.types.File]] = None,
                     pending_context: Optional[str] = None) -> str:
        """
        Sends a message to the LLM, manages history, and returns the response text.
        """
        if not self.chat:
            logger.error("LLMAgentCore: Chat not initialized. Cannot send message.")
            return "Error: Chat session is not active. Please check API key and initialization."

        message_to_send_text = user_message_text
        if pending_context:
            logger.info("LLMAgentCore: Prepending pending context to the message.")
            message_to_send_text = f"{pending_context}\n\n{message_to_send_text}"

        # --- Construct message content (Text + Files) ---
        message_content_parts: List[Any] = [genai.types.Part(text=message_to_send_text)] # Changed from genai_types
        if active_files:
            # Ensure active_files are of the correct type (genai.types.File or compatible Part)
            # For now, assume they are genai.types.File objects as per typical usage.
            # If they are just paths, they need to be uploaded first (outside this method's scope for now)
            message_content_parts.extend(active_files)
            logger.info(f"LLMAgentCore: Including {len(active_files)} active files in the prompt parts.")

        # The user's turn/message to be added to history and sent
        new_user_content = genai.types.Content(parts=message_content_parts, role="user") # Changed from genai_types

        # Append new user message to our managed history *before* sending
        self.conversation_history.append(new_user_content)
        logger.info(f"LLMAgentCore: Sending message to LLM. History length: {len(self.conversation_history)}")

        # Prepare tool configuration for the send_message call
        # Prepare tool configuration for the send_message call
        # The ChatSession.send_message method takes tools and generation_config directly.

        try:
            # Pass new_user_content directly as 'content'.
            # Pass tool_functions directly to 'tools'.
            # Pass self.generation_config (now an instance of GenerationConfig) to 'generation_config'.
            response = self.chat.send_message(
                content=new_user_content,
                tools=self.tool_functions,
                generation_config=self.generation_config
            )

            agent_response_text = ""
            # Process response: extract text, handle tool calls if any
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                text_parts = [p.text for p in response.candidates[0].content.parts if hasattr(p, "text") and p.text]
                agent_response_text = " ".join(text_parts).strip()

                # Check for function calls if no direct text
                if not agent_response_text:
                    if any(hasattr(p, "function_call") for p in response.candidates[0].content.parts):
                        # If there was a function call, the text might be empty or just confirmation.
                        # The actual tool execution and result handling happens outside LLMAgentCore,
                        # driven by the main application loop which checks for FunctionCall parts.
                        agent_response_text = "[Tool call requested by LLM]"
                        logger.info("LLMAgentCore: LLM requested a tool call.")
                    else:
                        agent_response_text = "[No textual response from LLM]"
            elif not agent_response_text: # Fallback if structure is unexpected
                agent_response_text = "[Empty or malformed response from LLM]"

            # Append agent's response to our managed history
            # Even if it's a tool call message, we record it.
            hist_agent_content = genai.types.Content(role="model", parts=[genai.types.Part(text=agent_response_text)]) # Changed from genai_types
            self.conversation_history.append(hist_agent_content)

            # Calculate token count based on the updated self.conversation_history
            if self.client:
                try:
                    # Ensure history for token counting contains only compatible parts (text/file data)
                    # For now, assume self.conversation_history is correctly formatted.
                    # If file parts are complex (e.g. genai_types.File directly), count_tokens might need specific handling.
                    # The current structure (Content objects with Part(text=...) or Part(file_data=...)) should be fine.
                    token_info = self.client.models.count_tokens(
                        model=self.model_name,
                        contents=self.conversation_history # Send the whole history
                    )
                    self.current_token_count = token_info.total_tokens
                    logger.info(f"LLMAgentCore: Current token count: {self.current_token_count}")
                except Exception as count_err:
                    logger.error(f"LLMAgentCore: Error calculating token count: {count_err}", exc_info=True)
                    # self.current_token_count might be stale or could be set to an error indicator

            return agent_response_text

        except Exception as e:
            logger.error(f"LLMAgentCore: Error sending message to LLM: {e}", exc_info=True)
            # Construct an error message to return and add to history
            error_response_text = f"Error: Could not get response from LLM. Details: {str(e)}"
            # Append this error as a "model" response in history
            hist_error_content = genai.types.Content(role="model", parts=[genai.types.Part(text=error_response_text)]) # Changed from genai_types
            self.conversation_history.append(hist_error_content)
            # Optionally, try to update token count here as well
            return error_response_text

    def reset_chat(self):
        """Resets the conversation history, token count, and re-initializes the chat session."""
        logger.info("LLMAgentCore: Resetting chat...")
        self.conversation_history = []
        self.current_token_count = 0
        if self.client:
            self._initialize_chat() # Re-initialize the chat session with the LLM
            logger.info("LLMAgentCore: Chat reset and re-initialized successfully.")
        else:
            logger.warning("LLMAgentCore: Client not available. Chat history cleared, but session not re-initialized.")

    def clear_history_turns(self, turns_to_keep: int = 5):
        """
        Clears older history, keeping a specified number of recent turns.
        A "turn" consists of one user message and one model response.
        """
        logger.info(f"LLMAgentCore: Clearing history, attempting to keep last {turns_to_keep} turns.")
        if turns_to_keep <= 0:
            self.conversation_history = []
        else:
            # Each turn is 2 items (user, model)
            items_to_keep = turns_to_keep * 2
            if len(self.conversation_history) > items_to_keep:
                self.conversation_history = self.conversation_history[-items_to_keep:]

        # Recalculate token count
        if self.client and self.conversation_history:
            try:
                token_info = self.client.models.count_tokens(
                    model=self.model_name,
                    contents=self.conversation_history
                )
                self.current_token_count = token_info.total_tokens
                logger.info(f"LLMAgentCore: History cleared. New token count: {self.current_token_count}")
            except Exception as count_err:
                logger.error(f"LLMAgentCore: Error recalculating token count after clearing history: {count_err}", exc_info=True)
        elif not self.conversation_history:
            self.current_token_count = 0
            logger.info("LLMAgentCore: History cleared completely. Token count reset to 0.")

    def clear_history_by_tokens(self, tokens_to_clear_target: int) -> tuple[int, int]:
        """
        Clears older history by a target number of tokens.
        Returns a tuple of (messages_removed_count, actual_tokens_cleared_count).
        """
        if not self.conversation_history:
            logger.info("LLMAgentCore: History is already empty. Nothing to clear by tokens.")
            return 0, 0
        if not self.client:
            logger.error("LLMAgentCore: Client not available. Cannot accurately clear tokens by count.")
            # Potentially could clear by message count as a fallback, but returning 0,0 for now.
            return 0, 0

        logger.info(f"LLMAgentCore: Attempting to clear approx. {tokens_to_clear_target} tokens from history.")

        new_history = list(self.conversation_history)
        tokens_counted_for_removal = 0
        messages_removed_count = 0

        temp_history_for_counting = []

        while tokens_counted_for_removal < tokens_to_clear_target and new_history:
            first_message = new_history[0]
            message_tokens = 0
            try:
                # Count tokens for the message being considered for removal
                temp_history_for_counting.append(first_message)
                count_response = self.client.models.count_tokens(
                    model=self.model_name,
                    contents=temp_history_for_counting # Count only this message in context of prior kept ones
                )
                # We need token count of *just this message*.
                # A simple way is to count history with and without it, if API allows counting single Content.
                # Or, if count_tokens on a single Content object works:
                single_message_count_response = self.client.models.count_tokens(model=self.model_name, contents=[first_message])
                message_tokens = single_message_count_response.total_tokens
                temp_history_for_counting.pop() # remove after counting if only counting one by one

            except Exception as e_count:
                logger.error(f"LLMAgentCore: Could not count tokens for a message during clear_history_by_tokens: {e_count}. Using estimate (75).", exc_info=True)
                message_tokens = 75 # Fallback estimate

            tokens_counted_for_removal += message_tokens
            new_history.pop(0)
            messages_removed_count += 1

        logger.info(f"LLMAgentCore: After token counting pass, {messages_removed_count} messages ({tokens_counted_for_removal} tokens) selected for removal.")

        # Ensure history starts with a 'user' message if not empty
        additional_messages_removed_for_role = 0
        while new_history and new_history[0].role != "user":
            logger.warning("LLMAgentCore: History after token clear pass starts with a model turn. Removing additional leading model messages.")
            new_history.pop(0)
            messages_removed_count += 1
            additional_messages_removed_for_role += 1
        if additional_messages_removed_for_role > 0:
            logger.info(f"LLMAgentCore: Removed {additional_messages_removed_for_role} additional model messages to ensure user turn start.")

        self.conversation_history = new_history

        # Recalculate current_token_count
        if not self.conversation_history:
            self.current_token_count = 0
        else:
            try:
                self.current_token_count = self.client.models.count_tokens(
                    model=self.model_name,
                    contents=self.conversation_history
                ).total_tokens
            except Exception as e_recount:
                logger.error(f"LLMAgentCore: Error recounting tokens after clear_history_by_tokens: {e_recount}. Token count may be inaccurate.", exc_info=True)
                self.current_token_count = -1 # Indicate error

        # Re-initialize chat with the modified history
        if self.client:
            self._initialize_chat() # This will now use the updated self.conversation_history

        logger.info(f"LLMAgentCore: Cleared {messages_removed_count} message(s). Actual tokens cleared approx {tokens_counted_for_removal}. New total tokens: {self.current_token_count if self.current_token_count != -1 else 'Error counting'}")
        return messages_removed_count, tokens_counted_for_removal

    def set_thinking_budget(self, budget: int):
        """Sets the thinking budget. Note: Direct impact on GenerationConfig is TBD."""
        if 0 <= budget <= 24000: # Assuming same validation as before
            self.thinking_budget = budget
            # self.generation_config currently has no direct 'thinking_budget'.
            # If there were other parameters in GenerationConfig to set based on this, they would go here.
            # For example, if tool execution budget was part of it.
            logger.info(f"LLMAgentCore: Thinking budget value set to: {self.thinking_budget}. Effect on GenerationConfig is pending SDK feature mapping.")
            # Example: If GenerationConfig had a relevant field:
            # self.generation_config.some_tool_execution_parameter = budget
        else:
            logger.warning(f"LLMAgentCore: Invalid thinking budget value: {budget}. Must be between 0 and 24000.")


# Example usage (for testing within the file if needed)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for more verbose output during testing

    # --- Mock Tools for Testing ---
    def mock_search_tool(query: str) -> str:
        return f"Search results for: {query}"

    def mock_read_file_tool(path: str) -> str:
        return f"Contents of file: {path}"

    # --- Test Setup ---
    sample_config_main = {
        'model_name': os.getenv("GEMINI_MODEL_NAME", 'gemini-1.5-flash-latest'),
        'default_thinking_budget': 150, # Lower for testing
        'verbose': True, # Test verbose tool logging
    }

    # API key is essential
    sample_api_key_main = os.getenv("GEMINI_API_KEY")

    if not sample_api_key_main:
        logger.error("Please set GEMINI_API_KEY environment variable for this example to run.")
    else:
        logger.info(f"Using Model: {sample_config_main['model_name']}")
        logger.info("--- Initializing LLMAgentCore ---")
        core_agent_main = LLMAgentCore(
            config=sample_config_main,
            api_key=sample_api_key_main,
            model_name=sample_config_main['model_name']
        )

        if core_agent_main.client and core_agent_main.chat:
            logger.info("--- LLMAgentCore initialized successfully ---")

            # Register mock tools
            core_agent_main.register_tools([mock_search_tool, mock_read_file_tool], verbose=True)

            # Test send_message
            logger.info("--- Test 1: Sending a simple message ---")
            response1 = core_agent_main.send_message("Hello, LLM agent core!")
            logger.info(f"Response 1: {response1}")
            logger.info(f"History after msg 1: {len(core_agent_main.conversation_history)} entries, Token count: {core_agent_main.current_token_count}")

            # Test send_message with pending_context
            logger.info("\n--- Test 2: Sending message with pending_context ---")
            response2 = core_agent_main.send_message("Tell me about this.", pending_context="Some important background info here.")
            logger.info(f"Response 2: {response2}")
            logger.info(f"History after msg 2: {len(core_agent_main.conversation_history)} entries, Token count: {core_agent_main.current_token_count}")

            # Test clear_history_tokens
            logger.info("\n--- Test 3: Clearing history (keeping 1 turn) ---")
            core_agent_main.clear_history_turns(turns_to_keep=1)
            logger.info(f"History after clear: {len(core_agent_main.conversation_history)} entries, Token count: {core_agent_main.current_token_count}")
            if len(core_agent_main.conversation_history) == 2: # 1 user, 1 model
                 logger.info("History clear (1 turn) seems successful.")
            else:
                 logger.error(f"History clear (1 turn) failed. Expected 2 entries, got {len(core_agent_main.conversation_history)}")


            # Test reset_chat
            logger.info("\n--- Test 4: Resetting chat ---")
            core_agent_main.reset_chat()
            logger.info(f"History after reset: {len(core_agent_main.conversation_history)} entries, Token count: {core_agent_main.current_token_count}")
            if not core_agent_main.conversation_history and core_agent_main.current_token_count == 0:
                logger.info("Chat reset successful.")
            else:
                logger.error("Chat reset failed.")

            # Test sending message after reset
            logger.info("\n--- Test 5: Sending message after reset ---")
            response3 = core_agent_main.send_message("Are you fresh now?")
            logger.info(f"Response 3: {response3}")
            logger.info(f"History after msg 3: {len(core_agent_main.conversation_history)} entries, Token count: {core_agent_main.current_token_count}")

            # Test message that might trigger a tool (if model is capable and tools are understood)
            # This is harder to deterministically test without a sophisticated mock model.
            logger.info("\n--- Test 6: Sending a message that might use a tool ---")
            response4 = core_agent_main.send_message("Search for information about Large Language Models.")
            logger.info(f"Response 4: {response4}")
            logger.info(f"History after msg 4: {len(core_agent_main.conversation_history)} entries, Token count: {core_agent_main.current_token_count}")
            # Check if response indicates a tool call (model dependent)
            if "[Tool call requested by LLM]" in response4:
                logger.info("Model indicated a tool call as expected (potentially).")
            elif "Search results for:" in response4 : # If model directly uses tool-like phrasing
                 logger.info("Model provided a direct response that looks like tool output.")
            else:
                 logger.info("Model provided a general response.")


        else:
            logger.error("--- LLMAgentCore initialization FAILED (client or chat not available) ---")
            if not core_agent_main.api_key:
                logger.error("Reason: API key is missing.")
            if not core_agent_main.client:
                logger.error("Reason: GenAI Client failed to initialize.")
            if not core_agent_main.chat:
                logger.error("Reason: Chat session failed to initialize.")
