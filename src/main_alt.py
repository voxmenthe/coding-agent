# -*- coding: utf-8 -*-

# --- Main Code Agent Logic ---

import asyncio
import functools
import logging
import os
import sys
import textwrap
import traceback
import argparse

import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_google_genai import ChatGoogleGenerativeAI # Moved here

from src.config import Config
# Import base tools (browser tools will be added dynamically if browser init succeeds)
from src.tools import (
    arxiv_search,
    bash_command,
    edit_file,
    exit_agent,
    get_file_content,
    github_read_issue, # Assuming this exists and is imported correctly
    list_files,        # Assuming this exists and is imported correctly
    python_interpreter,# Assuming this exists and is imported correctly
    read_url_content,
    write_to_file,
)
from browser_use import Browser, Agent
from browser_use.browser.context import BrowserContext
from browser_use.utils.settings import BrowserConfig, BrowserContextConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)-8s [%(name)s] %(message)s')
logger = logging.getLogger("CodeAgent")

# Constants
TEXT_WRAP_WIDTH = 100
MAX_THINKING_BUDGET = 24000
DEFAULT_THINKING_BUDGET = 256

class CodeAgent:
    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.get_api_key()
        self.model_name = config.get_model_name()
        self.verbose = config.is_verbose()
        self.client = None
        self.chat_session = None
        self.llm = None # To store the Langchain LLM instance
        self.thinking_budget = config.get_default_thinking_budget()
        self.tools = [] # Registered tools dictionary for API call
        self.conversation_history = [] # For display/context, not token counting

        self.browser: Browser | None = None
        self.browser_context: BrowserContext | None = None

        # Base tool functions before potential browser init
        # Removed browse_web and browse_url_on_web from this initial list
        self.tool_functions_base = [
            arxiv_search,
            bash_command,
            edit_file,
            exit_agent,
            get_file_content,
            github_read_issue,
            list_files,
            python_interpreter,
            read_url_content,
            write_to_file,
        ]
        self.tool_functions = self.tool_functions_base[:] # Copy initially

        # Store tool functions with metadata for registration
        self.tool_metadata = {}

    async def _initialize_browser(self):
        """Initializes the browser and browser context asynchronously."""
        if self.browser and self.browser_context:
             logger.info("⚪ Browser already initialized.")
             return # Avoid re-initialization

        try:
            logger.info("🌐 Initializing browser...")
            # Use default BrowserConfig for now
            self.browser = Browser()
            # Use default BrowserContextConfig for now
            self.browser_context = await self.browser.new_context()
            logger.info("✅ Browser initialized successfully.")

            # Add the browse_web method to tool functions *after* context is ready
            if hasattr(self, 'browse_web') and callable(self.browse_web):
                 # Ensure it's not already added
                 if self.browse_web not in self.tool_functions:
                     self.tool_functions.append(self.browse_web)
                     logger.info("➕ Added 'browse_web' tool.")
            else:
                 logger.warning("⚠️ browse_web method not found on CodeAgent, skipping tool addition.")

        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}\n{traceback.format_exc()}")
            self.browser = None
            self.browser_context = None # Ensure context is None if browser fails

    async def close_browser(self):
        """Closes the browser context and browser if they exist."""
        if self.browser_context:
            try:
                await self.browser_context.close()
                logger.info("⚪ Browser context closed.")
            except Exception as e:
                logger.error(f"❌ Error closing browser context: {e}")
            finally:
                 self.browser_context = None # Ensure it's None even if close fails
        if self.browser:
            try:
                await self.browser.close()
                logger.info("⚪ Browser closed.")
            except Exception as e:
                logger.error(f"❌ Error closing browser: {e}")
            finally:
                 self.browser = None # Ensure it's None even if close fails

    def _configure_client(self):
        """Configures the generative AI client and Langchain LLM."""
        logger.info("⚒️ Configuring genai client...")
        try:
            if not self.api_key:
                logger.error("❌ GEMINI_API_KEY not found. Please set the environment variable.")
                sys.exit(1)

            genai.configure(
                api_key=self.api_key,
                # transport="rest", # Optional: Use REST for broader compatibility
                # client_options={"api_endpoint": "..."} # Optional: Specify endpoint
            )
            # Use a compatible model name for the Python SDK client
            # TODO: Verify model name compatibility if issues arise
            self.client = genai.GenerativeModel(
                 model_name=self.model_name, # e.g., 'gemini-1.5-flash-latest' or 'gemini-1.5-pro-latest'
                 safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
             )

            # Setup Langchain LLM for browser_use Agent
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name, # Use the same model
                google_api_key=self.api_key,
                # Optional: Add safety_settings if needed for Langchain wrapper too
                # safety_settings={...}
                convert_system_message_to_human=True # Often needed for Gemini
            )

            logger.info("✅ Client configured successfully.")
        except Exception as e:
            logger.error(f"❌ Error configuring client: {e}")
            traceback.print_exc()
            sys.exit(1)

    async def _initialize_chat(self):
        """Initializes a new chat session asynchronously."""
        logger.info("⚒️ Initializing chat session...")
        if not self.client:
             logger.error("❌ Client not configured. Cannot initialize chat.")
             return
        try:
            # Start_chat is synchronous in google-generativeai v0.5+
            # No await needed here. Handle potential history if needed.
            self.chat_session = self.client.start_chat(
                enable_automatic_function_calling=True,
                history=[] # Start fresh history for this session
            )
            self.conversation_history = [] # Clear local display history
            logger.info("✅ Chat session initialized.")
        except Exception as e:
             logger.error(f"❌ Error initializing chat session: {e}")
             traceback.print_exc()


    def _register_tools(self):
        """Registers tool functions, applying verbose wrapper if enabled."""
        logger.info("⚒️ Registering tools...")
        # Use the potentially updated self.tool_functions list
        tools_dict = {}
        self.tool_metadata = {} # Clear previous metadata
        for func in self.tool_functions:
            tool_name = func.__name__
            original_func = func # Assume non-verbose initially
            verbose_func = func

            # Check if it's already a wrapped verbose function
            # This logic might need adjustment depending on how _make_verbose_tool works
            is_already_verbose = hasattr(func, '__wrapped__') and func.__name__ == '_verbose_wrapper'

            if self.verbose and not is_already_verbose:
                # Need to get the original if func is already wrapped somehow?
                # Or assume func is the base function here.
                original_func = func # Store original before wrapping
                verbose_func = self._make_verbose_tool(func) # Create verbose wrapper
                tools_dict[tool_name] = verbose_func # Register the verbose version
            else:
                # If not verbose or already wrapped, register func directly
                # but still store original_func for async check later
                # If already verbose, find the original?
                if is_already_verbose:
                     original_func = getattr(func, '__wrapped__', func) # Try to get original

                tools_dict[tool_name] = func # Register whatever func is
                verbose_func = func # Keep verbose same as original if not wrapping

            # Store metadata, ensuring original is the unwrapped callable
            self.tool_metadata[tool_name] = {'original': original_func, 'verbose': verbose_func}

        logger.info(f"✅ Registered {len(tools_dict)} tools: {list(tools_dict.keys())}")
        # Return the dictionary expected by genai.GenerativeModel tool_config
        # Important: This should be the schema/declaration, not the functions themselves
        # Let's return the functions for now, assuming start_chat handles it,
        # but this might need adjustment based on google-generativeai version.
        # For automatic function calling, just enabling it in start_chat might be enough.
        # We need the registered functions later in _handle_function_call.
        return tools_dict # Keep returning dict for _handle_function_call lookup


    async def _send_message(self, user_input: str):
        """Sends a message to the chat session and handles responses/function calls."""
        if not self.chat_session:
            return "Chat session not initialized."

        try:
            # --- Append user message to local history for display ---
            self.conversation_history.append({"role": "user", "parts": [user_input]})

            # --- Send Message to Gemini ---
            # Use the potentially updated registered tools dictionary
            # For automatic function calling, tool_config might not be needed here anymore
            # if enable_automatic_function_calling=True was set in start_chat.
            # Verify this with the library version documentation.
            # Let's assume we still pass it for clarity or older versions.
            tool_config = types.ToolConfig(
                 function_calling_config=types.FunctionCallingConfig(
                     mode=types.FunctionCallingConfig.Mode.AUTO # or ANY or NONE
                 )
             )
            response = await self.chat_session.send_message_async(
                user_input,
                # stream=False, # Use False for simpler handling now
                # tool_config=tool_config # Pass tool config if needed by version
            )

            # --- Handle Function Calls ---
            while response.candidates[0].content.parts and response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
                logger.info(f"⚙️ Function Call requested: {function_call.name}")
                function_response_content = await self._handle_function_call(function_call)

                # --- Append function call & response to local history ---
                self.conversation_history.append({"role": "model", "parts": [function_call]})
                # Ensure function_response_content is serializable if needed
                self.conversation_history.append({"role": "function", "parts": [function_response_content.parts[0]]}) # parts[0] assuming FunctionResponse structure

                # --- Send Function Response back to Gemini ---
                response = await self.chat_session.send_message_async(
                    function_response_content,
                    # tool_config=tool_config # Pass tool config if needed
                )
                # Loop continues if another function call is made

            # --- Process Final Agent Response ---
            response_text = ""
            if response.candidates and response.candidates[0].content.parts:
                 response_text = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, 'text'))
                 # --- Append agent response to local history ---
                 self.conversation_history.append({"role": "model", "parts": [part for part in response.candidates[0].content.parts]}) # Store full parts
            else:
                 logger.warning("⚠️ Agent response seemed empty.")
                 response_text = "(Agent response was empty)"


            return response_text

        except Exception as e:
            logger.error(f"❌ Error during message sending/processing: {e}\n{traceback.format_exc()}")
            return f"Error processing message: {e}"

    async def _handle_function_call(self, function_call):
        """Handles a function call from the model."""
        function_name = function_call.name
        args = function_call.args

        logger.info(f"  -> Executing: {function_name} with args: {args}")

        if function_name not in self.tool_metadata:
             result_str = f"Error: Tool '{function_name}' not found or not registered correctly."
             logger.error(result_str)
        else:
            tool_info = self.tool_metadata[function_name]
            # Always use the 'verbose' version for execution, as it might be wrapped
            # The verbose wrapper handles calling the 'original' internally.
            tool_to_call = tool_info['verbose']
            original_func = tool_info['original'] # Get original for async check

            try:
                # Check if the *original* function is async
                if asyncio.iscoroutinefunction(original_func):
                     logger.debug(f"   -> Awaiting async tool: {function_name}")
                     # Await the execution (the verbose wrapper should handle calling await on original)
                     # If verbose wrapper is sync, this needs rework. Assume wrapper handles it.
                     # Let's explicitly await the *original* if not verbose,
                     # and assume the verbose wrapper handles the await if verbose.
                     if self.verbose:
                         # Assume verbose wrapper handles await internally
                         result = tool_to_call(**args)
                     else:
                          # Call original async func directly
                         result = await original_func(**args)
                else:
                    # Synchronous tool call
                    logger.debug(f"   -> Calling sync tool: {function_name}")
                    result = tool_to_call(**args)

                # Ensure result is a string for the API response
                result_str = str(result)
                logger.info(f"  <- Tool {function_name} result type: {type(result)}") # Log type for debug

            except Exception as e:
                error_msg = f"Error executing tool '{function_name}': {e}"
                logger.error(f"  <- {error_msg}\n{traceback.format_exc()}")
                result_str = error_msg

        # Prepare response content for the model (FunctionResponse format)
        response_content = types.Content(
             parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                         name=function_name,
                         response={"content": result_str} # API expects specific structure
                    )
                )
             ]
        )
        return response_content


    # <<< Define async def browse_web(self, params: dict) method here in the next step >>>


    def _make_verbose_tool(self, func):
        """Wrap tool function to print verbose info when called."""
        # Ensure we are wrapping the original function if it was already verbose
        original_func = getattr(func, '__wrapped__', func)

        @functools.wraps(original_func) # Wrap the original
        def _verbose_wrapper(*args, **kwargs):
            # This wrapper itself is SYNC. If original_func is ASYNC,
            # this wrapper needs to run it in an event loop or be async itself.
            # For now, assume sync or that caller handles await. Needs refinement.
            logger.info(f"🛠️ Tool: {original_func.__name__} args: {args}, kwargs: {kwargs}")
            try:
                # How to handle await if original_func is async? This is tricky.
                # If we are in an async context already, maybe we can await?
                # Let's assume the caller (_handle_function_call) manages await correctly.
                result = original_func(*args, **kwargs) # Call the original
                # Truncate long results for logging
                result_display = str(result)
                if len(result_display) > 200:
                     result_display = result_display[:200] + "... (truncated)"
                logger.info(f"  ✅ Result ({original_func.__name__}): {result_display}")
                return result
            except Exception as e:
                 logger.error(f"  ❌ Error in tool {original_func.__name__}: {e}")
                 raise # Re-raise exception to be caught by caller

        # Store original func info on the wrapper for later checks
        _verbose_wrapper.__wrapped__ = original_func
        _verbose_wrapper._is_verbose_wrapper = True
        return _verbose_wrapper

async def main():
    """Main asynchronous function to run the Code Agent."""
    parser = argparse.ArgumentParser(description="Run the Code Agent")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose tool logging')
    args = parser.parse_args()

    config = Config()
    agent = CodeAgent(config)
    # Override verbosity from args
    agent.verbose = args.verbose

    try:
        # --- Initialization (Order matters) ---
        agent._configure_client() # Sync setup (includes LLM)
        await agent._initialize_chat() # Async setup
        await agent._initialize_browser() # Async setup (conditionally adds browse_web)

        # Register tools *after* browser init potentially adds browse_web
        agent.tools = agent._register_tools() # Sync registration for now

        logger.info("⚒️ Agent ready. Ask me anything. Type 'exit' to quit.")
        logger.info("   Use '/upload <path/to/file.pdf>' to seed PDF into context.")
        logger.info("   Use '/reset' to clear the chat and start fresh.")

        # --- Interaction Loop ---
        while True:
            try:
                # Get thinking budget
                budget_input = await asyncio.to_thread(
                    input,
                    f"Enter thinking budget (0 to {MAX_THINKING_BUDGET}) for this session [{agent.thinking_budget}]: "
                )
                if budget_input:
                    agent.thinking_budget = int(budget_input)
                    if not 0 <= agent.thinking_budget <= MAX_THINKING_BUDGET:
                        agent.thinking_budget = DEFAULT_THINKING_BUDGET
                        logger.warning(f"⚠️ Invalid budget. Using default: {agent.thinking_budget}")
                # TODO: Update thinking budget in API call if library supports it dynamically
            except ValueError:
                logger.warning(f"⚠️ Invalid input. Using previous budget: {agent.thinking_budget}")

            # Get user input
            user_input = await asyncio.to_thread(input, f"\n🔵 You ({len(agent.conversation_history)} turns): ") # Simple turn counter
            if not user_input:
                 continue
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "/reset":
                logger.info("🔄 Resetting chat session...")
                await agent._initialize_chat() # Re-initialize chat
                await agent._initialize_browser() # Re-initialize browser too? Or keep existing? Let's re-init for now.
                agent.tools = agent._register_tools() # Re-register tools
                logger.info("✅ Chat session reset.")
                continue
            # TODO: Add /upload functionality using tools maybe?

            logger.info("\n⏳ Sending message and processing...")
            response = await agent._send_message(user_input)
            logger.info(f"\n🟢 Agent:")
            for line in textwrap.wrap(response, width=TEXT_WRAP_WIDTH):
                     print(f"   {line}")

    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupt received.")
    except Exception as e:
        logger.error(f"💥 An unexpected error occurred in the main loop: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("\n⚪ Cleaning up resources...")
        await agent.close_browser() # Ensure browser is closed
        logger.info("\n👋 Goodbye!")

if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Exiting gracefully.")
    except Exception as e:
         logger.critical(f"🚨 Unhandled exception at top level: {e}\n{traceback.format_exc()}")
         sys.exit(1)
