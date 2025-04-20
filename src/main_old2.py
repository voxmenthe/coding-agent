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

import google.genai as genai
from google.genai import types
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
        self.api_key = config.get_api_key() # Still need the key for Langchain/explicit client
        self.model_name = config.get_model_name()
        self.verbose = config.is_verbose()
        self.client = None
        self.model = None
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
                logger.warning("⚠️ GEMINI_API_KEY/GOOGLE_API_KEY not found explicitly. Relying on environment variable for genai.Client(). Ensure GOOGLE_API_KEY is set.")
                # sys.exit(1)
            else:
                 # Still configure the API key for potential direct use or Langchain
                 # genai.configure is not used with genai.Client() in the new SDK
                 pass

            # Initialize the client using the new SDK style
            # It will automatically pick up GOOGLE_API_KEY env var if api_key is not passed
            self.client = genai.Client()

            # Model interaction is done via client.get_model or client.generate_content etc.
            # Let's get the specific model instance we'll use for chat
            self.model = self.client.get_model(f"models/{self.model_name}") # Use client.get_model

            # Check if the model instance supports tools directly (new SDK might differ)
            # We might need to pass tools during chat start or send_message
            # self.model = genai.GenerativeModel(
            #    model_name=self.model_name,
            #    tools=self.tool_schemas, # Pass generated tool schemas here?
            #    safety_settings={
            #        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            #        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            #        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            #        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            #    }
            # )

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
            self.chat_session = self.model.start_chat(
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

    async def browse_web(self, params: dict) -> str:
        """Tool: Browse the web using browser_use. Handles URLs and tasks.
        params keys:
            url (str): URL to open (required if no context/task implies it).
            task (str): Specific task to perform (e.g., "Click login", "Extract prices").
            headless (bool): Run headless (default uses agent setting, often True).
            highlight (bool): Highlight elements (default False).
            record (bool): Save recording (default False).
            wait_time (float): Wait seconds for network idle (default 5.0).
        Returns HTML content, task result, or error string.
        """
        url = params.get("url")
        task = params.get("task", "Describe the content of this page.") # Default task
        # Defaults for browser settings - can be overridden by params if needed
        headless = params.get("headless", True) # Default to headless
        highlight = params.get("highlight", False)
        record = params.get("record", False)
        wait_time = params.get("wait_time", 5.0)

        logger.info(f"\n🧭 Tool: browse_web params: url={url}, task='{task}', headless={headless}, highlight={highlight}, record={record}, wait_time={wait_time}")

        # Check if browser context and LLM are initialized
        if not self.browser_context:
            return "Error: Browser context is not initialized. Cannot browse."
        if not self.llm:
            return "Error: LLM for browser agent is not initialized. Cannot browse."

        try:
            # Configure browser settings for this specific call if needed
            # This might override the context's initial config - check browser_use docs
            context_config_overrides = BrowserContextConfig(
                wait_for_network_idle_page_load_time=wait_time,
                highlight_elements=highlight,
                save_recording_path="./recordings" if record else None
            )
            # TODO: Check if context config can be updated per-call or needs new context
            # For now, we assume the Agent uses the provided context's config.

            # Set up initial actions *only if* a specific URL is given for this task
            initial_actions = []
            if url:
                initial_actions.append({"open_tab": {"url": url}})
            elif not self.browser_context.pages: # If no URL and no pages open, open default
                default_url = "https://www.google.com"
                logger.warning(f"⚠️ No URL provided and no active pages. Opening default: {default_url}")
                initial_actions.append({"open_tab": {"url": default_url}})

            # Create and run the agent using the persistent context
            agent = Agent(
                task=task,
                llm=self.llm,
                browser=self.browser, # Pass the main browser instance
                browser_context=self.browser_context, # Use the persistent context
                use_vision=True,
                generate_gif=record,
                initial_actions=initial_actions if initial_actions else None
                # config=context_config_overrides # Check if Agent accepts config override
            )

            # Run the agent asynchronously
            result = await agent.run()

            # Get the final result
            final_result = result.final_result() if result else "No result from browser agent."
            logger.info(f"  -> Browser agent final result: {final_result[:200]}...")
            return final_result

        except Exception as e:
            logger.error(f"\n⚠️ browse_web exception: {e}\n{traceback.format_exc()}")
            # Fallback to simple HTTP GET if browser automation fails?
            # Let's return the error for now, as fallback might lose context.
            # Consider adding fallback using 'read_url_content' tool if needed.
            # return read_url_content(url=url) # Example fallback
            return f"Error during browser operation: {e}"

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
