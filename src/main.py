# -*- coding: utf-8 -*-

import asyncio
import functools
import logging
import os
import sys
import textwrap
import traceback
import argparse
import inspect # For schema generation

from google import genai # Correct new SDK import
from google.genai.types import HarmCategory, HarmBlockThreshold # Keep for safety settings
# google.generativeai.types doesn't seem correct based on new SDK structure for FunctionCall/FunctionResponse etc.
# Let's rely on genai.FunctionCall and genai.FunctionResponse directly.

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import Config
# Import base tools
from src.tools import (
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
)
# Import browser components
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
        self.client = None # genai.Client instance
        self.model = None # genai.GenerativeModel instance (retrieved via client)
        self.chat_session = None # model.start_chat() result
        self.llm = None # Langchain LLM instance
        self.thinking_budget = config.get_default_thinking_budget()

        # Tool storage
        self.tool_functions = [] # List of callable tool functions
        self.tool_schemas = [] # List of genai.Tool schemas for the API
        self.tool_metadata = {} # Dict mapping name to {'original': callable, 'verbose': callable}

        self.conversation_history = [] # For display/context

        self.browser: Browser | None = None
        self.browser_context: BrowserContext | None = None

        # Base tool functions before potential browser init
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

    # --- Browser Initialization and Management ---
    async def _initialize_browser(self):
        """Initializes the browser and browser context asynchronously."""
        if self.browser and self.browser_context:
             logger.info("⚪ Browser already initialized.")
             return

        try:
            logger.info("🌐 Initializing browser...")
            self.browser = Browser()
            self.browser_context = await self.browser.new_context()
            logger.info("✅ Browser initialized successfully.")

            # Add the browse_web method to tool functions *after* context is ready
            if hasattr(self, 'browse_web') and callable(self.browse_web):
                 if self.browse_web not in self.tool_functions:
                     self.tool_functions.append(self.browse_web)
                     # Re-register tools to include the new one's schema
                     self._register_tools()
                     logger.info(f"➕ Added 'browse_web' tool and updated schemas.")
            else:
                 logger.warning("⚠️ browse_web method not found on CodeAgent, skipping tool addition.")

        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}\n{traceback.format_exc()}")
            self.browser = None
            self.browser_context = None

    async def close_browser(self):
        """Closes the browser context and browser if they exist."""
        if self.browser_context:
            try:
                await self.browser_context.close()
                logger.info("⚪ Browser context closed.")
            except Exception as e:
                logger.error(f"❌ Error closing browser context: {e}")
            finally:
                 self.browser_context = None
        if self.browser:
            try:
                await self.browser.close()
                self.browser = None # Ensure it's None after closing
                logger.info("⚪ Browser closed.")
            except Exception as e:
                logger.error(f"❌ Error closing browser: {e}")
            finally:
                 self.browser = None # Ensure it's None even if close fails

    # --- Client and Chat Initialization ---
    def _configure_client(self):
        """Configures the genai client and Langchain LLM."""
        logger.info("⚒️ Configuring genai client...")
        try:
            # Initialize the client using the new SDK style
            # It will automatically pick up GOOGLE_API_KEY env var if api_key is not passed
            # Explicitly passing key if available, otherwise relies on env var
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("   Using provided API key for genai.Client.")
            else:
                 self.client = genai.Client()
                 logger.warning("   No API key provided explicitly. Relying on GOOGLE_API_KEY env var for genai.Client().")

            # Get the specific model instance we'll use
            # Models are specified like 'models/gemini-1.5-flash-latest'
            model_id = f"models/{self.model_name}"
            logger.info(f"   Getting model: {model_id}")
            self.model = self.client.get_model(model_id)

            # Setup Langchain LLM for browser_use Agent
            if self.api_key:
                self.llm = ChatGoogleGenerativeAI(
                    # Langchain might need model name without 'models/' prefix
                    model=self.config.get_model_name(strip_prefix=True),
                    google_api_key=self.api_key,
                    convert_system_message_to_human=True
                )
            else:
                logger.warning("⚠️ Cannot initialize Langchain LLM for browser: API key not available.")
                self.llm = None

            logger.info("✅ Client configured successfully.")
        except Exception as e:
            logger.error(f"❌ Error configuring client: {e}")
            traceback.print_exc()
            sys.exit(1)

    async def _initialize_chat(self):
        """Initializes a new chat session asynchronously."""
        logger.info("⚒️ Initializing chat session...")
        if not self.model:
             logger.error("❌ Model not available. Cannot initialize chat.")
             return

        # Generate tool schemas *before* starting chat
        self._register_tools()

        try:
            # Start chat with automatic function calling enabled.
            # The model uses the schemas generated by _register_tools implicitly.
            # Safety settings can be passed here if needed per chat.
            self.chat_session = self.model.start_chat(
                enable_automatic_function_calling=True,
                history=[] # Start fresh history for this session
            )
            self.conversation_history = [] # Clear *local* display history
            logger.info("✅ Chat session initialized.")
        except Exception as e:
             logger.error(f"❌ Error initializing chat session: {e}")
             traceback.print_exc()

    # --- Tool Registration and Handling ---
    def _register_tools(self):
        """Generates tool schemas and prepares functions for execution."""
        logger.info("⚒️ Registering tools and generating schemas...")
        # Use the potentially updated self.tool_functions list
        tool_schemas = []
        self.tool_metadata = {} # Clear previous metadata

        for func in self.tool_functions:
            tool_name = func.__name__
            original_func = func
            verbose_func = func

            # --- Verbose Wrapping ---
            is_already_verbose = hasattr(func, '_is_verbose_wrapper') and func._is_verbose_wrapper
            if self.verbose and not is_already_verbose:
                original_func = func
                verbose_func = self._make_verbose_tool(func)
            else:
                if is_already_verbose:
                     original_func = getattr(func, '__wrapped__', func)
                verbose_func = func

            # Store metadata for execution lookup
            self.tool_metadata[tool_name] = {'original': original_func, 'verbose': verbose_func}

            # --- Schema Generation (Simplified) ---
            try:
                sig = inspect.signature(original_func)
                docstring = inspect.getdoc(original_func) or f"Executes the {tool_name} tool."
                description = docstring.split('\n\n')[0] # Use first paragraph as description

                parameters_schema = {}
                required_params = []
                for name, param in sig.parameters.items():
                    # Basic type mapping (needs improvement for complex types/hints)
                    param_type = "string" # Default
                    if param.annotation in (int, float):
                        param_type = "number"
                    elif param.annotation is bool:
                        param_type = "boolean"
                    elif param.annotation in (list, dict): # Basic complex types
                         param_type = "array" if param.annotation is list else "object"
                         # TODO: Add items/properties schema if possible from type hints

                    # Extract description from docstring (simple parsing)
                    param_desc = f"Parameter {name}"
                    if docstring:
                        for line in docstring.split('\n'):
                            if line.strip().startswith(f":param {name}:") or line.strip().startswith(f"Args:\n    {name} "): # Common patterns
                                param_desc = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                                break


                    parameters_schema[name] = {
                        "type": param_type.upper(), # Schema types are usually uppercase
                        "description": param_desc,
                    }
                    if param.default is inspect.Parameter.empty:
                        required_params.append(name)

                # Create the genai.Tool object
                tool_declaration = genai.protos.FunctionDeclaration(
                    name=tool_name,
                    description=description,
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties=parameters_schema,
                        required=required_params if required_params else None
                    )
                )
                tool_schemas.append(genai.Tool(function_declarations=[tool_declaration]))

            except Exception as schema_e:
                logger.error(f"❌ Failed to generate schema for tool '{tool_name}': {schema_e}")

        self.tool_schemas = tool_schemas
        logger.info(f"✅ Registered {len(self.tool_metadata)} tools: {list(self.tool_metadata.keys())}")
        logger.info(f"   Generated {len(self.tool_schemas)} tool schemas.")
        # Note: We don't return anything here, registration updates self.tool_schemas

    async def _send_message(self, user_input: str):
        """Sends a message to the chat session and handles responses/function calls."""
        if not self.chat_session:
            logger.error("❌ Chat session not initialized.")
            return "Error: Chat session not initialized."

        try:
            # --- Append user message to local history for display ---
            self.conversation_history.append({"role": "user", "parts": [{"text": user_input}]}) # Use correct Parts structure

            # --- Send Message to Gemini ---
            # Automatic function calling uses schemas provided during model/chat init.
            logger.debug(f"--> Sending to Gemini: {user_input[:100]}...")
            response = await self.chat_session.send_message_async(user_input)
            logger.debug(f"<-- Received response from Gemini.")

            # --- Handle Function Calls (New SDK Style) ---
            # The response might contain function calls requested by the model.
            # Check response.function_calls (it's a list)
            while response.function_calls:
                 logger.info(f"⚙️ Model requested {len(response.function_calls)} function call(s).")
                 api_responses = [] # Collect responses for all calls in this turn

                 for function_call in response.function_calls:
                     # Append model's function call request to local history
                     # Need to convert proto FunctionCall to a dict/displayable format
                     fc_dict = {'name': function_call.name, 'args': dict(function_call.args)}
                     self.conversation_history.append({"role": "model", "parts": [{"function_call": fc_dict}]})

                     # Execute the function
                     function_response_content = await self._handle_function_call(function_call) # Returns genai.FunctionResponse

                     # Append our function response to local history
                     # Need to convert proto FunctionResponse to a dict/displayable format
                     fr_dict = {'name': function_response_content.name, 'response': dict(function_response_content.response)}
                     self.conversation_history.append({"role": "function", "parts": [{"function_response": fr_dict}]})

                     api_responses.append(function_response_content) # Add to list for API

                 # --- Send Function Responses back to Gemini ---
                 logger.debug(f"--> Sending {len(api_responses)} function response(s) back to Gemini...")
                 response = await self.chat_session.send_message_async(api_responses) # Send list of FunctionResponse objects
                 logger.debug(f"<-- Received response after function call(s).")
                 # Loop continues if the new response *also* contains function calls

            # --- Process Final Agent Response ---
            response_text = ""
            try:
                 # Access text directly from response.text
                 response_text = response.text
                 # Append final agent response to local history
                 self.conversation_history.append({"role": "model", "parts": [{"text": response_text}]})
            except ValueError:
                 # Handle cases where the response might be blocked or lack text
                 logger.warning("⚠️ Agent response might be blocked or empty.")
                 # Log finish reason if available
                 try:
                    finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
                    logger.warning(f"   Finish Reason: {finish_reason}")
                    if response.candidates and response.candidates[0].safety_ratings:
                        logger.warning(f"   Safety Ratings: {response.candidates[0].safety_ratings}")
                 except Exception:
                    pass # Avoid errors during error reporting
                 response_text = "(Agent response blocked or empty)"
            except Exception as e:
                 logger.error(f"❌ Error extracting final text from response: {e}")
                 response_text = f"(Error extracting response: {e})"

            return response_text

        except Exception as e:
            logger.error(f"❌ Error during message sending/processing: {e}\n{traceback.format_exc()}")
            return f"Error processing message: {e}"

    async def _handle_function_call(self, function_call: genai.FunctionCall) -> genai.FunctionResponse:
        """Executes a requested function call and returns the result."""
        function_name = function_call.name
        args = dict(function_call.args) # Convert proto MapComposite to dict

        logger.info(f"  -> Executing: {function_name} with args: {args}")

        if function_name not in self.tool_metadata:
             result_str = f"Error: Tool '{function_name}' not found or not registered correctly."
             logger.error(result_str)
        else:
            tool_info = self.tool_metadata[function_name]
            # Use the 'verbose' wrapper if enabled, otherwise 'original'
            tool_to_call = tool_info['verbose'] if self.verbose else tool_info['original']
            original_func = tool_info['original'] # For async check

            try:
                # Check if the *original* function is async
                if asyncio.iscoroutinefunction(original_func):
                     logger.debug(f"   -> Awaiting async tool: {function_name}")
                     # Await the execution (verbose wrapper MUST handle await internally if used)
                     # Let's assume verbose wrapper handles it, or we directly await original
                     if self.verbose and hasattr(tool_to_call, '_is_verbose_wrapper'):
                         # Assume wrapper correctly awaits original if needed
                         result = await tool_to_call(**args)
                     else:
                         # Call original async func directly
                         result = await original_func(**args)
                else:
                    # Synchronous tool call
                    logger.debug(f"   -> Calling sync tool: {function_name}")
                    result = tool_to_call(**args)

                # Ensure result is JSON serializable for the API response content
                # Convert complex objects to strings or simplified dicts if necessary
                if not isinstance(result, (str, int, float, bool, list, dict)):
                    logger.warning(f"   -> Tool result type {type(result)} is not directly JSON serializable. Converting to string.")
                    result_str = str(result)
                else:
                     result_str = result # Keep simple types as is

                logger.info(f"  <- Tool {function_name} result type: {type(result)}")

            except Exception as e:
                error_msg = f"Error executing tool '{function_name}': {e}"
                logger.error(f"  <- {error_msg}\n{traceback.format_exc()}")
                result_str = error_msg # Return error message as content

        # Prepare response content for the model using genai.FunctionResponse
        # The 'response' field should contain a dict representing the JSON output
        function_response = genai.FunctionResponse(
            name=function_name,
            response={"content": result_str} # API expects a dict here
        )
        return function_response

    # --- Browser Tool Method ---
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
        headless = params.get("headless", True)
        highlight = params.get("highlight", False)
        record = params.get("record", False)
        wait_time = params.get("wait_time", 5.0)

        logger.info(f"\n🧭 Tool: browse_web params: url={url}, task='{task}', headless={headless}, highlight={highlight}, record={record}, wait_time={wait_time}")

        if not self.browser_context:
            return "Error: Browser context is not initialized. Cannot browse."
        if not self.llm:
            # Use self.model (genai) as fallback? Needs browser_use compatibility check.
            logger.warning("⚠️ Langchain LLM for browser agent not available.")
            # For now, require the Langchain LLM setup via API key
            return "Error: LLM for browser agent is not initialized (requires API Key). Cannot browse."

        try:
            context_config_overrides = BrowserContextConfig(
                wait_for_network_idle_page_load_time=wait_time,
                highlight_elements=highlight,
                save_recording_path="./recordings" if record else None
            )

            initial_actions = []
            if url:
                initial_actions.append({"open_tab": {"url": url}})
            elif not self.browser_context.pages:
                default_url = "[https://www.google.com](https://www.google.com)"
                logger.warning(f"⚠️ No URL provided and no active pages. Opening default: {default_url}")
                initial_actions.append({"open_tab": {"url": default_url}})

            agent = Agent(
                task=task,
                llm=self.llm, # browser_use Agent uses Langchain LLM
                browser=self.browser,
                browser_context=self.browser_context,
                use_vision=True,
                generate_gif=record,
                initial_actions=initial_actions if initial_actions else None
            )

            result = await agent.run()
            final_result = result.final_result() if result else "No result from browser agent."
            logger.info(f"  -> Browser agent final result: {final_result[:200]}...")
            return final_result

        except Exception as e:
            logger.error(f"\n⚠️ browse_web exception: {e}\n{traceback.format_exc()}")
            return f"Error during browser operation: {e}"

    # --- Verbose Tool Wrapper ---
    def _make_verbose_tool(self, func):
        """Wrap tool function to print verbose info when called."""
        original_func = getattr(func, '__wrapped__', func)

        # Make the wrapper async if the original function is async
        if asyncio.iscoroutinefunction(original_func):
            @functools.wraps(original_func)
            async def _verbose_wrapper(*args, **kwargs):
                logger.info(f"🛠️ Tool: {original_func.__name__} args: {args}, kwargs: {kwargs}")
                try:
                    result = await original_func(*args, **kwargs) # Await the original async func
                    result_display = str(result)
                    if len(result_display) > 200:
                         result_display = result_display[:200] + "... (truncated)"
                    logger.info(f"  ✅ Result ({original_func.__name__}): {result_display}")
                    return result
                except Exception as e:
                     logger.error(f"  ❌ Error in tool {original_func.__name__}: {e}")
                     raise
        else:
            @functools.wraps(original_func)
            def _verbose_wrapper(*args, **kwargs):
                logger.info(f"🛠️ Tool: {original_func.__name__} args: {args}, kwargs: {kwargs}")
                try:
                    result = original_func(*args, **kwargs) # Call the original sync func
                    result_display = str(result)
                    if len(result_display) > 200:
                         result_display = result_display[:200] + "... (truncated)"
                    logger.info(f"  ✅ Result ({original_func.__name__}): {result_display}")
                    return result
                except Exception as e:
                     logger.error(f"  ❌ Error in tool {original_func.__name__}: {e}")
                     raise

        _verbose_wrapper.__wrapped__ = original_func
        _verbose_wrapper._is_verbose_wrapper = True
        return _verbose_wrapper

# --- Main Execution ---
async def main():
    """Main asynchronous function to run the Code Agent."""
    parser = argparse.ArgumentParser(description="Run the Code Agent")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose tool logging')
    # Add model selection argument
    parser.add_argument(
        '-m', '--model', type=str, default=None,
        help='Specify the Gemini model name (e.g., gemini-1.5-flash-latest). Overrides config file.'
    )
    args = parser.parse_args()

    config = Config()
    # Override model from command line if provided
    if args.model:
         config.set_model_name(args.model) # Assumes Config has a setter method

    agent = CodeAgent(config)
    agent.verbose = args.verbose # Override verbosity from args

    try:
        # --- Initialization ---
        agent._configure_client() # Sync setup (includes LLM)
        await agent._initialize_browser() # Async setup (conditionally adds browse_web & registers tools)
        await agent._initialize_chat() # Async setup (uses model and registered tools)

        # Tools are now registered within _initialize_browser or _initialize_chat

        logger.info(f"✅ Agent ready (Model: {agent.model_name}). Ask me anything. Type 'exit' to quit.")
        # logger.info("   Use '/upload <path/to/file.pdf>' to seed PDF into context.") # Add back if needed
        logger.info("   Use '/reset' to clear the chat and start fresh.")

        # --- Interaction Loop ---
        while True:
            # Removed thinking budget input for simplicity now
            user_input = await asyncio.to_thread(input, f"\n🔵 You ({len(agent.conversation_history)} turns): ")
            if not user_input:
                 continue
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "/reset":
                logger.info("🔄 Resetting chat session...")
                # Re-initialize chat (clears history)
                await agent._initialize_chat()
                # Consider if browser needs reset too? Let's keep it persistent for now.
                # If browser needs reset: await agent.close_browser(); await agent._initialize_browser()
                logger.info("✅ Chat session reset.")
                continue
            # TODO: Add /upload functionality back if needed

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