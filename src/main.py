from google import genai
from google.genai import types
import os
import sys
from pathlib import Path
from . import database
from . import tools
from .autocomplete import PdfCompleter  # <-- Import PdfCompleter
import traceback
import argparse
import functools
import logging
import asyncio, threading, uuid # Added uuid
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn # Added TextColumn, BarColumn
from prompt_toolkit.document import Document # For completer
from prompt_toolkit.completion import Completer, Completion, NestedCompleter, WordCompleter, PathCompleter, FuzzyWordCompleter, CompleteEvent # For completer, Added CompleteEvent
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition # Added
from prompt_toolkit.keys import Keys # Added
from prompt_toolkit.application.current import get_app # Added
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import CompleteStyle # Added for complete_style argument
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from typing import Iterable, Dict, Any, Optional, Set, Union # For completer and CustomNestedCompleter
import bisect # Added for efficient searching
import yaml
import sqlite3
from typing import Optional, List, Dict, Callable 
from dotenv import load_dotenv # Import dotenv
from datetime import datetime, timezone # Added for processed_timestamp
import sys # For sys.executable
import subprocess # For running scripts
# from google.generativeai import types as genai_types # This was incorrect for the new SDK as the `google.generativeai` SDK is deprecated and we should use `google.genai` SDK instead

# --- Import slash command handlers ---
from . import slashcommands # Import the new module


# Setup basic logging
# Configure logging with a default of WARNING (less verbose)
logging.basicConfig(level=logging.WARNING) # Default to less verbose logging
logger = logging.getLogger(__name__) # Define module-level logger


# --- Completer Logging Wrapper ---
class LoggingCompleterWrapper(Completer):
    def __init__(self, wrapped_completer: Completer, name: str = "Wrapped"):
        self.wrapped_completer = wrapped_completer
        self.name = name

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        logger.info(f"[{self.name}] get_completions CALLED. Text: '{document.text_before_cursor}', UserInvoked: {complete_event.completion_requested}")
        raw_completions = list(self.wrapped_completer.get_completions(document, complete_event))
        logger.info(f"[{self.name}] Raw completions from wrapped: {[c.text for c in raw_completions]} for input '{document.text_before_cursor}'")
        yield from raw_completions
# --- End Completer Logging Wrapper ---

class CustomNestedCompleter(NestedCompleter):
    def __init__(self, options: Dict[str, Optional[Completer]], ignore_case: bool = True):
        super().__init__(options, ignore_case=ignore_case)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text_before_cursor = document.text_before_cursor.lstrip()

        # If there are no spaces in the text before the cursor (after stripping leading space),
        # it means we are typing the first command word.
        if ' ' not in text_before_cursor:
            # Get actual top-level command strings from the options dictionary keys
            actual_keys = [k for k in self.options.keys() if isinstance(k, str)]
            
            first_word_completer = WordCompleter(
                words=actual_keys,
                ignore_case=self.ignore_case,
                match_middle=True,  # Key for filtering as we type
                sentence=False      # Match the whole command like "/help" as one word
            )
            # logger.debug(f"CustomNestedCompleter: Using FirstWordCompleter for '{document.text_before_cursor}'") # Optional debug
            yield from first_word_completer.get_completions(document, complete_event)
        else:
            # If there are spaces, it means we are beyond the first command word.
            # Delegate to the original NestedCompleter logic to handle sub-commands.
            # logger.debug(f"CustomNestedCompleter: Delegating to super for '{document.text_before_cursor}'") # Optional debug
            yield from super().get_completions(document, complete_event)

# Helper function for the key binding condition
def is_typing_slash_command_prefix(current_buffer):
    text = current_buffer.text
    # Only trigger if text starts with / and has no spaces, and is not just "/"
    return text.startswith('/') and ' ' not in text and len(text) > 0

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)


MODEL_NAME = config.get('model_name', "gemini-2.5-flash-preview-04-17")
DEFAULT_THINKING_BUDGET = config.get('default_thinking_budget', 256)
MAX_PDF_CONTEXT_LENGTH = config.get('MAX_PDF_CONTEXT_LENGTH', None) # Max chars from PDF to prepend 

# --- Utility Functions ---
def load_config(config_path: Path):
    """Loads configuration, prioritizing environment variables for API key."""
    # 1. Load environment variables from ~/.env (if it exists)
    # find_dotenv will search directories upwards from the script location
    # for a .env file. We specify the home directory explicitly.
    dotenv_home_path = Path.home() / '.env'
    if dotenv_home_path.is_file():
        load_dotenv(dotenv_path=dotenv_home_path)
        logger.info(f"Loaded environment variables from {dotenv_home_path}")
    else:
        logger.info(f"No .env file found at {dotenv_home_path}, checking system environment variables.")

    # 2. Check for API key in environment variables first
    env_api_key = os.getenv('GEMINI_API_KEY')

    # 4. Prioritize environment variable for API key
    if env_api_key:
        config['gemini_api_key'] = env_api_key
        logger.info("Using GEMINI_API_KEY from environment.")
    elif 'gemini_api_key' in yaml_data and yaml_data['gemini_api_key']:
        logger.info("Using gemini_api_key from config.yaml (environment variable not set).")
        # It's already in config from the update() step
    else:
        logger.warning("GEMINI_API_KEY not found in environment or config.yaml.")


    logger.info(f"Using model: {MODEL_NAME}")

    # 5. Resolve paths (relative to project root, which is parent of src/)
    project_root = Path(__file__).parent.parent 
    for key in ['PDFS_TO_CHAT_WITH_DIRECTORY', 
                'SAVED_CONVERSATIONS_DIRECTORY', 
                'PAPER_DB_PATH', 
                'PAPER_BLOBS_DIR']:
        if key in config and isinstance(config[key], str):
            resolved_path = project_root / config[key]
            if key == 'PAPER_DB_PATH':
                 # Ensure parent dir exists for DB path
                 resolved_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                 # Ensure the directory itself exists for others
                 resolved_path.mkdir(parents=True, exist_ok=True)
            config[key] = resolved_path
            logger.info(f"Resolved path for {key}: {config[key]}")
        elif key in config and isinstance(config[key], Path):
             # Path might already be resolved if loaded from previous runs/complex config
             logger.info(f"Path for {key} already resolved: {config[key]}")
             # Ensure directories exist even if path was pre-resolved
             if key == 'PAPER_DB_PATH':
                  config[key].parent.mkdir(parents=True, exist_ok=True)
             else:
                  config[key].mkdir(parents=True, exist_ok=True)

    return config

def print_welcome_message(config):
    """Prints the initial welcome and help message."""
    print("\n🚀 Starting Code Agent...")
    # Add other initial prints if needed

# --- Agent Class ---
class CodeAgent:
    def __init__(self, config: dict, conn: Optional[sqlite3.Connection]):
        """Initializes the CodeAgent."""
        self.config = config
        self.api_key = config.get('gemini_api_key') # Correct key name
        self.model_name = MODEL_NAME
        self.pdf_processing_method = config.get('pdf_processing_method', 'Gemini') # Default method
        self.client = None
        self.chat = None
        self.db_path_str = str(config.get('PAPER_DB_PATH')) if config.get('PAPER_DB_PATH') else None # Store DB path string
        self.prefill_prompt_content: Optional[str] = None # For pre-filling the next prompt
        
        # Background asyncio loop (daemon so app exits cleanly)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        # Async GenAI client that lives on that loop
        if self.api_key: # Ensure API key exists before creating async client
            self.async_client = genai.Client(api_key=self.api_key).aio
        else:
            self.async_client = None
            logger.warning("GEMINI_API_KEY not found. Async GenAI client not initialized.")

        # Collection to keep track of active background tasks (Future objects and metadata)
        self.active_background_tasks = {}
        self.pending_script_output: Optional[str] = None # For output from /run_script
        
        self.conversation_history = [] # Manual history for token counting ONLY
        self.current_token_count = 0 # Store token count for the next prompt
        self.active_files = [] # List to store active File objects
        self.prompt_time_counts = [0] # Stores total tokens just before prompt
        self.messages_per_interval = [0] # Stores # messages added in the last interval
        self._messages_this_interval = 0 # Temporary counter
        self.thinking_budget = config.get('default_thinking_budget', DEFAULT_THINKING_BUDGET)
        self.thinking_config = None # Will be set in start_interaction
        self.pdfs_dir_rel_path = config.get('PDFS_TO_CHAT_WITH_DIRECTORY') # Relative path from config
        self.pdfs_dir_abs_path = Path(self.pdfs_dir_rel_path).resolve() if self.pdfs_dir_rel_path else None
        self.blob_dir_rel_path = config.get('PAPER_BLOBS_DIR')
        self.blob_dir = Path(self.blob_dir_rel_path).resolve() if self.blob_dir_rel_path else None
        # Store the database connection passed from main
        self.conn = conn 
        # Use imported tool functions
        self.tool_functions = [
            tools.read_file,
            tools.list_files,
            tools.edit_file,
            tools.execute_bash_command,
            tools.run_in_sandbox,
            tools.find_arxiv_papers,
            tools.download_arxiv_paper,
            tools.get_current_date_and_time,
            tools.google_search,
            tools.open_url,
            tools.upload_pdf_for_gemini,
            tools.run_sql_query
        ]
        if self.config.get('verbose', config.get('verbose', False)):
            self.tool_functions = [self._make_verbose_tool(f) for f in self.tool_functions]
        if self.pdfs_dir_abs_path:
            self.pdfs_dir_abs_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"PDF directory set to: {self.pdfs_dir_abs_path}")
        else:
            logger.warning("PDF directory not configured in config.yaml. /pdf command will be disabled.")

        if self.blob_dir:
            self.blob_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Blob directory set to: {self.blob_dir}")
        else:
            logger.warning("Blob directory not configured in config.yaml. Saving extracted text will be disabled.")

        if not self.conn:
             logger.warning("Database connection not established. Database operations will be disabled.")

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in config or environment. Gemini features disabled.")
        else:
            self._configure_client()
            if self.client:
                self._initialize_chat()

        self.pending_pdf_context: Optional[str] = None # For prepending PDF context
        self.prompts_dir = Path('src/prompts').resolve()

        # Ensure directories exist (moved from load_config for clarity)
        if self.pdfs_dir_abs_path:
            self.pdfs_dir_abs_path.mkdir(parents=True, exist_ok=True)
        if self.blob_dir:
            self.blob_dir.mkdir(parents=True, exist_ok=True)
        if self.prompts_dir:
            self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def _configure_client(self):
        """Configures the Google Generative AI client."""
        print("\n\u2692\ufe0f Configuring genai client...")
        try:
            # Configure the client with our API key
            self.client = genai.Client(api_key=self.api_key)
            print("\u2705 Client configured successfully.")
        except Exception as e:
            print(f"\u274c Error configuring genai client: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _initialize_chat(self):
        """Initializes the chat session."""
        print("\n\u2692\ufe0f Initializing chat session...")
        try:
            # Create a chat session using the client
            self.chat = self.client.chats.create(model=self.model_name, history=[])
            print("\u2705 Chat session initialized.")
        except Exception as e:
            print(f"\u274c Error initializing chat session: {e}")
            traceback.print_exc()
            sys.exit(1)

    # --- Prompt Helper Methods ---
    def _list_available_prompts(self) -> list[str]:
        """Lists available prompt names from the prompts directory."""
        if not self.prompts_dir.is_dir():
            return []
        prompt_files = [f.stem for f in self.prompts_dir.glob('*.txt')] + \
                       [f.stem for f in self.prompts_dir.glob('*.md')]
        return sorted(list(set(prompt_files))) # Sort and remove duplicates

    def _load_prompt(self, prompt_name: str) -> Optional[str]:
        """Loads the content of a prompt file."""
        if not self.prompts_dir.is_dir():
            return None
        
        # Check for .txt first, then .md
        txt_path = self.prompts_dir / f"{prompt_name}.txt"
        md_path = self.prompts_dir / f"{prompt_name}.md"

        path_to_load = None
        if txt_path.is_file():
            path_to_load = txt_path
        elif md_path.is_file():
            path_to_load = md_path
        
        if path_to_load:
            try:
                return path_to_load.read_text()
            except Exception as e:
                logger.error(f"Error reading prompt file {path_to_load}: {e}", exc_info=True)
                return None
        else:
             return None
    # ---------------------------

    def _get_dynamic_prompt_message(self):
        """Returns the dynamic prompt message with status indicators."""
        active_files_info = f" [{len(self.active_files)} files]" if self.active_files else ""
        token_info = f"({self.current_token_count})"
        
        return HTML(
            f'<ansiblue>🔵 You</ansiblue> '
            f'<ansicyan>{token_info}</ansicyan>'
            f'<ansigreen>{active_files_info}</ansigreen>: '
        )

    def _get_continuation_prompt(self, width, line_number, is_soft_wrap):
        """Returns the continuation prompt for multi-line input."""
        if is_soft_wrap:
            return ' ' * 2  # Indent for soft wraps
        return HTML('<ansiyellow>│ </ansiyellow>')  # Visual line continuation

    def _get_bottom_toolbar(self):
        """Returns the bottom toolbar with helpful key bindings."""
        return HTML(
            '<ansigray>'
            '[Alt+Enter] Submit │ [Enter] New line │ [Ctrl+D] Quit │ '
            '[↑↓] History │ [Tab] Complete'
            '</ansigray>'
        )

    def _create_enhanced_prompt_session(self, command_completer, history):
        """Creates an enhanced PromptSession with multi-line editing capabilities."""
        
        # Custom key bindings for multi-line editing
        kb = KeyBindings()
        
        @kb.add('enter')
        def _(event):
            """Enter creates a new line in multi-line mode."""
            event.current_buffer.insert_text('\n')
        
        @kb.add('escape', 'enter')  # Alt+Enter or Esc then Enter
        def _(event):
            """Alt+Enter submits the multi-line input."""
            event.current_buffer.validate_and_handle()
        
        @kb.add('c-j')  # Ctrl+J as alternative submit
        def _(event):
            """Ctrl+J submits the multi-line input (alternative)."""
            event.current_buffer.validate_and_handle()
        
        @kb.add('c-d')  # Ctrl+D to quit on empty line
        def _(event):
            """Ctrl+D on empty line to quit."""
            if not event.current_buffer.text.strip():
                event.app.exit(result='exit_eof') # Changed result to distinguish from KeyboardInterrupt

        @kb.add(Keys.Any, filter=Condition(lambda: is_typing_slash_command_prefix(get_app().current_buffer)))
        def _handle_slash_command_typing(event):
            """
            Handles typing characters for slash commands, inserts the char,
            and forces completion refresh.
            """
            event.current_buffer.insert_text(event.data) # Insert the character typed
            event.app.current_buffer.start_completion(select_first=False)

        # @kb.add('tab')
        # def _(event):
        #     """Handle tab completion."""
        #     b = event.current_buffer
        #     if b.complete_state:
        #         b.complete_next()
        #     else:
        #         b.start_completion(select_first=False) # Use select_first=False for a better UX usually
        
        # Create the enhanced session
        return PromptSession(
            message=self._get_dynamic_prompt_message,  # Dynamic prompt function
            multiline=True, # Re-enable multi-line
            wrap_lines=True, # Re-enable wrap lines
            mouse_support=True,  # Enable mouse support
            complete_style=CompleteStyle.MULTI_COLUMN, # Optional: can be nice with more completions
            completer=command_completer, # Now using CustomNestedCompleter
            history=history,
            key_bindings=kb, # Enable our custom key bindings
            auto_suggest=AutoSuggestFromHistory(), # Can re-enable if desired
            # enable_history_search=True, # Keep False with complete_while_typing=True
            search_ignore_case=True, # Good for history search if enabled
            prompt_continuation=self._get_continuation_prompt,  # Custom continuation
            bottom_toolbar=self._get_bottom_toolbar,  # Status bar
            complete_while_typing=True,  # THE KEY SETTING TO TEST
            enable_history_search=False, # Must be False if complete_while_typing is True
            # input_processors=None, 
        )

    def print_initial_help(self):
        """Prints the initial brief help message."""
        print("\n\u2692\ufe0f Agent ready. Ask me anything or type '/help' for commands.")
        print("   Type '/exit' or '/q' to quit.")
        # Key commands can be highlighted if desired, but /help is the main source.

    def start_interaction(self):
        """Starts the main interaction loop with enhanced multi-line editing."""
        if not self.client:
            print("\n\u274c Client not configured. Exiting.")
            return

        self.print_initial_help() # Use the new brief help message

        # Set initial thinking budget from default/config
        self.thinking_config = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        print(f"\n🧠 Initial thinking budget set to: {self.thinking_budget} tokens.")

        # Print multi-line editing instructions
        print("\n📝 Multi-line editing enabled:")
        print("   • Press [Enter] to create new lines")
        print("   • Press [Alt+Enter] or [Ctrl+J] to submit")
        print("   • Use mouse to select text and position cursor")
        print("   • Press [↑↓] to navigate history")

        # Define slash commands and setup nested completer
        # The keys in slash_command_completer_map should match COMMAND_HANDLERS keys
        slash_command_completer_map: Dict[str, Optional[Completer]] = {
            '/reset': None,
            '/exit': None,
            '/q': None,
            '/clear': None, # Takes an arg, but simple WordCompleter not ideal
            '/save': None,  # Takes an optional arg
            '/thinking_budget': None, # Takes an arg
            '/cancel': None, # Takes an arg (task_id)
            '/tasks': None,
            '/run_script': None, # Complex args, PathCompleter for script_path might be good
            '/help': None,
            '/history': None, # New history command
             # /pdf and /load have specific completers
        }

        # PDF file completer
        # PdfCompleter now takes 'self' (the CodeAgent instance)
        slash_command_completer_map['/pdf'] = PdfCompleter(self)

        # Saved conversations completer
        saved_conversations_dir_path = self.config.get('SAVED_CONVERSATIONS_DIRECTORY')
        saved_files = []
        if saved_conversations_dir_path and isinstance(saved_conversations_dir_path, Path) and saved_conversations_dir_path.is_dir():
             saved_files = [f.name for f in saved_conversations_dir_path.glob('*.json') if f.is_file()]
        elif isinstance(saved_conversations_dir_path, str): 
             logger.warning("SAVED_CONVERSATIONS_DIRECTORY was a string, expected Path. Attempting to resolve.")
             try:
                  # Assuming project_root is defined if this fallback is hit, or use Path(__file__).parent.parent
                  project_root = Path(__file__).parent.parent 
                  resolved_path = project_root / saved_conversations_dir_path
                  if resolved_path.is_dir():
                       saved_files = [f.name for f in resolved_path.glob('*.json') if f.is_file()]
             except Exception as e:
                  logger.error(f"Error resolving string path for saved conversations: {e}")
        else: 
             default_save_dir = Path(__file__).parent.parent / 'SAVED_CONVERSATIONS/'
             if default_save_dir.is_dir():
                  saved_files = [f.name for f in default_save_dir.glob('*.json') if f.is_file()]
        slash_command_completer_map['/load'] = WordCompleter(saved_files, ignore_case=True)

        # Prompt name completer
        # slash_command_completer_map['/prompt'] = WordCompleter(self._list_available_prompts(), ignore_case=True) # Temporarily disable for flat test
        
        # Add all command names from COMMAND_HANDLERS to the completer if not already specified
        for cmd_name in slashcommands.COMMAND_HANDLERS.keys():
            if cmd_name not in slash_command_completer_map:
                slash_command_completer_map[cmd_name] = None

        # Create the CustomNestedCompleter instance
        command_completer_instance = CustomNestedCompleter(
            options=slash_command_completer_map,
            ignore_case=True
        )

        logger.debug(f"Using CustomNestedCompleter with options: {list(slash_command_completer_map.keys())}")

        history = InMemoryHistory()
        # Create enhanced session
        session = self._create_enhanced_prompt_session(command_completer_instance, history)

        while True:
            try:
                # ─── 1 · house‑keeping before we prompt ──────────────────────────
                self.prompt_time_counts.append(self.current_token_count)
                self.messages_per_interval.append(self._messages_this_interval)
                self._messages_this_interval = 0

                # Get multi-line input with the enhanced session
                # The prompt message is now handled by _get_dynamic_prompt_message
                # Prefill logic needs to be integrated with PromptSession's `default` if needed, or handled before prompt.
                # For now, we simplify and remove direct prefill_text here as the plan's session.prompt() is simpler.
                # If prefill is critical, it should be passed to session.prompt(default=prefill_text)
                
                # Check if there's content to prefill from /prompt command
                current_prefill_text = ""
                if self.prefill_prompt_content:
                    current_prefill_text = self.prefill_prompt_content
                    self.prefill_prompt_content = None # Clear after retrieving
                
                try:
                    user_input = session.prompt(default=current_prefill_text).strip()
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:  # Ctrl+D (or result from custom keybinding)
                    print("\n👋 Goodbye!") # Or handle 'exit_eof' if needed
                    break

                # ─── 2 · trivial exits / empty line ─────────────────────────────
                # Handle exit commands (already covered by Ctrl+D and KeyboardInterrupt for session)
                # The custom Ctrl+D binding in _create_enhanced_prompt_session handles empty line exit.
                # If user types 'exit' or 'quit' it will be processed as a command or message.
                # The plan's version of this check is slightly different, let's ensure it's robust.
                if user_input.lower() in {"exit", "quit"}: # This is from the plan, simpler than existing /exit /q
                    print("\n👋 Goodbye!")
                    break
                
                # If Ctrl+D was used on an empty line, app.exit(result='exit_eof') is called.
                # The prompt loop will break due to EOFError in that case.
                # No specific check for 'exit_eof' needed here if EOFError is caught.

                if not user_input:
                    self.prompt_time_counts.pop()
                    self.messages_per_interval.pop()
                    continue

                # --- 3 · Command Parsing and Dispatch ---
                command_parts = user_input.split()
                command_name = command_parts[0].lower()
                command_args = command_parts[1:]

                handler: Optional[Callable] = slashcommands.COMMAND_HANDLERS.get(command_name)

                if handler:
                    # Determine if the handler takes arguments.
                    # This is a simple check; more robust would be `inspect.signature`.
                    # For now, we assume handlers like /reset, /tasks, /help don't need args.
                    # Updated logic to pass session to /prompt handler
                    if command_name in ["/reset", "/tasks", "/help"]:
                        handler(self) # Call with agent only
                    elif command_name == "/prompt":
                        handler(self, session, command_args) # Call /prompt with agent, session, args
                    elif command_name == "/save" and not command_args:
                        handler(self, []) # Call /save with agent and empty args list
                    else:
                        # Default for commands taking agent and args
                        handler(self, command_args)

                    continue # Command handled, loop to next prompt
                
                # --- If not a known slash command, proceed as LLM message ---
                self._messages_this_interval += 1 
                message_to_send = user_input 

                pdf_context_was_included = False
                script_output_was_included = False # New flag
                 
                # Check for PDF context AFTER prompt (so prompt comes first)
                if self.pending_pdf_context:
                    print("[Including context from previously processed PDF in this message.]\n")
                    message_to_send = f"{self.pending_pdf_context}\n\n{message_to_send}"
                    pdf_context_was_included = True
                
                # Prepend script output AFTER PDF but BEFORE loaded prompt
                if self.pending_script_output:
                    print("[Including output from previously run script in this message.]\n")
                    # Task name might not be easily available here, use a generic header
                    message_to_send = f"OUTPUT FROM EXECUTED SCRIPT:\n---\n{self.pending_script_output}\n---\n\n{message_to_send}"
                    script_output_was_included = True
                
                # --- Log message details before sending --- 
                pdf_context_len_before_send = len(self.pending_pdf_context) if self.pending_pdf_context and pdf_context_was_included else 0
                script_output_len_before_send = len(self.pending_script_output) if self.pending_script_output and script_output_was_included else 0
                final_message_len = len(message_to_send)
                logger.info(f"Preparing to send message.")
                logger.info(f"  - Original user input length: {len(user_input)}")
                logger.info(f"  - Included pending PDF context length: {pdf_context_len_before_send}")
                logger.info(f"  - Included pending script output length: {script_output_len_before_send}")
                logger.info(f"  - Final message_to_send length: {final_message_len}")
                # Log snippets for verification
                if final_message_len > 200:
                    logger.info(f"  - Final message start: {message_to_send[:100]}...")
                    logger.info(f"  - Final message end: ...{message_to_send[-100:]}")

                # --- Prepare message content (Text + Files) ---
                message_content = [message_to_send]
                if self.active_files:
                    message_content.extend(self.active_files)
                    if self.config.get('verbose', False):
                        print(f"\n📎 Attaching {len(self.active_files)} files to the prompt:")
                        for f in self.active_files:
                            print(f"   - {f.display_name} ({f.name})")

                # --- Update manual history (for token counting ONLY - Use Text Only) --- 
                new_user_content =types.Content(parts=[types.Part(text=message_to_send)], role="user")
                self.conversation_history.append(new_user_content)

                # --- Send Message --- 
                print("\n⏳ Sending message and processing...")
                # Prepare tool configuration **inside the loop** to use the latest budget
                tool_config = types.GenerateContentConfig(
                    tools=self.tool_functions, 
                    thinking_config=self.thinking_config
                )

                # Send message using the chat object's send_message method
                # Pass the potentially combined list of text and files
                response = self.chat.send_message(
                    message=message_content, # Pass the list here
                    config=tool_config
                )

                agent_response_text = ""
                if response.candidates and response.candidates[0].content:
                    agent_parts = response.candidates[0].content.parts
                    agent_response_text = " ".join(p.text for p in agent_parts
                                                   if hasattr(p, "text"))

                if agent_response_text:
                    hist_agent_content = types.Content(role="model",
                                                     parts=[types.Part(text=agent_response_text)])
                    self.conversation_history.append(hist_agent_content)

                print(f"\n🟢 \x1b[92mAgent:\x1b[0m {agent_response_text or '[No response text]'}")

                # --- Detailed History Logging Before Token Count --- 
                logger.debug(f"Inspecting conversation_history (length: {len(self.conversation_history)}) before count_tokens:")
                history_seems_ok = True
                for i, content in enumerate(self.conversation_history):
                    logger.debug(f"  [{i}] Role: {getattr(content, 'role', 'N/A')}")
                    if hasattr(content, 'parts'):
                        for j, part in enumerate(content.parts):
                            part_type = type(part)
                            part_info = f"Part {j}: Type={part_type.__name__}"
                            if hasattr(part, 'text'):
                                part_info += f", Text='{part.text[:50]}...'"
                            elif hasattr(part, 'file_data'):
                                part_info += f", FileData URI='{getattr(part.file_data, 'file_uri', 'N/A')}'"
                                history_seems_ok = False # Found a file part!
                                logger.error(f"    🚨 ERROR: Found unexpected file_data part in history for token counting: {part_info}")
                            elif hasattr(part, 'function_call'):
                                part_info += f", FunctionCall Name='{getattr(part.function_call, 'name', 'N/A')}'"
                                history_seems_ok = False # Found a function call part!
                                logger.error(f"    🚨 ERROR: Found unexpected function_call part in history for token counting: {part_info}")
                            else:
                                # Log other unexpected part types
                                history_seems_ok = False
                                logger.error(f"    🚨 ERROR: Found unexpected part type in history for token counting: {part_info}")
                            logger.debug(f"    {part_info}")
                    else:
                        logger.warning(f"  [{i}] Content object has no 'parts' attribute.")
                if history_seems_ok:
                    logger.debug("History inspection passed: Only text parts found.")
                else:
                    logger.error("History inspection FAILED: Non-text parts found. Token counting will likely fail.")
                # --- End Detailed History Logging --- 

                # Calculate and display token count using client.models
                try:
                    token_info = self.client.models.count_tokens(
                        model=self.model_name,
                        contents=self.conversation_history
                    )
                    self.current_token_count = token_info.total_tokens
                    print(f"\n[Token Count: {self.current_token_count}]")
                except Exception as count_err:
                    logger.error(f"Error calculating token count: {count_err}", exc_info=True)
                    print("🚨 Error: Failed to calculate token count.")

                # --- NOW clear contexts that were actually sent --- 
                if pdf_context_was_included:
                    self.pending_pdf_context = None
                    logger.info("Cleared pending_pdf_context after sending to LLM.")
                if script_output_was_included:
                    self.pending_script_output = None
                    logger.info("Cleared pending_script_output after sending to LLM.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n🔴 An error occurred during interaction: {e}")
                traceback.print_exc()
                # Potentially add a small delay or a prompt to continue/exit here

    def _make_verbose_tool(self, func):
        """Wrap tool function to print verbose info when called."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n🔧 Tool called: {func.__name__}, args: {args}, kwargs: {kwargs}")
            result = func(*args, **kwargs)
            print(f"\n▶️ Tool result ({func.__name__}): {result}")
            return result
        return wrapper

    def _handle_pdf_command(self, args: list):
        """Handles the /pdf command to asynchronously process a PDF file.
        
        Args:
            args: List of command arguments. Expected format:
                 [filename] [--sort <field>] [--reverse] [arxiv_id]
        """
        if not self.pdfs_dir_abs_path:
            print("\n⚠️ PDF directory not configured. Cannot process PDFs.")
            return
        
        # Use the connection established during initialization
        if not self.conn:
            print("\n⚠️ Database connection not available. Cannot save PDF metadata.")
            return

        # Async client check (used by _process_pdf_async_v2)
        if not self.async_client:
            print("\n⚠️ Async Gemini client not initialized. Cannot process PDF asynchronously.")
            return

        if not args:
            print("\n⚠️ Usage: /pdf <filename> [--sort <field>] [--reverse] [arxiv_id]")
            print("       Available sort fields: name, time (default: name)")
            return

        # Parse command line arguments
        sort_by = 'name'  # Default sort field
        reverse_sort = False
        arxiv_id = None
        filename_arg = None
        
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == '--sort':
                if i + 1 < len(args) and not args[i + 1].startswith('-'):
                    sort_by = args[i + 1]
                    i += 1  # Skip the next argument as it's the sort field
                    # Check if there's a sort direction (D for descending)
                    if i + 1 < len(args) and args[i + 1] == 'D':
                        reverse_sort = True
                        i += 1  # Skip the direction argument
            elif arg == '--reverse':
                reverse_sort = True
            elif not filename_arg and not arg.startswith('-'):
                # First non-flag argument is treated as filename
                filename_arg = arg
            elif not arg.startswith('-'):
                # Any subsequent non-flag argument is treated as arxiv_id
                arxiv_id = arg
            i += 1

        if not filename_arg:
            print("\n⚠️ Error: No filename provided.")
            print("Usage: /pdf <filename> [--sort <field>] [--reverse] [arxiv_id]")
            return

        # Basic security: Ensure filename doesn't contain path separators
        filename = Path(filename_arg).name
        if filename != filename_arg:
            print(f"\n⚠️ Invalid filename '{filename_arg}'. Please provide only the filename, not a path.")
            return

        pdf_path = self.pdfs_dir_abs_path / filename

        # --- Check for cached version first ---
        cached_paper_info = database.get_processed_paper_by_filename(self.conn, filename)
        if cached_paper_info and self.blob_dir:
            blob_filename = cached_paper_info.get("blob_path")
            paper_id = cached_paper_info.get("paper_id")
            if blob_filename:
                blob_full_path = self.blob_dir / blob_filename
                if blob_full_path.is_file():
                    try:
                        extracted_text = blob_full_path.read_text()
                        context_header = f"CONTEXT FROM CACHED PDF (\'{filename}\', ID: {paper_id}):\\n---"
                        
                        if MAX_PDF_CONTEXT_LENGTH is not None and isinstance(MAX_PDF_CONTEXT_LENGTH, int) and MAX_PDF_CONTEXT_LENGTH > 0:
                            text_to_truncate = extracted_text
                            # Adjust max_text_len to account for the header and truncation marker
                            max_text_len = MAX_PDF_CONTEXT_LENGTH - len(context_header) - len("\\n---") - 20 # a bit of buffer
                            if max_text_len < 0: max_text_len = 0
                            
                            truncated_text = text_to_truncate[:max_text_len]
                            if len(text_to_truncate) > max_text_len:
                                truncated_text += "\\n... [TRUNCATED]"
                        else:
                            truncated_text = extracted_text
                        
                        self.pending_pdf_context = f"{context_header}\\n{truncated_text}\\n---"
                        print(f"\n📄 Using cached version of \'{filename}\'. Context prepared.")
                        logger.info(f"Loaded cached PDF \'{filename}\' (ID: {paper_id}) from blob: {blob_filename}")
                        
                        # Optionally, if you want to also add the GenAI file to active_files if available
                        genai_uri = cached_paper_info.get("genai_file_uri")
                        if genai_uri and self.client:
                            try:
                                # We need to "reconstruct" a File object or decide if it's needed
                                # For now, we focus on context. If direct file interaction is needed,
                                # this part might require fetching the file resource via client.files.get(uri)
                                # and then potentially adding it to self.active_files.
                                # However, this adds complexity (async call in sync context or another bg task)
                                # For now, let's assume context from text is primary.
                                logger.info(f"Cached PDF \'{filename}\' also has GenAI URI: {genai_uri}. Context set from text blob.")
                            except Exception as e:
                                logger.warning(f"Could not fully utilize cached GenAI URI {genai_uri} for {filename}: {e}")
                        return # Skip reprocessing
                    except Exception as e:
                        logger.error(f"Error reading cached blob for {filename}: {e}", exc_info=True)
                        print(f"\n⚠️ Error reading cached version of \'{filename}\'. Will attempt to reprocess.")
                else:
                    logger.warning(f"Blob file {blob_full_path} not found for cached PDF \'{filename}\'. Will reprocess.")
            else:
                logger.warning(f"Cached PDF record for \'{filename}\' found but blob_path is missing. Will reprocess.")
        # --- End cache check ---

        if not pdf_path.exists() or not pdf_path.is_file():
            print(f"\n⚠️ Error: PDF file '{filename}' not found in {self.pdfs_dir_abs_path}.")
            return

        arxiv_id_arg = args[1] if len(args) > 1 else None

        # --- Pre-async DB Operation: Create minimal paper record --- 
        paper_id: Optional[int] = None
        try:
            paper_id = database.add_minimal_paper(self.conn, filename)
            if not paper_id:
                print("\n⚠️ Error: Failed to create initial database record.")
                return

            logger.info(f"Created initial DB record for {filename} with ID: {paper_id}")

            if arxiv_id_arg:
                if isinstance(arxiv_id_arg, str) and len(arxiv_id_arg) > 5: 
                    database.update_paper_field(self.conn, paper_id, 'arxiv_id', arxiv_id_arg)
                    logger.info(f"Updated record {paper_id} with provided arXiv ID: {arxiv_id_arg}")
                else:
                    logger.warning(f"Provided arXiv ID '{arxiv_id_arg}' seems invalid for paper {paper_id}. Skipping update.")
            # Update status to indicate async processing is starting
            database.update_paper_field(self.conn, paper_id, 'status', 'processing_async')
        
        except Exception as e: # Reinstate the main exception handler for the function's try block
            logger.error(f"An error occurred during PDF processing for '{filename}': {e}", exc_info=True)
            if paper_id and self.conn:
                try:
                    # Attempt to mark as error if possible
                    database.update_paper_field(self.conn, paper_id, 'status', 'error_setup')
                except Exception as db_err:
                    logger.error(f"Additionally failed to update status to error for paper {paper_id}: {db_err}")
            print(f"\n❌ An unexpected error occurred before starting async processing: {e}")
            return

        # --- Launch Asynchronous Task --- 
        # Use functools.partial to prepare the coroutine with its specific arguments
        # The _process_pdf_async_v2 coroutine expects (task_id, progress_bar, rich_task_id) from _launch_background_task
        # and we add pdf_path, arxiv_id_arg, and paper_id via partial.
        specific_pdf_task_coro_creator = functools.partial(self._process_pdf_async_v2,
                                                       pdf_path=pdf_path,
                                                       arxiv_id=arxiv_id_arg,
                                                       paper_id=paper_id) # _process_pdf_async_v2 will access self.db_path_str

        self._launch_background_task(specific_pdf_task_coro_creator, task_name=f"PDF-{pdf_path.name}")
        # The _launch_background_task will print a message like:
        # "⏳ 'PDF-mypaper.pdf' (ID: <uuid>) started in background – you can keep chatting."
        # The actual result/feedback will come from the _on_task_done callback via prints.

    def _finalize_pdf_ingest(self, pdf_file_resource: types.File, arxiv_id: Optional[str], original_pdf_path: Path, paper_id: Optional[int], db_path_str: Optional[str]):
        """Synchronous method for final PDF ingestion steps after GenAI processing is ACTIVE.
        This method is called via asyncio.to_thread() from an async task.
        It handles text extraction (potentially blocking), blob saving (blocking), 
        database updates, and preparing chat context.
        Args:
            pdf_file_resource: The genai.File object (after it's ACTIVE).
            arxiv_id: Optional arXiv ID from the user or previous steps.
            original_pdf_path: The original Path to the PDF file on local disk.
            paper_id: The database paper_id created before async processing started.
            db_path_str: The database path string to establish a new connection in this thread.
        """
        logger.info(f"Finalize Thread (Paper ID: {paper_id}): Starting for '{pdf_file_resource.display_name}'. Using DB path: {db_path_str}")

        local_conn: Optional[sqlite3.Connection] = None
        try:
            if not paper_id:
                logger.error(f"Finalize Thread: Critical - Missing paper_id for {pdf_file_resource.display_name}. Aborting finalization.")
                return

            if not db_path_str:
                logger.error(f"Finalize Thread Paper ID {paper_id}: Database path not provided. Aborting.")
                return # Cannot update status without DB path
            
            local_conn = database.get_db_connection(Path(db_path_str))
            if not local_conn:
                logger.error(f"Finalize Thread Paper ID {paper_id}: Could not establish new DB connection. Aborting.")
                return # Cannot update status without DB connection

            # Ensure synchronous GenAI client is available (it's on self, created in main thread, typically fine for read-only attributes or creating new requests)
            if not self.client:
                logger.error(f"Finalize Thread Paper ID {paper_id}: Synchronous GenAI client (self.client) not available. Aborting text extraction.")
                database.update_paper_field(local_conn, paper_id, 'status', 'error_extraction_final_no_client')
                return

            extracted_text: Optional[str] = None
            try:
                logger.info(f"Finalize Thread Paper ID {paper_id}: Extracting text from '{original_pdf_path.name}'.")
                extracted_text = tools.extract_text_from_pdf_gemini(original_pdf_path, self.client, self.model_name)
                if not extracted_text:
                    logger.warning(f"Finalize Thread Paper ID {paper_id}: Text extraction returned no content for '{original_pdf_path.name}'.")
                    database.update_paper_field(local_conn, paper_id, 'status', 'error_extraction_final_empty')
                    return
                logger.info(f"Finalize Thread Paper ID {paper_id}: Successfully extracted text ({len(extracted_text)} chars).")
            except Exception as e:
                logger.error(f"Finalize Thread Paper ID {paper_id}: Error during text extraction for '{original_pdf_path.name}': {e}", exc_info=True)
                database.update_paper_field(local_conn, paper_id, 'status', 'error_extraction_final')
                return

            if self.blob_dir:
                blob_filename = f"paper_{paper_id}_text.txt"
                blob_full_path = self.blob_dir / blob_filename
                try:
                    logger.info(f"Finalize Thread Paper ID {paper_id}: Saving extracted text to {blob_full_path}.")
                    save_success = tools.save_text_blob(blob_full_path, extracted_text)
                    if not save_success:
                        logger.error(f"Finalize Thread Paper ID {paper_id}: Failed to save text blob to {blob_full_path}.")
                        database.update_paper_field(local_conn, paper_id, 'status', 'error_blob_final')
                    else:
                        logger.info(f"Finalize Thread Paper ID {paper_id}: Successfully saved text blob. Updating DB.")
                        database.update_paper_field(local_conn, paper_id, 'blob_path', blob_filename)
                except Exception as e:
                    logger.error(f"Finalize Thread Paper ID {paper_id}: Error saving text blob to {blob_full_path}: {e}", exc_info=True)
                    database.update_paper_field(local_conn, paper_id, 'status', 'error_blob_final_exception')
            else:
                logger.warning(f"Finalize Thread Paper ID {paper_id}: Blob directory not configured. Skipping saving text.")

            database.update_paper_field(local_conn, paper_id, 'genai_file_uri', pdf_file_resource.uri)
            database.update_paper_field(local_conn, paper_id, 'processed_timestamp', datetime.now(timezone.utc))

            try:
                context_header = f"CONTEXT FROM PDF ('{pdf_file_resource.display_name}', ID: {paper_id}):\n---"
                if MAX_PDF_CONTEXT_LENGTH is not None and isinstance(MAX_PDF_CONTEXT_LENGTH, int) and MAX_PDF_CONTEXT_LENGTH > 0:
                    text_to_truncate = extracted_text if extracted_text else ""
                    max_text_len = MAX_PDF_CONTEXT_LENGTH - len(context_header) - 50
                    if max_text_len < 0: max_text_len = 0
                    truncated_text = text_to_truncate[:max_text_len]
                    if len(text_to_truncate) > max_text_len:
                        truncated_text += "\n... [TRUNCATED]"
                else:
                    truncated_text = extracted_text if extracted_text else "[No text extracted or available for context]"
                self.pending_pdf_context = f"{context_header}\n{truncated_text}\n---"
                logger.info(f"Finalize Thread Paper ID {paper_id}: Stored context ({len(self.pending_pdf_context)} chars). Updating status.")
                database.update_paper_field(local_conn, paper_id, 'status', 'completed_pending_context')
            except Exception as e:
                logger.error(f"Finalize Thread Paper ID {paper_id}: Error preparing chat context: {e}", exc_info=True)
                database.update_paper_field(local_conn, paper_id, 'status', 'error_context_prep_final')

            logger.info(f"Finalize Thread Paper ID {paper_id}: Finalization complete for '{pdf_file_resource.display_name}'.")
        
        except Exception as e: # Catch-all for unexpected errors during setup or within the try block
            logger.error(f"Finalize Thread Paper ID {paper_id if paper_id else 'Unknown'}: Unhandled exception: {e}", exc_info=True)
            if local_conn and paper_id:
                try:
                    database.update_paper_field(local_conn, paper_id, 'status', 'error_finalize_unhandled')
                except Exception as db_err:
                    logger.error(f"Finalize Thread Paper ID {paper_id if paper_id else 'Unknown'}: Failed to update status to unhandled error: {db_err}")
        finally:
            if local_conn:
                logger.info(f"Finalize Thread Paper ID {paper_id if paper_id else 'Unknown'}: Closing thread-local DB connection.")
                database.close_db_connection(local_conn)

    async def _process_pdf_async_v2(self, task_id: str, pdf_path: Path, arxiv_id: str | None, progress_bar: Progress, rich_task_id, paper_id: Optional[int]):
        """
        Processes a PDF file asynchronously using client.aio.files: uploads to GenAI, monitors processing,
        and finalizes ingestion. Designed for cooperative cancellation.
        """
        if not self.async_client:
            error_message = f"Task {task_id}: Async client not available. Cannot process {pdf_path.name}."
            logging.error(error_message)
            progress_bar.update(rich_task_id, description=f"❌ {pdf_path.name} failed: Async client missing", completed=100, total=100)
            raise RuntimeError(error_message)

        client = self.async_client
        pdf_file_display_name = pdf_path.name
        progress_bar.update(rich_task_id, description=f"Starting {pdf_file_display_name}…")

        genai_file_resource: Optional[types.File] = None

        try:
            progress_bar.update(rich_task_id, description=f"Uploading {pdf_file_display_name}…")
            # TODO: Add timeout for upload if necessary, e.g., asyncio.timeout(60, ...)
            upload_config = types.UploadFileConfig(
                display_name=pdf_path.name # Use the original PDF filename as the display name
            )
            genai_file_resource = await client.files.upload(
                file=pdf_path, 
                config=upload_config
            )
            logger.info(f"Task {task_id}: Uploaded {pdf_file_display_name} as {genai_file_resource.name} ({genai_file_resource.display_name})")

            progress_bar.update(rich_task_id, description=f"Processing {genai_file_resource.display_name} with GenAI…")
            while genai_file_resource.state.name == "PROCESSING":
                await asyncio.sleep(5) # Non-blocking poll interval
                # Refresh file state using its unique resource name (genai_file_resource.name)
                # TODO: Add timeout for get if necessary
                genai_file_resource = await client.files.get(name=genai_file_resource.name)
                logger.debug(f"Task {task_id}: Polled {genai_file_resource.name}, state: {genai_file_resource.state.name}")

            if genai_file_resource.state.name != "ACTIVE": # Check for "ACTIVE" as the desired terminal success state
                error_message = f"Task {task_id}: PDF {genai_file_resource.display_name} processing failed or unexpected state: {genai_file_resource.state.name}"
                logging.error(error_message)
                if genai_file_resource.state.name == "FAILED" and hasattr(genai_file_resource, 'error') and genai_file_resource.error:
                    genai_error_msg = f"GenAI Error Code: {genai_file_resource.error.code}, Message: {genai_file_resource.error.message}"
                    logging.error(f"Task {task_id}: {genai_error_msg}")
                    error_message += f" ({genai_error_msg})"
                progress_bar.update(rich_task_id, description=f"❌ {genai_file_resource.display_name} failed: {genai_file_resource.state.name}", completed=100, total=100)
                raise RuntimeError(error_message)

            progress_bar.update(rich_task_id, description=f"✅ {genai_file_resource.display_name} ready for finalization.", completed=100, total=100)
            logger.info(f"Task {task_id}: {genai_file_resource.display_name} is ACTIVE.")

            # Finalize: DB insert, text extraction (if local), etc.
            # _finalize_pdf_ingest contains blocking calls (text extraction, file I/O for blob)
            # so it must be run in a separate thread to avoid blocking the asyncio event loop.
            await asyncio.to_thread(
                self._finalize_pdf_ingest, 
                genai_file_resource, 
                arxiv_id, 
                pdf_path, # This is original_pdf_path
                paper_id,
                self.db_path_str # Pass the DB path string from self
            )
            
            logging.info(f"Task {task_id}: Successfully processed and finalized {genai_file_resource.display_name}")
            return f"Successfully processed {genai_file_resource.display_name}"

        except asyncio.CancelledError:
            display_name = genai_file_resource.display_name if genai_file_resource else pdf_file_display_name
            logging.info(f"Task {task_id} ({display_name}): Cancelled.")
            progress_bar.update(rich_task_id, description=f"🚫 {display_name} cancelled.", completed=100, total=100)
            # Perform any necessary cleanup specific to this task on cancellation
            if genai_file_resource and genai_file_resource.name:
                try:
                    logger.info(f"Task {task_id} ({display_name}): Attempting to delete GenAI file {genai_file_resource.name} due to cancellation.")
                    await client.files.delete(name=genai_file_resource.name)
                    logger.info(f"Task {task_id} ({display_name}): Successfully deleted GenAI file {genai_file_resource.name} after cancellation.")
                except Exception as del_e:
                    logger.error(f"Task {task_id} ({display_name}): Failed to delete GenAI file {genai_file_resource.name} after cancellation: {del_e}")
            raise

        except Exception as e:
            display_name = genai_file_resource.display_name if genai_file_resource else pdf_file_display_name
            logging.exception(f"Task {task_id} ({display_name}): Error during processing: {str(e)}")
            progress_bar.update(rich_task_id, description=f"❌ {display_name} error: {type(e).__name__}", completed=100, total=100)
            # Optionally, attempt to delete the file from GenAI if it was uploaded before erroring
            if genai_file_resource and genai_file_resource.name:
                try:
                    logger.info(f"Task {task_id} ({display_name}): Attempting to delete GenAI file {genai_file_resource.name} due to error.")
                    await client.files.delete(name=genai_file_resource.name)
                    logger.info(f"Task {task_id} ({display_name}): Successfully deleted GenAI file {genai_file_resource.name} after error.")
                except Exception as del_e:
                    logger.error(f"Task {task_id} ({display_name}): Failed to delete GenAI file {genai_file_resource.name} after error: {del_e}")
            raise

        finally:
            # Ensure progress bar always stops if not already explicitly marked completed/failed by an update.
            # This check might be overly cautious if all paths correctly update and stop the bar.
            # progress_bar.stop() # This is handled by _on_task_done
            pass

    def _on_task_done(self, task_id: str, task_name: str, future: asyncio.Future):
        """Callback executed when a background task finishes."""
        try:
            result = future.result() # Raise exception if task failed
            print(f"\n✅ Task '{task_name}' (ID: {task_id}) completed successfully. Result: {result}")
            logging.info(f"Task '{task_name}' (ID: {task_id}) completed successfully. Result: {result}")
        except asyncio.CancelledError:
            print(f"\n🚫 Task '{task_name}' (ID: {task_id}) was cancelled.")
            logging.warning(f"Task '{task_name}' (ID: {task_id}) was cancelled.")
        except Exception as e:
            print(f"\n❌ Task '{task_name}' (ID: {task_id}) failed: {type(e).__name__}: {e}")
            logging.error(f"Task '{task_name}' (ID: {task_id}) failed.", exc_info=True)
        finally:
            # Remove task from active list
            task_info = self.active_background_tasks.pop(task_id, None)
            # Stop progress bar if it exists and task_info is not None
            if task_info and "progress_bar" in task_info:
                task_info["progress_bar"].stop()
            # Potentially refresh prompt or UI
            # Check task_meta for script execution output
            if task_info and future.exception() is None and not future.cancelled():
                meta = task_info.get("meta", {})
                if meta.get("type") == "script_execution":
                    script_output = future.result() # This is the string output from _execute_script_async
                    self.pending_script_output = script_output
                    original_command = meta.get("original_command", "Unknown script")
                    # Truncate for print message if too long
                    output_summary = (script_output[:100] + '...') if len(script_output) > 103 else script_output
                    print(f"\n📄 Output from '{original_command}' is ready and will be included in the next context. Output preview:\n{output_summary}")
                    logger.info(f"Task '{task_name}' (ID: {task_id}) was a script execution. Output stored in pending_script_output.")

    def _launch_background_task(self, coro_func, task_name: str, progress_total: float = 100.0, task_meta: Optional[dict] = None):
        """
        Launches a coroutine as a background task with progress display.
        `coro_func` should be a functools.partial or lambda that creates the coroutine,
        and the coroutine it creates should accept (task_id, progress_bar, rich_task_id) as arguments.
        `task_meta` is an optional dictionary to store extra info about the task.
        """
        task_id = str(uuid.uuid4())

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            # transient=True # Consider if progress should disappear after completion
        )
        rich_task_id = progress.add_task(description=f"Initializing {task_name}...", total=progress_total)

        # The coroutine (created by coro_func) must accept these specific arguments.
        task_coro = coro_func(task_id=task_id, progress_bar=progress, rich_task_id=rich_task_id)

        fut = asyncio.run_coroutine_threadsafe(task_coro, self.loop)
        self.active_background_tasks[task_id] = {
            "future": fut,
            "name": task_name,
            "progress_bar": progress, # Store to stop it in _on_task_done
            "rich_task_id": rich_task_id,
            "meta": task_meta if task_meta else {} # Store metadata
        }

        fut.add_done_callback(
            lambda f: self._on_task_done(task_id, task_name, f)
        )

        print(f"⏳ '{task_name}' (ID: {task_id}) started in background – you can keep chatting.")
        # The Progress object itself will be updated by the task coroutine.
        # How it's displayed (e.g., via rich.Live or direct printing) will be
        # handled by the calling environment or a dedicated UI management part.
        return task_id # Return task_id along with progress for better management if needed in future

    def _handle_cancel_command(self, task_id_str: str):
        """Attempts to cancel an active background task."""
        task_info = self.active_background_tasks.get(task_id_str)
        if not task_info:
            print(f"\n❌ Task ID '{task_id_str}' not found or already completed.")
            return

        future = task_info.get("future")
        task_name = task_info.get("name", "Unnamed Task")

        if future and not future.done():
            cancelled = future.cancel()
            if cancelled:
                print(f"\n➡️ Cancellation request sent for task '{task_name}' (ID: {task_id_str}).")
                # The _on_task_done callback will eventually report it as cancelled.
            else:
                print(f"\n❌ Failed to send cancellation request for task '{task_name}' (ID: {task_id_str}). It might be already completing or uncancelable.")
        elif future and future.done():
            print(f"\nℹ️ Task '{task_name}' (ID: {task_id_str}) has already completed.")
        else:
            print(f"\n⚠️ Could not cancel task '{task_name}' (ID: {task_id_str}). Future object missing or invalid state.")

    def _handle_list_tasks_command(self):
        """Lists active background tasks."""
        if not self.active_background_tasks:
            print("\nℹ️ No active background tasks.")
            return

        print("\n📋 Active Background Tasks:")
        for task_id, info in self.active_background_tasks.items():
            future = info.get("future")
            name = info.get("name", "Unnamed Task")
            status = "Running"
            if future:
                if future.cancelled():
                    status = "Cancelling"
                elif future.done(): # Should ideally be removed by _on_task_done, but check just in case
                    status = "Completed (Pending Removal)"
            print(f"  - ID: {task_id}, Name: {name}, Status: {status}")

    def _handle_run_script_command(self, script_type: str, script_path: str, script_args: list[str]):
        """Handles the /run_script command to execute a script asynchronously."""
        logger.info(f"Received /run_script command: type={script_type}, path={script_path}, args={script_args}")
        
        # Basic validation for script_path to prevent execution of arbitrary system commands if script_type is 'shell'
        # and script_path is not a path but a command itself. For now, we assume script_path is a path.
        # More robust validation might be needed depending on security requirements.
        if ".." in script_path or script_path.startswith("/"):
            print("\n⚠️ Error: Script path should be relative and within the current workspace/scripts directory.")
            logger.warning(f"Potentially unsafe script path provided: {script_path}")
            return

        task_name = f"Script-{Path(script_path).name}"
        original_full_command = f"{script_type} {script_path} {' '.join(script_args)}".strip()
        task_meta = {"type": "script_execution", "original_command": original_full_command}

        script_coro_creator = functools.partial(self._execute_script_async,
                                              script_type=script_type,
                                              script_path_str=script_path,
                                              script_args=script_args)
        
        self._launch_background_task(script_coro_creator, task_name=task_name, task_meta=task_meta)

    async def _execute_script_async(self, task_id: str, progress_bar: Progress, rich_task_id, script_type: str, script_path_str: str, script_args: list[str]):
        """Asynchronously executes a python or shell script and captures its output."""
        progress_bar.update(rich_task_id, description=f"Preparing {script_type} script: {Path(script_path_str).name}")
        
        # Define a scripts directory (e.g., project_root / 'scripts')
        # For now, let's assume scripts are relative to the workspace_root (agent's CWD)
        # A more robust solution would involve a config for allowed script directories.
        workspace_root = Path(os.getcwd()) # Or a configured workspace root
        abs_script_path = (workspace_root / script_path_str).resolve()

        # Security check: Ensure the script is within the intended workspace/scripts directory
        # This is a basic check; more sophisticated sandboxing might be needed for untrusted scripts.
        if not str(abs_script_path).startswith(str(workspace_root)):
            error_msg = f"Error: Script path '{script_path_str}' is outside the allowed workspace."
            logger.error(f"Task {task_id}: {error_msg}")
            progress_bar.update(rich_task_id, description=f"❌ Error: Path outside workspace", completed=100, total=100)
            return error_msg # Return error message for _on_task_done to handle

        if not abs_script_path.is_file():
            error_msg = f"Error: Script not found at '{abs_script_path}'."
            logger.error(f"Task {task_id}: {error_msg}")
            progress_bar.update(rich_task_id, description=f"❌ Error: Script not found", completed=100, total=100)
            return error_msg

        command_list = []
        if script_type == "python":
            command_list = [sys.executable, str(abs_script_path)] + script_args
        elif script_type == "shell":
            # Ensure the shell script itself is executable by the user
            if not os.access(abs_script_path, os.X_OK):
                error_msg = f"Error: Shell script '{abs_script_path}' is not executable. Please use chmod +x."
                logger.error(f"Task {task_id}: {error_msg}")
                progress_bar.update(rich_task_id, description=f"❌ Error: Script not executable", completed=100, total=100)
                return error_msg
            command_list = [str(abs_script_path)] + script_args
        else:
            error_msg = f"Error: Unsupported script type '{script_type}'. Must be 'python' or 'shell'."
            logger.error(f"Task {task_id}: {error_msg}")
            progress_bar.update(rich_task_id, description=f"❌ Error: Invalid script type", completed=100, total=100)
            return error_msg

        try:
            progress_bar.update(rich_task_id, description=f"Running {Path(script_path_str).name}...")
            logger.info(f"Task {task_id}: Executing command: {' '.join(command_list)}")

            # Execute in a separate thread as subprocess.run is blocking
            process = await asyncio.to_thread(
                subprocess.run, 
                command_list, 
                capture_output=True, 
                text=True, 
                check=False, # Handle non-zero exit codes manually
                cwd=workspace_root # Run script with CWD as workspace root
            )
            
            output = f"--- Script: {Path(script_path_str).name} ---"
            output += f"\n--- Return Code: {process.returncode} ---"
            if process.stdout:
                output += f"\n--- STDOUT ---\n{process.stdout.strip()}"
            if process.stderr:
                output += f"\n--- STDERR ---\n{process.stderr.strip()}"
            
            if process.returncode == 0:
                progress_bar.update(rich_task_id, description=f"✅ {Path(script_path_str).name} finished.", completed=100, total=100)
            else:
                progress_bar.update(rich_task_id, description=f"⚠️ {Path(script_path_str).name} finished with errors.", completed=100, total=100)
            
            logger.info(f"Task {task_id}: Script '{Path(script_path_str).name}' finished. RC: {process.returncode}")
            return output

        except asyncio.CancelledError:
            logger.info(f"Task {task_id} (Script: {Path(script_path_str).name}): Cancelled.")
            progress_bar.update(rich_task_id, description=f"🚫 {Path(script_path_str).name} cancelled.", completed=100, total=100)
            raise # Re-raise to be handled by _on_task_done
        except Exception as e:
            logger.exception(f"Task {task_id} (Script: {Path(script_path_str).name}): Error during execution.")
            progress_bar.update(rich_task_id, description=f"❌ {Path(script_path_str).name} error: {type(e).__name__}", completed=100, total=100)
            return f"Error executing script {Path(script_path_str).name}: {type(e).__name__}: {e}" # Return error for _on_task_done

def main():
    config_path = Path('src/config.yaml')
    config = load_config(config_path)
    if not config:
        sys.exit(1)

    print_welcome_message(config)

    # --- Database Setup --- 
    db_path_str = config.get('PAPER_DB_PATH')
    conn = None
    if db_path_str:
        db_path = Path(db_path_str).resolve()
        logger.info(f"Attempting to connect to database: {db_path}")
        conn = database.get_db_connection(db_path)
        if conn:
            try:
                 # Ensure tables exist
                 database.create_tables(conn)
                 logger.info("Database tables checked/created successfully.")
            except Exception as db_init_err:
                 logger.error(f"Failed to initialize database tables: {db_init_err}", exc_info=True)
                 database.close_db_connection(conn)
                 conn = None # Prevent agent from using bad connection
                 print("\n⚠️ CRITICAL: Failed to initialize database. Exiting.")
                 sys.exit(1)
        else:
            print("\n⚠️ Warning: Failed to establish database connection. Proceeding without database features.")
    else:
        print("\n⚠️ Warning: 'PAPER_DB_PATH' not specified in config.yaml. Proceeding without database features.")
    # --- End Database Setup ---

    try:
        # Pass the established connection (or None) to the agent
        agent = CodeAgent(config=config, conn=conn)
        agent.start_interaction()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")
    finally:
        # Ensure database connection is closed on exit
        if conn:
             logger.info("Closing database connection...")
             database.close_db_connection(conn)
             logger.info("Database connection closed.")
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
