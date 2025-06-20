import os
import sys
from pathlib import Path
from . import database
from . import tools
from .autocomplete import PdfCompleter
import traceback
import argparse
import functools # Keep for CodeAgent._make_verbose_tool if not fully removed, or for other potential uses
import logging
import asyncio, threading, uuid
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn
from prompt_toolkit.document import Document
from prompt_toolkit.completion import Completer, Completion, NestedCompleter, WordCompleter, PathCompleter, FuzzyWordCompleter, CompleteEvent
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.keys import Keys
from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import re
from typing import Iterable, Dict, Any, Optional, Set, Union, List, Callable # Ensure List, Callable are here
import bisect
import yaml
import sqlite3
from dotenv import load_dotenv
from datetime import datetime, timezone
import subprocess

# --- Import LLMAgentCore ---
from .llm_agent_core import LLMAgentCore
import google.generativeai as genai # Use the working import style
from .async_task_manager import AsyncTaskManager # Import AsyncTaskManager

# --- Import slash command handlers ---
from . import slashcommands


# Setup basic logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# --- Completer Logging Wrapper ---
class LoggingCompleterWrapper(Completer): # Keep this class as it's UI related
    def __init__(self, wrapped_completer: Completer, name: str = "Wrapped"):
        self.wrapped_completer = wrapped_completer
        self.name = name

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        logger.info(f"[{self.name}] get_completions CALLED. Text: '{document.text_before_cursor}', UserInvoked: {complete_event.completion_requested}")
        raw_completions = list(self.wrapped_completer.get_completions(document, complete_event))
        logger.info(f"[{self.name}] Raw completions from wrapped: {[c.text for c in raw_completions]} for input '{document.text_before_cursor}'")
        yield from raw_completions
# --- End Completer Logging Wrapper ---

class CustomNestedCompleter(NestedCompleter): # Keep this UI related class
    def __init__(self, options: Dict[str, Optional[Completer]], ignore_case: bool = True):
        super().__init__(options, ignore_case=ignore_case)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        # Use current line for command context to simplify multi-line scenarios.
        # Assumption: A slash command and its arguments reside on a single line.
        line_text_before_cursor = document.current_line_before_cursor
        line_cursor_col = document.cursor_position_col # Cursor position within the current line

        # Find all non-space sequences (words) in the current line before the cursor
        words_info = list(re.finditer(r'\S+', line_text_before_cursor))

        command_segment_doc = None

        # Iterate backwards through words in the current line to find the last one starting with '/' 
        # that the cursor is engaged with.
        for i in range(len(words_info) - 1, -1, -1):
            word_match = words_info[i]
            word_text = word_match.group(0)
            word_start_col_in_line = word_match.start()
            word_end_col_in_line = word_match.end()

            is_cursor_engaged_with_word_in_line = \
                (line_cursor_col >= word_start_col_in_line and line_cursor_col <= word_end_col_in_line) or \
                (line_cursor_col == word_end_col_in_line + 1 and line_text_before_cursor.endswith(' '))

            if word_text.startswith('/') and is_cursor_engaged_with_word_in_line:
                segment_text_on_line = line_text_before_cursor[word_start_col_in_line:]
                segment_cursor_pos_in_line = line_cursor_col - word_start_col_in_line

                command_segment_doc = Document(
                    text=segment_text_on_line,
                    cursor_position=segment_cursor_pos_in_line
                )
                # logger.debug(f"Active command segment on line: '{command_segment_doc.text}', cursor at {command_segment_doc.cursor_position}")
                break
        
        if not command_segment_doc:
            # logger.debug("No active command segment on current line for completion.")
            yield from []
            return

        # Now, use command_segment_doc for completion logic (this part remains the same)
        sub_text_before_cursor = command_segment_doc.text_before_cursor.lstrip()

        if ' ' not in sub_text_before_cursor:
            # Completing the first word of the active command segment (e.g., /cmd)
            actual_keys = [k for k in self.options.keys() if isinstance(k, str)]
            first_word_completer = WordCompleter(
                words=actual_keys,
                ignore_case=self.ignore_case,
                match_middle=True,
                sentence=False
            )
            # logger.debug(f"Using FirstWordCompleter for segment: '{command_segment_doc.text}'")
            for c in first_word_completer.get_completions(command_segment_doc, complete_event):
                insert_text = c.text
                display_text = c.text
                # Handle the slash duplication only if the segment itself starts with '/' (it should)
                if command_segment_doc.text.lstrip().startswith('/') and c.text.startswith('/') and len(c.text) > 1:
                    insert_text = c.text[1:]
                
                yield Completion(
                    text=insert_text,
                    start_position=c.start_position, # Relative to command_segment_doc cursor
                    display=display_text,
                    display_meta=c.display_meta,
                    style=c.style,
                    selected_style=c.selected_style
                )
        else:
            # Completing sub-commands of the active command segment
            # logger.debug(f"Delegating to super for segment: '{command_segment_doc.text}'")
            for c in super().get_completions(command_segment_doc, complete_event):
                yield Completion(
                    text=c.text,
                    start_position=c.start_position, # Relative to command_segment_doc cursor
                    display=c.display,
                    display_meta=c.display_meta,
                    style=c.style,
                    selected_style=c.selected_style
                )

# Helper function for the key binding condition
def is_typing_slash_command_prefix(current_buffer):
    # This filter is for the Keys.Any binding. It's evaluated *before* the typed char is inserted.
    text_up_to_cursor = current_buffer.document.text_before_cursor

    # If the cursor is immediately after a '/', we are likely starting/typing a command.
    if text_up_to_cursor.endswith('/'): # Catches "foo /" or just "/"
        return True

    # Find the current "word" or segment the cursor is in or at the end of.
    # A "word" here is a sequence of non-space characters.
    match = re.search(r'(\S+)$', text_up_to_cursor)
    if match:
        current_segment = match.group(1)
        # If this segment starts with '/', we're typing a slash command or its argument.
        # e.g., "foo /c" (current_segment='/c') or "foo /cmd" (current_segment='/cmd')
        if current_segment.startswith('/'):
            return True
    
    # This covers cases like:
    # "/cm" -> current_segment = "/cm", returns True
    # "foo /cm" -> current_segment = "/cm", returns True
    # "foo /cmd arg" -> current_segment = "arg". Fails here. This is intended.
    #   The completer should handle this. The filter's job is to see if we're in a slash context.
    #   Actually, for "foo /cmd arg", if the completer logic is correct, it will create a
    #   segment_doc = "/cmd arg" and then delegate to super().get_completions. 
    #   The Keys.Any filter should probably be true if the *active command segment* (as defined by the completer) exists.
    #   However, re-evaluating that full logic in the filter is too much.
    #   The current refined logic for the filter is a heuristic: if the current word at cursor starts with /, or prev char is /.

    # Let's reconsider "foo /cmd arg":
    # If cursor is after 'g' in 'arg': text_up_to_cursor = "foo /cmd arg"
    # current_segment = "arg". Does not start with '/'. Returns False.
    # This means live suggestions for arguments of a mid-line command won't trigger via Keys.Any with this filter.
    # This might be acceptable if Tab completion still works for those arguments.
    # The CustomNestedCompleter *will* provide completions if Tab is pressed in "foo /cmd arg|".

    # To enable Keys.Any for arguments of mid-line commands, the filter would need to be smarter,
    # potentially mirroring the CustomNestedCompleter's segment finding logic. For now, let's keep it simpler.
    # The primary goal is that typing `/` mid-line starts completions, and typing inside `/cmd` mid-line continues them.

    return False

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

    # 1. Load YAML file specified by config_path
    try:
        with open(config_path) as f_local:
            loaded_yaml_config = yaml.safe_load(f_local) or {}
    except Exception as e:
        logger.error(f"Failed to load or parse YAML configuration at {config_path}: {e}", exc_info=True)
        # Return a minimal config or raise error, depending on desired handling
        # For now, return a dictionary that might miss keys, leading to later warnings/defaults.
        loaded_yaml_config = {}

    # 2. Load environment variables from ~/.env (if it exists)
    dotenv_home_path = Path.home() / '.env'
    if dotenv_home_path.is_file():
        load_dotenv(dotenv_path=dotenv_home_path)
        logger.info(f"Loaded environment variables from {dotenv_home_path}")
    else:
        logger.info(f"No .env file found at {dotenv_home_path}, checking system environment variables.")

    # 3. Check for API key in environment variables first
    env_api_key = os.getenv('GEMINI_API_KEY')

    # 4. Prioritize environment variable for API key
    if env_api_key:
        loaded_yaml_config['gemini_api_key'] = env_api_key
        logger.info("Using GEMINI_API_KEY from environment.")
    elif 'gemini_api_key' in loaded_yaml_config and loaded_yaml_config['gemini_api_key']:
        logger.info("Using gemini_api_key from config.yaml (environment variable not set).")
    else:
        logger.warning("GEMINI_API_KEY not found in environment or config.yaml.")

    # Update MODEL_NAME based on the loaded config (or use its existing global default)
    # This function will return a config dictionary, the caller should update global MODEL_NAME if needed,
    # or better, MODEL_NAME should be sourced from this returned config by the caller.
    # For now, just log based on what this function sees.
    current_model_name = loaded_yaml_config.get('model_name', MODEL_NAME) # Use global MODEL_NAME as fallback for logging
    logger.info(f"Model specified in config/default: {current_model_name}")


    # 5. Resolve paths (relative to project root, which is parent of src/)
    project_root = Path(__file__).parent.parent 
    for key in ['PDFS_TO_CHAT_WITH_DIRECTORY', 
                'SAVED_CONVERSATIONS_DIRECTORY', 
                'PAPER_DB_PATH', 
                'PAPER_BLOBS_DIR']:
        if key in loaded_yaml_config and isinstance(loaded_yaml_config[key], str):
            resolved_path = project_root / loaded_yaml_config[key]
            if key == 'PAPER_DB_PATH':
                 resolved_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                 resolved_path.mkdir(parents=True, exist_ok=True)
            loaded_yaml_config[key] = resolved_path
            logger.info(f"Resolved path for {key}: {loaded_yaml_config[key]}")
        elif key in loaded_yaml_config and isinstance(loaded_yaml_config[key], Path):
             logger.info(f"Path for {key} already resolved: {loaded_yaml_config[key]}")
             if key == 'PAPER_DB_PATH':
                  loaded_yaml_config[key].parent.mkdir(parents=True, exist_ok=True)
             else:
                  loaded_yaml_config[key].mkdir(parents=True, exist_ok=True)
        # If key is missing or not a string/Path, it will be handled by downstream code using .get() with defaults

    return loaded_yaml_config

def print_welcome_message(config):
    """Prints the initial welcome and help message."""
    print("\n🚀 Starting Code Agent...")
    # Add other initial prints if needed

# --- Agent Class ---
class CodeAgent:
    def __init__(self, config: dict, conn: Optional[sqlite3.Connection]):
        """Initializes the CodeAgent."""
        self.config = config
        self.api_key = config.get('gemini_api_key')
        # LLMAgentCore now manages its own model_name, client, chat, thinking_budget, etc.
        
        self.llm_core = LLMAgentCore(
            config=config,
            api_key=self.api_key,
            model_name=MODEL_NAME # Pass MODEL_NAME loaded from top of main.py
        )

        self.pdf_processing_method = config.get('pdf_processing_method', 'Gemini')
        self.db_path_str = str(config.get('PAPER_DB_PATH')) if config.get('PAPER_DB_PATH') else None
        self.prefill_prompt_content: Optional[str] = None

        # Initialize AsyncTaskManager, it handles its own loop and thread
        self.task_manager = AsyncTaskManager(main_app_handler=self)

        # Async client for PDF processing is now accessed via self.llm_core.async_client

        # active_background_tasks is now managed by self.task_manager
        self.pending_script_output: Optional[str] = None # Script output callback will set this

        # active_files are GenAI File objects, managed by CodeAgent (UI layer)
        self.active_files: List[genai.types.File] = [] # Changed genai_types.File to genai.types.File

        # UI-related stats, keep them here
        self.prompt_time_counts = [0]
        self.messages_per_interval = [0]
        self._messages_this_interval = 0
        
        # PDF and Blob directories remain relevant for CodeAgent's handling of files
        self.pdfs_dir_rel_path = config.get('PDFS_TO_CHAT_WITH_DIRECTORY')
        self.pdfs_dir_abs_path = Path(self.pdfs_dir_rel_path).resolve() if self.pdfs_dir_rel_path else None
        self.blob_dir_rel_path = config.get('PAPER_BLOBS_DIR')
        self.blob_dir = Path(self.blob_dir_rel_path).resolve() if self.blob_dir_rel_path else None
        self.conn = conn

        # Define the list of tool functions that CodeAgent makes available
        original_tool_functions = [
            tools.read_file, tools.list_files, tools.edit_file,
            tools.execute_bash_command, tools.run_in_sandbox,
            tools.find_arxiv_papers, tools.download_arxiv_paper,
            tools.get_current_date_and_time, tools.google_search, # Re-enabled
            tools.open_url, # Re-enabled
            tools.upload_pdf_for_gemini, tools.run_sql_query
        ]

        # Register tools with LLMAgentCore
        # LLMAgentCore's register_tools now handles verbose wrapping internally.
        if self.llm_core:
            self.llm_core.register_tools(
                original_tool_functions,
                verbose=self.config.get('verbose', False)
            )

        if self.pdfs_dir_abs_path:
            self.pdfs_dir_abs_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"PDF directory set to: {self.pdfs_dir_abs_path}")
        else:
            logger.warning("PDF directory not configured. /pdf command will be disabled.")

        if self.blob_dir:
            self.blob_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Blob directory set to: {self.blob_dir}")
        else:
            logger.warning("Blob directory not configured. Saving extracted text will be disabled.")

        if not self.conn:
             logger.warning("DB connection not established. DB operations will be disabled.")

        # LLMAgentCore handles its own client/chat initialization and warnings.
        # No need for direct calls to self._configure_client() or self._initialize_chat() here.

        self.pending_pdf_context: Optional[str] = None
        self.prompts_dir = Path('src/prompts').resolve()
        self.prompts_dir.mkdir(parents=True, exist_ok=True) # Ensure prompts dir exists

    # _configure_client, _initialize_chat, _make_verbose_tool are removed from CodeAgent
    # _on_task_done, _launch_background_task are also removed (moved to AsyncTaskManager)

    def handle_script_completion(self, task_id: str, task_name: str, script_output: str):
        """Callback for AsyncTaskManager to set script output."""
        logger.info(f"CodeAgent: Script task '{task_name}' (ID: {task_id}) completed. Output received.")
        self.pending_script_output = script_output
        output_summary = (script_output[:100] + '...') if len(script_output) > 103 else script_output
        # This print might be better handled by a UI update method if using a more complex UI
        print(f"\n📄 Output from '{task_name}' is ready and will be included in the next context. Output preview:\n{output_summary}")
        self.refresh_prompt_display() # Example: if prompt needs to be re-rendered

    def refresh_prompt_display(self):
        """Placeholder for refreshing the prompt_toolkit display if needed after async updates."""
        # In a real prompt_toolkit app, this might involve app.invalidate() or similar
        app = get_app()
        if app:
            app.invalidate()
            logger.debug("CodeAgent: Prompt display invalidated for refresh.")

    def update_task_status_display(self, task_id: str, message: str):
        """Placeholder for updating task status in a more integrated UI."""
        # This could update a Rich Panel, status bar, etc. For now, just prints.
        print(message) # Rich progress is handled by AsyncTaskManager for now

    # --- Prompt Helper Methods (UI related, stay in CodeAgent) ---
    def _list_available_prompts(self) -> List[str]:
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
        # Get token count from LLMAgentCore
        token_count = self.llm_core.current_token_count if self.llm_core else 0
        token_info = f"({token_count})"
        
        return HTML(
            f'<ansiblue>🔵 You</ansiblue> '
            f'<ansicyan>{token_info}</ansicyan>'
            f'<ansigreen>{active_files_info}</ansigreen>: '
        )

    def _get_continuation_prompt(self, width, line_number, is_soft_wrap): # UI related
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

    def print_initial_help(self): # UI related
        """Prints the initial brief help message."""
        print("\n\u2692\ufe0f Agent ready. Ask me anything or type '/help' for commands.")
        print("   Type '/exit' or '/q' to quit.")

    def start_interaction(self):
        """Starts the main interaction loop with enhanced multi-line editing."""
        if not self.llm_core or not self.llm_core.client or not self.llm_core.chat:
            logger.error("LLMAgentCore not initialized properly. Exiting.")
            print("\n\u274c LLM Core not configured (API key or model issue likely). Exiting.")
            return

        self.print_initial_help()

        # thinking_budget is now managed by llm_core, display it
        print(f"\n🧠 Initial thinking budget set to: {self.llm_core.thinking_budget} tokens.")

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
        slash_command_completer_map['/prompt'] = WordCompleter(self._list_available_prompts(), ignore_case=True)
        
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
                # --- House-keeping before prompt ---
                # Use token count from llm_core for UI stats if needed by prompt_time_counts
                self.prompt_time_counts.append(self.llm_core.current_token_count if self.llm_core else 0)
                self.messages_per_interval.append(self._messages_this_interval)
                self._messages_this_interval = 0

                # --- Get multi-line input ---
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

                # Combine pending PDF context and script output for the message
                combined_pending_context = ""
                if self.pending_pdf_context:
                    combined_pending_context += self.pending_pdf_context + "\n\n"
                    print("[Including context from previously processed PDF in this message.]\n")
                if self.pending_script_output:
                    combined_pending_context += f"OUTPUT FROM EXECUTED SCRIPT:\n---\n{self.pending_script_output}\n---\n\n"
                    print("[Including output from previously run script in this message.]\n")
                
                # Log message details before sending
                logger.info(f"Preparing to send message to LLMAgentCore.")
                logger.info(f"  - Original user input: '{user_input}'")
                if combined_pending_context:
                     logger.info(f"  - Combined pending context length: {len(combined_pending_context)}")
                if self.active_files:
                     logger.info(f"  - Active files: {[f.name for f in self.active_files]}")

                print("\n⏳ Sending message to LLM Core and processing...")
                agent_response_text = self.llm_core.send_message(
                    user_message_text=user_input,
                    active_files=self.active_files, # Pass the list of genai.File objects
                    pending_context=combined_pending_context.strip() if combined_pending_context else None
                )

                print(f"\n🟢 \x1b[92mAgent:\x1b[0m {agent_response_text or '[No response text]'}")

                # Display token count from LLMAgentCore
                print(f"\n[Token Count: {self.llm_core.current_token_count}]")

                # Clear contexts now that they've been sent
                if self.pending_pdf_context:
                    self.pending_pdf_context = None
                    logger.info("Cleared pending_pdf_context in CodeAgent after sending to LLM Core.")
                if self.pending_script_output:
                    self.pending_script_output = None
                    logger.info("Cleared pending_script_output in CodeAgent after sending to LLM Core.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n🔴 An error occurred during interaction: {e}")
                traceback.print_exc()

    # _make_verbose_tool is removed, LLMAgentCore handles verbose tool logging.

    def _handle_pdf_command(self, args: list): # UI related, but calls core async client
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

        # Async client check (used by _process_pdf_async_v2, now from llm_core)
        if not self.llm_core or not self.llm_core.async_client:
            print("\n⚠️ Async Gemini client (via LLM Core) not initialized. Cannot process PDF asynchronously.")
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
                        if genai_uri and self.llm_core and self.llm_core.client: # Check llm_core.client
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
                                                       arxiv_id=arxiv_id_arg, # This was arxiv_id_arg before, ensure consistency
                                                       paper_id=paper_id)

        self.task_manager.submit_task(
            specific_pdf_task_coro_creator,
            task_name=f"PDF-{pdf_path.name}",
            task_meta={"type": "pdf_processing", "original_filename": pdf_path.name}
        )

    def handle_pdf_completion(self, task_id: str, task_name: str, result: Any):
        """Callback for AsyncTaskManager upon PDF processing completion."""
        # Result might be a status message or the genai.File object if processing was successful
        # The _process_pdf_async_v2 already sets self.pending_pdf_context via _finalize_pdf_ingest
        logger.info(f"CodeAgent: PDF Processing task '{task_name}' (ID: {task_id}) completed by manager. Result: {result}")
        # Further actions can be taken here based on the result if needed.
        # For example, if result indicates success, print a specific message to user.
        # The _on_task_done in AsyncTaskManager already prints a generic completion message.
        # self.pending_pdf_context is set within _finalize_pdf_ingest, called by the async task.

    def _finalize_pdf_ingest(self, pdf_file_resource: genai.types.File, arxiv_id: Optional[str], original_pdf_path: Path, paper_id: Optional[int], db_path_str: Optional[str]):
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

            # Ensure synchronous GenAI client is available via llm_core
            if not self.llm_core or not self.llm_core.client:
                logger.error(f"Finalize Thread Paper ID {paper_id}: Synchronous GenAI client (self.llm_core.client) not available. Aborting text extraction.")
                database.update_paper_field(local_conn, paper_id, 'status', 'error_extraction_final_no_client')
                return

            extracted_text: Optional[str] = None
            try:
                logger.info(f"Finalize Thread Paper ID {paper_id}: Extracting text from '{original_pdf_path.name}'.")
                # Use client and model_name from llm_core
                extracted_text = tools.extract_text_from_pdf_gemini(original_pdf_path, self.llm_core.client, self.llm_core.model_name)
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
        # progress_bar and rich_task_id are now passed by AsyncTaskManager.submit_task
        if not self.llm_core or not self.llm_core.async_client:
            error_message = f"Task {task_id}: Async client (via LLM Core) not available. Cannot process {pdf_path.name}."
            logger.error(error_message)
            if progress_bar: progress_bar.update(rich_task_id, description=f"❌ {pdf_path.name} failed: Async client missing", completed=True)
            raise RuntimeError(error_message)

        async_genai_client = self.llm_core.async_client
        pdf_file_display_name = pdf_path.name
        progress_bar.update(rich_task_id, description=f"Starting {pdf_file_display_name}…")

        genai_file_resource: Optional[genai.types.File] = None # Changed types.File to genai.types.File

        try:
            progress_bar.update(rich_task_id, description=f"Uploading {pdf_file_display_name}…")
            # TODO: Add timeout for upload if necessary, e.g., asyncio.timeout(60, ...)
            upload_config = genai.types.UploadFileConfig( # Changed genai_types to genai.types
                display_name=pdf_path.name
            )
            genai_file_resource = await async_genai_client.files.upload( # Use the alias
                file=pdf_path, 
                config=upload_config
            )
            logger.info(f"Task {task_id}: Uploaded {pdf_file_display_name} as {genai_file_resource.name} ({genai_file_resource.display_name})")

            progress_bar.update(rich_task_id, description=f"Processing {genai_file_resource.display_name} with GenAI…")
            while genai_file_resource.state.name == "PROCESSING":
                await asyncio.sleep(5)
                genai_file_resource = await async_genai_client.files.get(name=genai_file_resource.name) # Use the alias
                logger.debug(f"Task {task_id}: Polled {genai_file_resource.name}, state: {genai_file_resource.state.name}")

            if genai_file_resource.state.name != "ACTIVE":
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
                    await async_genai_client.files.delete(name=genai_file_resource.name) # Use the alias
                    logger.info(f"Task {task_id} ({display_name}): Successfully deleted GenAI file {genai_file_resource.name} after error.")
                except Exception as del_e:
                    logger.error(f"Task {task_id} ({display_name}): Failed to delete GenAI file {genai_file_resource.name} after error: {del_e}")
            raise

        finally:
            # The AsyncTaskManager._on_task_done will handle stopping the progress bar.
            pass

    # _on_task_done and _launch_background_task are now part of AsyncTaskManager.

    def _handle_cancel_command(self, task_id_str: str):
        """Delegates task cancellation to AsyncTaskManager."""
        if not task_id_str: # Basic validation, though slashcommands.py should ensure arg
            print("\n⚠️ Usage: /cancel <task_id>")
            return
        self.task_manager.cancel_task(task_id_str)

    def _handle_list_tasks_command(self):
        """Delegates listing tasks to AsyncTaskManager."""
        self.task_manager.list_tasks()

    def _handle_run_script_command(self, script_type: str, script_path: str, script_args: list[str]):
        """Submits a script execution task to AsyncTaskManager."""
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

        script_coro_creator = functools.partial(self._execute_script_async, # _execute_script_async is now part of CodeAgent
                                              script_type=script_type,
                                              script_path_str=script_path,
                                              script_args=script_args)
        
        self.task_manager.submit_task(script_coro_creator, task_name=task_name, task_meta=task_meta)

    async def _execute_script_async(self, task_id: str, progress_bar: Progress, rich_task_id: Any, script_type: str, script_path_str: str, script_args: list[str]):
        """Asynchronously executes a python or shell script and captures its output.
        This method is now a coroutine that will be run by AsyncTaskManager.
        """
        # progress_bar and rich_task_id are passed by the AsyncTaskManager
        if progress_bar: progress_bar.update(rich_task_id, description=f"Preparing {script_type} script: {Path(script_path_str).name}", start=True)
        
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

    agent = None  # Initialize agent to None for the finally block
    try:
        # Pass the established connection (or None) to the agent
        agent = CodeAgent(config=config, conn=conn)
        agent.start_interaction()
    except KeyboardInterrupt:
        print("\nExiting...") # User interruption
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")
    finally:
        # Ensure database connection is closed on exit
        if conn:
            logger.info("Closing database connection...")
            database.close_db_connection(conn)
            logger.info("Database connection closed.")

        # Shutdown async task manager if agent was initialized
        if agent and hasattr(agent, 'task_manager') and agent.task_manager:
            logger.info("Shutting down AsyncTaskManager from main...")
            agent.task_manager.shutdown()
            logger.info("AsyncTaskManager shutdown complete from main.")

        print("\nGoodbye!")

if __name__ == "__main__":
    main()
