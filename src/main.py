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
from prompt_toolkit.completion import WordCompleter, NestedCompleter, PathCompleter, Completer, Completion
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
import bisect # Added for efficient searching
import yaml
import sqlite3
from typing import Optional
from dotenv import load_dotenv # Import dotenv
from datetime import datetime, timezone # Added for processed_timestamp
import sys # For sys.executable
import subprocess # For running scripts
# from google.generativeai import types as genai_types # This was incorrect for the new SDK as the `google.generativeai` SDK is deprecated and we should use `google.genai` SDK instead


# Setup basic logging
# TODO: Configure logging more robustly (e.g., level, format, handler) if needed
logging.basicConfig(level=logging.INFO) # Basic config for now
logger = logging.getLogger(__name__) # Define module-level logger

# Choose your Gemini model - unless you want something crazy "gemini-2.5-flash-preview-04-17" is the default model
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
DEFAULT_THINKING_BUDGET = 256
MAX_PDF_CONTEXT_LENGTH = None # Max chars from PDF to prepend - TODO: get this from config.yaml

# Default configuration values
DEFAULT_CONFIG = {
    'gemini_api_key': None,
    'model_name': 'gemini-2.5-flash-preview-04-17',
    'verbose': False,
    'default_thinking_budget': 256,
    'PDFS_TO_CHAT_WITH_DIRECTORY': 'PDFS/',
    'SAVED_CONVERSATIONS_DIRECTORY': 'SAVED_CONVERSATIONS/',
    'PAPER_DB_PATH': 'paper_database.db',
    'PAPER_BLOBS_DIR': 'paper_blobs/'
}

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

    # 3. Load base configuration from YAML
    config = DEFAULT_CONFIG.copy()
    yaml_data = {}
    if config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f) or {}
                config.update(yaml_data)
                logger.info(f"Loaded configuration overrides from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}", exc_info=True)
            # Continue with defaults and environment variables
    else:
         logger.info(f"Config file {config_path} not found. Using defaults and environment variables.")

    # 4. Prioritize environment variable for API key
    if env_api_key:
        config['gemini_api_key'] = env_api_key
        logger.info("Using GEMINI_API_KEY from environment.")
    elif 'gemini_api_key' in yaml_data and yaml_data['gemini_api_key']:
        logger.info("Using gemini_api_key from config.yaml (environment variable not set).")
        # It's already in config from the update() step
    else:
        logger.warning("GEMINI_API_KEY not found in environment or config.yaml.")
        # config['gemini_api_key'] will remain None from DEFAULT_CONFIG

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
        self.model_name = config.get('model_name', 'gemini-2.5-flash-preview-04-17') # Default model
        self.pdf_processing_method = config.get('pdf_processing_method', 'Gemini') # Default method
        self.client = None
        self.chat = None
        self.db_path_str = str(config.get('PAPER_DB_PATH')) if config.get('PAPER_DB_PATH') else None # Store DB path string
        
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
            tools.get_current_date_and_time,
            tools.google_search,
            tools.open_url
        ]
        if self.config.get('verbose', DEFAULT_CONFIG['verbose']):
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
        self.pending_prompt: Optional[str] = None # For prepending loaded prompt
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

    def start_interaction(self):
        """Starts the main interaction loop using a stateful ChatSession via client.chats.create."""
        if not self.client:
            print("\n\u274c Client not configured. Exiting.")
            return

        print("\n\u2692\ufe0f Agent ready. Ask me anything. Type '/exit' or '/q' to quit.")
        print("   Use '/pdf <filename>' to seed PDF into context from the specified directory.")
        print("   Use '/reset' to clear the chat and start fresh.")
        print("   Use '/clear <n_tokens>' to remove <tokens> from the start of history.")
        print("   Use '/save <optional_filename>' to save the current conversation.")
        print("   Use '/load <filename>' to load a saved conversation.")
        print(f"   Use '/thinking_budget <value>' to set tool thinking budget (current: {self.thinking_budget}).") # Updated help
        print(f"   Use '/cancel <task_id>' to attempt to cancel a background task.")
        print(f"   Use '/tasks' to list active background tasks.")
        print(f"   Use '/run_script <python|shell> <script_path> [args...]' to run a script.")

        # Set initial thinking budget from default/config
        self.thinking_config = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        print(f"\n🧠 Initial thinking budget set to: {self.thinking_budget} tokens.")

        # Define slash commands and setup nested completer
        slash_commands = ['/reset', '/exit', '/q', '/clear', '/save', '/thinking_budget', '/cancel', '/tasks', '/run_script']
        # pdf_files = [] # <-- Remove old pdf_files list creation
        # if self.pdfs_dir_abs_path.is_dir():
        #     try:
        #         pdf_files = [f.name for f in self.pdfs_dir_abs_path.glob('*.pdf') if f.is_file()]
        #     except Exception as e:
        #         print(f"\n⚠️ Error listing PDF files in {self.pdfs_dir_abs_path}: {e}")
        # else:
        #     print(f"\n⚠️ PDF directory not found: {self.pdfs_dir_abs_path}. /pdf command may not work correctly.")

        # List saved conversations for /load autocomplete
        saved_conversations_dir_path = self.config.get('SAVED_CONVERSATIONS_DIRECTORY')
        saved_files = []
        if saved_conversations_dir_path and isinstance(saved_conversations_dir_path, Path) and saved_conversations_dir_path.is_dir():
             saved_files = [f.name for f in saved_conversations_dir_path.glob('*.json') if f.is_file()]
        elif isinstance(saved_conversations_dir_path, str): # Fallback if somehow it's still a string
             # This case should ideally not happen due to load_config resolving paths
             logger.warning("SAVED_CONVERSATIONS_DIRECTORY was a string, expected Path. Attempting to resolve.")
             try:
                  resolved_path = Path(__file__).parent.parent / saved_conversations_dir_path
                  if resolved_path.is_dir():
                       saved_files = [f.name for f in resolved_path.glob('*.json') if f.is_file()]
             except Exception as e:
                  logger.error(f"Error resolving string path for saved conversations: {e}")
        else: 
             # Use default relative path if config value is missing or invalid type
             default_save_dir = Path(__file__).parent.parent / 'SAVED_CONVERSATIONS/'
             if default_save_dir.is_dir():
                  saved_files = [f.name for f in default_save_dir.glob('*.json') if f.is_file()]

        # Nested completer for commands and their potential arguments (like PDF files)
        completer_dict = {cmd: None for cmd in slash_commands}
        # completer_dict['/pdf'] = WordCompleter(pdf_files, ignore_case=True) # <-- Comment out or remove old completer
        completer_dict['/pdf'] = PdfCompleter(self)  # <-- Use PdfCompleter
        completer_dict['/load'] = WordCompleter(saved_files, ignore_case=True)
        completer_dict['/prompt'] = WordCompleter(self._list_available_prompts(), ignore_case=True)

        command_completer = NestedCompleter.from_nested_dict(completer_dict)

        history = InMemoryHistory()
        session = PromptSession(">", completer=command_completer, history=history)

        while True:
            try:
                # ─── 1 · house‑keeping before we prompt ──────────────────────────
                self.prompt_time_counts.append(self.current_token_count)
                self.messages_per_interval.append(self._messages_this_interval)
                self._messages_this_interval = 0

                active_files_info = f" [{len(self.active_files)} files]" if self.active_files else ""
                prompt_text = f"\n🔵 You ({self.current_token_count}{active_files_info}): "
                user_input = session.prompt(prompt_text).strip()

                # ─── 2 · trivial exits / empty line ─────────────────────────────
                if user_input.lower() in {"exit", "quit", "/exit", "/quit", "/q"}:
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    # unwind the stats we pushed at the top of the loop
                    self.prompt_time_counts.pop()
                    self.messages_per_interval.pop()
                    continue

                # --- Handle User Commands FIRST --- 
                # (This is important so local commands don't accidentally consume contexts
                # that were meant for an LLM call that happens later)
                if user_input.lower().startswith("/pdf"): # Already handles its own continue
                    args = user_input.split()[1:]
                    self._handle_pdf_command(args)
                    # self._messages_this_interval is not incremented for this local command
                    continue  

                elif user_input.lower().startswith("/prompt "):
                    parts = user_input.split(maxsplit=1)
                    prompt_name = parts[1] if len(parts) == 2 else ""
                    prompt_content = self._load_prompt(prompt_name)
                    if prompt_content:
                        self.pending_prompt = prompt_content
                        print(f"\n✅ Prompt '{prompt_name}' loaded. "
                              "It will be included in your next message to the LLM.")
                    else:
                        print(f"\n❌ Prompt '{prompt_name}' not found.")
                    # self._messages_this_interval is not incremented
                    continue

                elif user_input.lower().startswith("/save"):
                    # ... (existing /save logic) ... 
                    # Ensure it continues and does not increment _messages_this_interval
                    import json
                    import datetime
                    parts = user_input.split()
                    if len(parts) > 1:
                        filename = parts[1]
                        if not filename.endswith('.json'):
                            filename += '.json'
                    else:
                            now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
                            filename = f'{now}.json'

                    save_path = Path(__file__).parent.parent / self.config.get('SAVED_CONVERSATIONS_DIRECTORY', 'SAVED_CONVERSATIONS/') / filename
                    save_state = {
                        'conversation_history': [],
                        'current_token_count': self.current_token_count,
                        'prompt_time_counts': self.prompt_time_counts,
                        'messages_per_interval': self.messages_per_interval,
                        '_messages_this_interval': self._messages_this_interval, # Save current state before potential LLM call
                        'active_files': [getattr(f, 'name', str(f)) for f in self.active_files],
                        'thinking_budget': self.thinking_budget,
                        'pending_prompt': self.pending_prompt, # Save pending states
                        'pending_pdf_context': self.pending_pdf_context
                    }
                    serializable_history = []
                    for content in self.conversation_history:
                        parts_text = [part.text for part in content.parts if hasattr(part, 'text') and part.text is not None]
                        serializable_history.append({
                            'role': content.role,
                            'parts': parts_text
                        })
                    save_state['conversation_history'] = serializable_history
                    try:
                        with open(save_path, 'w') as f:
                            json.dump(save_state, f, indent=2)
                        print(f"\n💾 Conversation saved as: {filename}")
                    except Exception as e:
                        print(f"\n❌ Failed to save conversation: {e}")
                    continue

                elif user_input.lower().startswith("/load"):
                    # ... (existing /load logic) ...
                    # Ensure it continues and does not increment _messages_this_interval
                    import json
                    parts = user_input.split()
                    if len(parts) > 1:
                        filename = parts[1]
                    else:
                        print("\n⚠️ Usage: /load <filename>")
                        continue
                    load_path = Path(__file__).parent.parent / self.config.get('SAVED_CONVERSATIONS_DIRECTORY', 'SAVED_CONVERSATIONS/') / filename
                    if not load_path.is_file():
                        print(f"\n❌ File not found: {filename}")
                        continue
                    try:
                        with open(load_path, 'r') as f:
                            load_state = json.load(f)
                        reconstructed_history = []
                        if 'conversation_history' in load_state and isinstance(load_state['conversation_history'], list):
                            for item in load_state['conversation_history']:
                                if isinstance(item, dict) and 'role' in item and 'parts' in item and isinstance(item['parts'], list):
                                    parts = [types.Part(text=part_text) for part_text in item['parts'] if isinstance(part_text, str)]
                                    content = types.Content(role=item['role'], parts=parts)
                                    reconstructed_history.append(content)
                                else:
                                    logger.warning(f"Skipping invalid item in loaded history: {item}")
                        else:
                             logger.warning(f"'conversation_history' key missing or not a list in {filename}")

                        self.conversation_history = reconstructed_history
                        self.current_token_count = load_state.get('current_token_count', 0)
                        self.prompt_time_counts = load_state.get('prompt_time_counts', [0])
                        self.messages_per_interval = load_state.get('messages_per_interval', [0])
                        self._messages_this_interval = load_state.get('_messages_this_interval', 0) # Restore this carefully
                        self.active_files = []  # Don't restore files by default
                        self.thinking_budget = load_state.get('thinking_budget', DEFAULT_THINKING_BUDGET)
                        self.pending_prompt = load_state.get('pending_prompt') # Restore pending states
                        self.pending_pdf_context = load_state.get('pending_pdf_context')
                        self.chat = self.client.chats.create(model=self.model_name, history=self.conversation_history)
                        print(f"\n📂 Loaded conversation from: {filename}")
                        if self.pending_prompt: print("   Loaded pending prompt is active.")
                        if self.pending_pdf_context: print("   Loaded pending PDF context is active.")
                    except Exception as e:
                        print(f"\n❌ Failed to load conversation: {e}")
                    continue

                elif user_input.lower().startswith("/thinking_budget"):
                    # ... (existing /thinking_budget logic) ...
                    parts = user_input.split()
                    if len(parts) == 2:
                        try:
                            new_budget = int(parts[1])
                            if 0 <= new_budget <= 24000: # Example range validation
                                self.thinking_budget = new_budget
                                self.thinking_config = types.ThinkingConfig(thinking_budget=self.thinking_budget)
                                print(f"\n🧠 Thinking budget updated to: {self.thinking_budget} tokens.")
                            else:
                                print("\n⚠️ Thinking budget must be between 0 and 24000.")
                        except ValueError:
                            print("\n⚠️ Invalid number format for thinking budget.")
                    else:
                        print("\n⚠️ Usage: /thinking_budget <number_of_tokens>")
                    continue

                elif user_input.lower() == "/reset":
                    # ... (existing /reset logic) ...
                    print("\n🎯 Resetting context and starting a new chat session...")
                    self.chat = self.client.chats.create(model=self.model_name, history=[])
                    self.conversation_history = []
                    self.current_token_count = 0
                    self.active_files = []
                    self.prompt_time_counts = [0]
                    self.messages_per_interval = [0]
                    self._messages_this_interval = 0 
                    self.pending_prompt = None # Clear pending states on reset
                    self.pending_pdf_context = None
                    print("\n✅ Chat session and history cleared.")
                    continue

                elif user_input.lower().startswith("/clear "):
                    # ... (existing /clear logic) ...
                    try:
                        parts = user_input.split()
                        if len(parts) != 2:
                            raise ValueError("Usage: /clear <number_of_tokens>")
                        tokens_to_clear_target = int(parts[1])
                        if tokens_to_clear_target <= 0:
                            raise ValueError("Number of tokens must be positive.")

                        if not self.conversation_history:
                            print("Chat history is already empty.")
                            continue

                        logger.info(f"Attempting to clear approx. {tokens_to_clear_target} tokens from history.")
                        
                        new_history = list(self.conversation_history)
                        tokens_counted_for_removal = 0
                        messages_removed_count = 0

                        while tokens_counted_for_removal < tokens_to_clear_target and new_history:
                            first_message = new_history[0]
                            try:
                                message_tokens = self.client.models.count_tokens(model=self.model_name, contents=[first_message]).total_tokens
                            except Exception as e_count:
                                logger.error(f"Could not count tokens for a message during /clear: {e_count}. Skipping message token count.")
                                message_tokens = 75 # Arbitrary average, or could stop clear
                            
                            tokens_counted_for_removal += message_tokens
                            new_history.pop(0)
                            messages_removed_count += 1
                        
                        logger.info(f"After initial pass, {messages_removed_count} messages ({tokens_counted_for_removal} tokens) selected for removal.")
                        logger.info(f"Remaining history length before role check: {len(new_history)}")

                        # Ensure remaining history starts with a user turn
                        if new_history and new_history[0].role != "user":
                            logger.warning("History after initial clear pass starts with a model turn. Removing additional leading model messages.")
                            additional_messages_removed_for_role = 0
                            while new_history and new_history[0].role != "user":
                                new_history.pop(0)
                                additional_messages_removed_for_role += 1
                            messages_removed_count += additional_messages_removed_for_role
                            logger.info(f"Removed {additional_messages_removed_for_role} additional messages to ensure user turn start.")
                        
                        if not new_history and not self.conversation_history: # No change if already empty
                             print("Chat history is already empty. No action taken.")
                             continue
                        elif len(new_history) == len(self.conversation_history) and messages_removed_count == 0: # No messages were actually removed
                            print(f"No messages were cleared. Requested {tokens_to_clear_target} tokens might be less than the first message(s) or history is too short.")
                            continue

                        self.conversation_history = new_history
                        history_was_modified = True # Assume modification if we got this far and something changed.

                        # Recalculate all tracking information based on the new history
                        if not self.conversation_history:
                            self.current_token_count = 0
                            self.prompt_time_counts = [0]
                            self.messages_per_interval = [0]
                        else:
                            self.current_token_count = self.client.models.count_tokens(
                                model=self.model_name,
                                contents=self.conversation_history
                            ).total_tokens
                            # Simplified tracking reset for prompt_time_counts and messages_per_interval
                            self.prompt_time_counts = [0, self.current_token_count] 
                            self.messages_per_interval = [0, len(self.conversation_history)]
                        
                        self._messages_this_interval = 0 # Reset for the current interval

                        if history_was_modified:
                            try:
                                self.chat = self.client.chats.create(model=self.model_name, history=self.conversation_history)
                                logger.info("Chat session re-initialized after /clear operation.")
                            except Exception as e_chat_reinit: # More general catch, includes ValueError
                                logger.error(f"Error re-initializing chat after /clear: {e_chat_reinit}", exc_info=True)
                                print(f"\n⚠️ Error re-initializing chat session: {e_chat_reinit}.")
                                # If re-init fails, history might be inconsistent with self.chat. 
                                # Forcing full clear as a fallback might be too drastic if the count was just off.
                                # For now, we alert and the history is what it is.

                            print(f"\n✅ Cleared {messages_removed_count} message(s) (approx. {tokens_counted_for_removal} tokens counted for removal). "
                                  f"New total tokens: {self.current_token_count}")
                        
                    except Exception as e:
                        print(f"⚠️ Error processing /clear command: {e}")
                        logger.error("Error in /clear command", exc_info=True)
                    continue

                elif user_input.lower().startswith("/run_script "):
                    args = user_input.split()
                    if len(args) >= 3:
                        script_type = args[1].lower()
                        script_path = args[2]
                        script_arguments = args[3:]
                        if script_type in ["python", "shell"]:
                            self._handle_run_script_command(script_type, script_path, script_arguments)
                        else:
                            print("\n⚠️ Invalid script type. Must be 'python' or 'shell'. Usage: /run_script <python|shell> <script_path> [args...]")
                    else:
                        print("\n⚠️ Usage: /run_script <python|shell> <script_path> [args...]")
                    continue

                elif user_input.lower().startswith("/cancel "):
                    args = user_input.split(maxsplit=1)
                    if len(args) > 1:
                        self._handle_cancel_command(args[1])
                    else:
                        print("\n⚠️ Usage: /cancel <task_id>")
                    continue

                elif user_input.lower() == "/tasks":
                    self._handle_list_tasks_command()
                    continue

                # If we haven't 'continued' from a local command, it's a message for the LLM.
                self._messages_this_interval += 1 # Count this turn as an LLM message
                message_to_send = user_input # Default

                # Flags to track if contexts were used in this specific message
                prompt_was_included = False
                pdf_context_was_included = False
                script_output_was_included = False # New flag
                 
                # ─── 3 · prompt injection ─────────────────────────────────────
                if self.pending_prompt:
                    print("[Including previously loaded prompt in this message.]\n")
                    message_to_send = f"{self.pending_prompt}\n\n{message_to_send}"
                    prompt_was_included = True
                
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
                prompt_len_before_send = len(self.pending_prompt) if self.pending_prompt and prompt_was_included else 0
                pdf_context_len_before_send = len(self.pending_pdf_context) if self.pending_pdf_context and pdf_context_was_included else 0
                script_output_len_before_send = len(self.pending_script_output) if self.pending_script_output and script_output_was_included else 0
                final_message_len = len(message_to_send)
                logger.info(f"Preparing to send message.")
                logger.info(f"  - Original user input length: {len(user_input)}")
                logger.info(f"  - Included pending prompt length: {prompt_len_before_send}")
                logger.info(f"  - Included pending PDF context length: {pdf_context_len_before_send}")
                logger.info(f"  - Included pending script output length: {script_output_len_before_send}")
                logger.info(f"  - Final message_to_send length: {final_message_len}")
                # Log snippets for verification
                if final_message_len > 200:
                    logger.info(f"  - Final message start: {message_to_send[:100]}...")
                    logger.info(f"  - Final message end: ...{message_to_send[-100:]}")
                
                # --- Handle User Commands --- 
                # --- /save command ---
                # --- /load command ---
                # --- /thinking_budget command ---
                # --- /pdf command ---
                # --- /reset command ---
                # --- /clear command ---
                # --- /prompt command ---

                # --- Prepare message content (Text + Files) ---
                message_content = [message_to_send]
                if self.active_files:
                    message_content.extend(self.active_files)
                    if self.config.get('verbose', DEFAULT_CONFIG['verbose']):
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
                if prompt_was_included:
                    self.pending_prompt = None
                    logger.info("Cleared pending_prompt after sending to LLM.")
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
        """Handles the /pdf command to asynchronously process a PDF file."""
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

        if len(args) < 1:
            print("\n⚠️ Usage: /pdf <filename> [optional: arxiv_id]")
            return

        filename_arg = args[0]
        # Basic security: Ensure filename doesn't contain path separators
        filename = Path(filename_arg).name
        if filename != filename_arg:
             print(f"\n⚠️ Invalid filename '{filename_arg}'. Please provide only the filename, not a path.")
             return

        pdf_path = self.pdfs_dir_abs_path / filename

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
