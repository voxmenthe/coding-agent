from google import genai
from google.genai import types
import os
import sys
from pathlib import Path
from . import database
from . import tools
import traceback
import argparse
import functools
import logging
import asyncio, threading
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from prompt_toolkit.completion import WordCompleter, NestedCompleter, PathCompleter, Completer, Completion
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
import bisect # Added for efficient searching
import yaml
import sqlite3
from typing import Optional
from dotenv import load_dotenv # Import dotenv


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
        
        # # Background asyncio loop (daemon so app exits cleanly)
        # self.loop = asyncio.new_event_loop()
        # threading.Thread(target=self.loop.run_forever, daemon=True).start()
        # # Async GenAI client that lives on that loop
        # self.async_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).aio
        
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

        # Set initial thinking budget from default/config
        self.thinking_config = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        print(f"\n🧠 Initial thinking budget set to: {self.thinking_budget} tokens.")

        # Define slash commands and setup nested completer
        slash_commands = ['/reset', '/exit', '/q', '/clear', '/save', '/thinking_budget'] # Added /thinking_budget
        pdf_files = []
        if self.pdfs_dir_abs_path.is_dir():
            try:
                pdf_files = [f.name for f in self.pdfs_dir_abs_path.glob('*.pdf') if f.is_file()]
            except Exception as e:
                print(f"\n⚠️ Error listing PDF files in {self.pdfs_dir_abs_path}: {e}")
        else:
            print(f"\n⚠️ PDF directory not found: {self.pdfs_dir_abs_path}. /pdf command may not work correctly.")

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
        completer_dict['/pdf'] = WordCompleter(pdf_files, ignore_case=True)
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

                # count this turn
                self._messages_this_interval += 1
                message_to_send = user_input                 # ← default
                
                # ─── 3 · prompt injection ─────────────────────────────────────
                prompt_len = len(self.pending_prompt) if self.pending_prompt else 0
                if self.pending_prompt:
                    print("[Including loaded prompt.]\n")
                    message_to_send = f"{self.pending_prompt}\n\n{message_to_send}"
                    self.pending_prompt = None
                
                # Check for PDF context AFTER prompt (so prompt comes first)
                pdf_context_len = len(self.pending_pdf_context) if self.pending_pdf_context else 0
                if self.pending_pdf_context:
                    print("[Including context from previously processed PDF.]\n")
                    message_to_send = f"{self.pending_pdf_context}\n\n{message_to_send}"
                    self.pending_pdf_context = None        # consume exactly once
                
                
                # --- Log message details before sending --- 
                final_message_len = len(message_to_send)
                logger.info(f"Preparing to send message.")
                logger.info(f"  - Original user input length: {len(user_input)}")
                logger.info(f"  - Pending prompt length (before clear): {prompt_len}")
                logger.info(f"  - Pending PDF context length (before clear): {pdf_context_len}")
                logger.info(f"  - Final message_to_send length: {final_message_len}")
                # Log snippets for verification
                if final_message_len > 200:
                    logger.info(f"  - Final message start: {message_to_send[:100]}...")
                    logger.info(f"  - Final message end: ...{message_to_send[-100:]}")
                
                # --- Handle User Commands --- 
                # --- /save command ---
                if user_input.lower().startswith("/save"):
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
                    # Prepare state dict
                    save_state = {
                        'conversation_history': [],
                        'current_token_count': self.current_token_count,
                        'prompt_time_counts': self.prompt_time_counts,
                        'messages_per_interval': self.messages_per_interval,
                        '_messages_this_interval': self._messages_this_interval,
                        'active_files': [getattr(f, 'name', str(f)) for f in self.active_files],
                        'thinking_budget': self.thinking_budget,
                    }
                    # Convert Content objects to serializable dictionaries
                    serializable_history = []
                    for content in self.conversation_history:
                        # Ensure parts is a list of strings
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

                # --- /load command ---
                if user_input.lower().startswith("/load"):
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
                        # Restore state
                        reconstructed_history = []
                        if 'conversation_history' in load_state and isinstance(load_state['conversation_history'], list):
                            for item in load_state['conversation_history']:
                                if isinstance(item, dict) and 'role' in item and 'parts' in item and isinstance(item['parts'], list):
                                    # Recreate Parts from the list of text strings
                                    parts = [types.Part(text=part_text) for part_text in item['parts'] if isinstance(part_text, str)]
                                    # Recreate Content object
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
                        self._messages_this_interval = load_state.get('_messages_this_interval', 0)
                        self.active_files = []  # Don't restore files by default
                        self.thinking_budget = load_state.get('thinking_budget', DEFAULT_THINKING_BUDGET)
                        # Re-create chat session with loaded history
                        self.chat = self.client.chats.create(model=self.model_name, history=self.conversation_history)
                        print(f"\n📂 Loaded conversation from: {filename}")
                    except Exception as e:
                        print(f"\n❌ Failed to load conversation: {e}")
                    continue

                # --- /thinking_budget command ---
                elif user_input.lower().startswith("/thinking_budget"):
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
                    continue # Skip sending command to model

                # Check if the user input starts with '/pdf '
                if user_input.lower().startswith("/pdf"):
                    args = user_input.split()[1:]
                    self._handle_pdf_command(args)
                    continue  

                elif user_input.lower() == "/reset":
                    print("\n🎯 Resetting context and starting a new chat session...")
                    self.chat = self.client.chats.create(model=self.model_name, history=[])
                    self.conversation_history = []
                    self.current_token_count = 0
                    self.active_files = []
                    # Reset tracking lists
                    self.prompt_time_counts = [0]
                    self.messages_per_interval = [0]
                    self._messages_this_interval = 0
                    print("\n✅ Chat session and history cleared.")
                    # Decrement message counter as this command 'message' doesn't persist
                    self._messages_this_interval = 0 # Should already be 0 but reset for safety
                    continue # Skip sending this command to the model

                elif user_input.lower().startswith("/clear "):
                    # Decrement message counter as this command 'message' doesn't persist
                    self._messages_this_interval -= 1 
                    try:
                        parts = user_input.split()
                        if len(parts) != 2:
                            raise ValueError("Usage: /clear <number_of_tokens>")
                        tokens_to_clear = int(parts[1])
                        if tokens_to_clear <= 0:
                            raise ValueError("Number of tokens must be positive.")

                        if not self.chat or not self.chat.history or len(self.prompt_time_counts) <= 1:
                            print("Chat history is empty or too short to clear.")
                            continue

                        # Find the index corresponding to the interval to clear up to
                        # We look for the last count <= tokens_to_clear
                        # `bisect_right` finds the insertion point for tokens_to_clear
                        # The interval *before* this insertion point is the one we clear up to.
                        idx = bisect.bisect_right(self.prompt_time_counts, tokens_to_clear)
                        
                        # We need at least one interval before the target to clear anything
                        if idx <= 1:
                            print(f"Requested tokens ({tokens_to_clear}) is less than the first recorded interval ({self.prompt_time_counts[1]}). No messages cleared.")
                            continue
                            
                        idx_to_clear_up_to = idx - 1 # Index of the last count <= N

                        tokens_actually_cleared = self.prompt_time_counts[idx_to_clear_up_to]
                        messages_to_remove = sum(self.messages_per_interval[0:idx_to_clear_up_to + 1])

                        if messages_to_remove <= 0:
                             print("No messages to clear based on calculated intervals.")
                             continue
                             
                        if messages_to_remove >= len(self.chat.history):
                            print("Clearing entire history based on calculation.")
                            messages_to_remove = len(self.chat.history) # Avoid index error
                            # Reset fully
                            self.chat.history = []
                            self.conversation_history = []
                            self.current_token_count = 0
                            self.prompt_time_counts = [0]
                            self.messages_per_interval = [0]
                            self._messages_this_interval = 0
                        else:
                            print(f"Attempting to remove {messages_to_remove} messages, clearing up to {tokens_actually_cleared} tokens.")
                            # Slice history
                            self.chat.history = self.chat.history[messages_to_remove:]

                            # Adjust tracking lists
                            remaining_message_counts = self.messages_per_interval[idx_to_clear_up_to + 1:]
                            remaining_token_counts = self.prompt_time_counts[idx_to_clear_up_to + 1:]
                            
                            self.messages_per_interval = [0] + remaining_message_counts
                            self.prompt_time_counts = [0] + [count - tokens_actually_cleared for count in remaining_token_counts]

                            # Recalculate the current token count accurately from the remaining history
                            self.current_token_count = self.client.models.count_tokens(
                                model=self.model_name,
                                contents=self.chat.history
                            ).total_tokens
                            # Ensure the last tracked count matches the actual count
                            if self.prompt_time_counts:
                                self.prompt_time_counts[-1] = self.current_token_count
                            else: # Should not happen if history exists, but safety check
                                self.prompt_time_counts = [0]
                                self.messages_per_interval = [0]

                        print(f"\n✅ Approximately cleared {messages_to_remove} message(s) "
                              f"(up to {tokens_actually_cleared} tokens). "
                              f"New total tokens: {self.current_token_count}")

                    except Exception as e:
                        print(f"⚠️ Error processing /clear command: {e}")
                    continue # Skip sending this command to the model

                # --- /prompt command ---
                if user_input.lower().startswith("/prompt "):
                    parts = user_input.split(maxsplit=1)
                    prompt_name = parts[1] if len(parts) == 2 else ""
                    prompt_content = self._load_prompt(prompt_name)
                    if prompt_content:
                        self.pending_prompt = prompt_content
                        print(f"\n✅ Prompt '{prompt_name}' loaded. "
                              "It will be included in your next message.")
                    else:
                        print(f"\n❌ Prompt '{prompt_name}' not found.")
                    continue                           # <‑‑ early return # Skip sending command to model

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
        """Handles the /pdf command to process a PDF file and save metadata/text.

        Returns:
            Optional[int]: The paper_id if processing is successful, otherwise None.
        """
        if not self.pdfs_dir_abs_path:
            print("\n⚠️ PDF directory not configured. Cannot process PDFs.")
            return None
        
        # Ensure blob_dir is a Path object (already resolved in __init__)
        blob_dir = self.blob_dir 

        # Use the connection established during initialization
        if not self.conn:
             print("\n⚠️ Database connection not available. Cannot save PDF metadata.")
             return None
        if not blob_dir:
             print("\n⚠️ Blob directory not configured. Cannot save extracted text.")
             return None
        
        # Gemini client check - ensure it's ready if needed
        if self.pdf_processing_method == 'Gemini' and not self.client:
            print("\n⚠️ Gemini client not initialized, but required for Gemini processing. Cannot process PDF.")
            return None

        if len(args) < 1:
            print("\n⚠️ Usage: /pdf <filename> [optional: arxiv_id]")
            # Consider adding a call to list available PDFs here if useful
            # self._list_available_pdfs() # If such a method exists
            return None

        filename_arg = args[0]
        # Basic security: Ensure filename doesn't contain path separators
        filename = Path(filename_arg).name
        if filename != filename_arg:
             print(f"\n⚠️ Invalid filename '{filename_arg}'. Please provide only the filename, not a path.")
             return None

        pdf_path = self.pdfs_dir_abs_path / filename

        if not pdf_path.exists() or not pdf_path.is_file():
            print(f"\n⚠️ Error: PDF file '{filename}' not found in {self.pdfs_dir_abs_path}.")
            # self._list_available_pdfs() # If exists
            return None

        arxiv_id_arg = args[1] if len(args) > 1 else None

        print(f"\n⏳ Processing PDF: {filename}...")

        paper_id = None
        # Remove the local try/except for connection, use self.conn directly
        try:
            # --- Database Operations using self.conn --- 
            paper_id = database.add_minimal_paper(self.conn, filename)
            if not paper_id:
                print("\n⚠️ Error: Failed to create initial database record.")
                # No connection to close here
                return None 

            print(f"  📄 Created database record with ID: {paper_id}")

            if arxiv_id_arg:
                if isinstance(arxiv_id_arg, str) and len(arxiv_id_arg) > 5: 
                    database.update_paper_field(self.conn, paper_id, 'arxiv_id', arxiv_id_arg)
                    print(f"     Updated record with provided arXiv ID: {arxiv_id_arg}")
                else:
                    print(f"  ⚠️ Warning: Provided arXiv ID '{arxiv_id_arg}' seems invalid. Skipping update.")
            # --- End Database Operations ---

            print(f"  ⚙️  Extracting text from '{filename}' (using {self.pdf_processing_method})...")
            extracted_text = None
            try:
                # TODO: Implement fallback to pypdf if configured or if gemini fails?
                if self.pdf_processing_method == 'Gemini':
                     extracted_text = tools.extract_text_from_pdf_gemini(pdf_path, self.client, self.model_name)
                # Add other methods like pypdf here if needed
                # elif self.pdf_processing_method == 'pypdf':
                #    extracted_text = tools.extract_text_from_pdf_pypdf(pdf_path) # Assuming this exists
                else:
                    raise ValueError(f"Unsupported pdf_processing_method: {self.pdf_processing_method}")
                
                if not extracted_text: # Check if extraction failed
                    raise ValueError("Text extraction resulted in empty or failed content.")

            except Exception as extract_err:
                print(f"\n⚠️ Error during text extraction ({self.pdf_processing_method}): {extract_err}")
                database.update_paper_field(self.conn, paper_id, 'status', 'error_process')
                # No conn.close() needed here
                return None 

            print(f"  💬 Successfully extracted text ({len(extracted_text)} chars) [{self.pdf_processing_method}].")

            blob_filename = f"paper_{paper_id}_text.txt"
            blob_full_path = blob_dir / blob_filename
            blob_rel_path = blob_filename # Store relative path in DB

            print(f"  💾 Saving extracted text to {blob_full_path}...")
            save_success = tools.save_text_blob(blob_full_path, extracted_text)

            if not save_success:
                print("\n⚠️ Error: Failed to save text blob.")
                database.update_paper_field(self.conn, paper_id, 'status', 'error_blob')
                # No conn.close() needed here
                return None
            else:
                print(f"     Successfully saved text blob.")
                update_success = database.update_paper_field(self.conn, paper_id, 'blob_path', blob_rel_path)
                if not update_success:
                    print(f"  ⚠️ Warning: Failed to update blob_path in database for ID {paper_id}.")
                    # Log and continue, as text is saved locally

            update_success = database.update_paper_field(self.conn, paper_id, 'status', 'processed') # Mark as processed before adding to history
            if not update_success:
                 logger.warning(f"Failed to update status to processed for paper ID {paper_id}")
                 # Continue anyway, try to add to history

            # Store context to be prepended next turn
            context_header = f"CONTEXT FROM PDF ('{filename}', ID: {paper_id}):\n---"
            if MAX_PDF_CONTEXT_LENGTH:
                max_text_len = MAX_PDF_CONTEXT_LENGTH - len(context_header) - 50 # Reserve space for header and separators
                truncated_text = extracted_text[:max_text_len]
                if len(extracted_text) > max_text_len:
                    truncated_text += "\n... [TRUNCATED]" 
            else:
                truncated_text = extracted_text
            
            self.pending_pdf_context = f"{context_header}\n{truncated_text}\n---"
            logger.info(f"Stored context from {filename} (ID: {paper_id}) to be prepended next turn ({len(self.pending_pdf_context)} chars).")
            print(f"\n✅ PDF processed. Content will be added to context on your next message. (ID: {paper_id}) ")
            database.update_paper_field(self.conn, paper_id, 'status', 'processed_pending_context')

            return paper_id
        except Exception as e: # Reinstate the main exception handler for the function's try block
            logger.error(f"An error occurred during PDF processing for '{filename}': {e}", exc_info=True)
            if paper_id and self.conn:
                try:
                    # Attempt to mark as error if possible
                    database.update_paper_field(self.conn, paper_id, 'status', 'error_process')
                except Exception as db_err:
                    logger.error(f"Additionally failed to update status to error for paper {paper_id}: {db_err}")
            # No conn.close() needed here
            print(f"\n❌ An unexpected error occurred: {e}")
            return None # Return None on major processing error

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
