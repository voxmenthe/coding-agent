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
from prompt_toolkit.completion import WordCompleter, NestedCompleter, PathCompleter, Completer, Completion
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
import bisect # Added for efficient searching
import yaml

# Setup basic logging
# TODO: Configure logging more robustly (e.g., level, format, handler) if needed
logging.basicConfig(level=logging.INFO) # Basic config for now
logger = logging.getLogger(__name__) # Define module-level logger

# Choose your Gemini model - unless you want something crazy "gemini-2.5-flash-preview-04-17" is the default model
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
DEFAULT_THINKING_BUDGET = 256

# Default configuration values
DEFAULT_CONFIG = {
    'api_key': None,
    'model_name': 'gemini-2.5-flash-preview-04-17', # Defaulting to Haiku
    'verbose': False,
    'default_thinking_budget': 256,
    'PDFS_TO_CHAT_WITH_DIRECTORY': 'PDFS/',
    'SAVED_CONVERSATIONS_DIRECTORY': 'SAVED_CONVERSATIONS/',
    'PAPER_DB_PATH': 'paper_database.db',
    'PAPER_BLOBS_DIR': 'paper_blobs/'
}

# --- Utility Functions ---
def load_config():
    config = DEFAULT_CONFIG.copy()
    project_root = Path(__file__).parent.parent # Assuming script is in src/
    config_path = project_root / 'src/config.yaml'
    
    if config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                config.update(yaml.safe_load(f))
            # Resolve relative paths to absolute paths based on project root
            if 'PDFS_TO_CHAT_WITH_DIRECTORY' in config:
                config['PDFS_TO_CHAT_WITH_DIRECTORY'] = project_root / config['PDFS_TO_CHAT_WITH_DIRECTORY']
                config['PDFS_TO_CHAT_WITH_DIRECTORY'].mkdir(parents=True, exist_ok=True)
            if 'SAVED_CONVERSATIONS_DIRECTORY' in config:
                config['SAVED_CONVERSATIONS_DIRECTORY'] = project_root / config['SAVED_CONVERSATIONS_DIRECTORY']
                config['SAVED_CONVERSATIONS_DIRECTORY'].mkdir(parents=True, exist_ok=True)
            if 'PAPER_DB_PATH' in config:
                config['PAPER_DB_PATH'] = project_root / config['PAPER_DB_PATH']
                # Directory creation handled by get_db_connection
            if 'PAPER_BLOBS_DIR' in config:
                config['PAPER_BLOBS_DIR'] = project_root / config['PAPER_BLOBS_DIR']
                config['PAPER_BLOBS_DIR'].mkdir(parents=True, exist_ok=True) # Ensure blob dir exists
        except Exception as e:
            print(f"Error loading config: {e}")
    return config

# --- Agent Class ---
class CodeAgent:
    """A simple coding agent using Google Gemini (google-genai SDK)."""

    def __init__(self, model_name: str, verbose: bool, api_key: str, default_thinking_budget: int, pdf_dir: str, db_path: str, blob_dir: str):
        """Initializes the agent with API key and model name."""
        self.model_name = model_name
        self.verbose = verbose
        self.api_key = api_key
        self.model_name = f'models/{model_name}' # Add 'models/' prefix
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
        if self.verbose:
            self.tool_functions = [self._make_verbose_tool(f) for f in self.tool_functions]
        self.client = None
        self.chat = None
        self.conversation_history = [] # Manual history for token counting ONLY
        self.current_token_count = 0 # Store token count for the next prompt
        self.active_files = [] # List to store active File objects
        self.prompt_time_counts = [0] # Stores total tokens just before prompt
        self.messages_per_interval = [0] # Stores # messages added in the last interval
        self._messages_this_interval = 0 # Temporary counter
        self.thinking_budget = default_thinking_budget
        self.thinking_config = None
        self.pdfs_dir_abs_path = Path(pdf_dir).resolve() # Store absolute path to PDF dir
        self.db_path = db_path
        self.blob_dir = blob_dir
        # Load saved conversations dir from config
        config = load_config()
        project_root = Path(__file__).parent.parent
        self.saved_conversations_dir = (project_root / config.get('SAVED_CONVERSATIONS_DIRECTORY', 'SAVED_CONVERSATIONS/')).resolve()
        self.saved_conversations_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self._configure_client()

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

    def start_interaction(self):
        """Starts the main interaction loop using a stateful ChatSession via client.chats.create."""
        if not self.client:
            print("\n\u274c Client not configured. Exiting.")
            return

        print("\n\u2692\ufe0f Initializing chat session...")
        try:
            # Create a chat session using the client
            self.chat = self.client.chats.create(model=self.model_name, history=[])
            print("\u2705 Chat session initialized.")
        except Exception as e:
            print(f"\u274c Error initializing chat session: {e}")
            traceback.print_exc()
            sys.exit(1)

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
        saved_files = [f.name for f in self.saved_conversations_dir.glob('*.json') if f.is_file()]

        # Nested completer for commands and their potential arguments (like PDF files)
        completer_dict = {cmd: None for cmd in slash_commands}
        completer_dict['/pdf'] = WordCompleter(pdf_files, ignore_case=True)
        completer_dict['/load'] = WordCompleter(saved_files, ignore_case=True)

        command_completer = NestedCompleter.from_nested_dict(completer_dict)

        history = InMemoryHistory()
        session = PromptSession(">", completer=command_completer, history=history)

        while True:
            try:
                # --- Update Tracking Lists Before Prompt --- 
                # Store the current total count and the number of messages added since last prompt
                self.prompt_time_counts.append(self.current_token_count)
                self.messages_per_interval.append(self._messages_this_interval)
                self._messages_this_interval = 0 # Reset counter for next interval
                # --- End Update --- 

                # Display token count from *previous* turn in the prompt
                # Also show number of active files
                active_files_info = f" [{len(self.active_files)} files]" if self.active_files else ""
                # Use the latest calculated total for the prompt display
                prompt_text = f"\n🔵 You ({self.current_token_count}{active_files_info}): "
                user_input = session.prompt(prompt_text).strip()

                if user_input.lower() in ["exit", "quit", "/exit", "/quit", "/q"]:
                    print("\n👋 Goodbye!")
                    break
                if not user_input:
                    # No input, reset interval message counter as no messages were added
                    self._messages_this_interval = 0 
                    # Remove the entries added at the start of this loop iteration
                    if len(self.prompt_time_counts) > 1:
                        self.prompt_time_counts.pop()
                        self.messages_per_interval.pop()
                    continue

                # Increment message counter for user input
                self._messages_this_interval += 1 

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
                    save_path = self.saved_conversations_dir / filename
                    # Prepare state dict
                    save_state = {
                        'conversation_history': [c.to_dict() if hasattr(c, 'to_dict') else str(c) for c in self.conversation_history],
                        'current_token_count': self.current_token_count,
                        'prompt_time_counts': self.prompt_time_counts,
                        'messages_per_interval': self.messages_per_interval,
                        '_messages_this_interval': self._messages_this_interval,
                        'active_files': [getattr(f, 'name', str(f)) for f in self.active_files],
                        'thinking_budget': self.thinking_budget,
                    }
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
                    load_path = self.saved_conversations_dir / filename
                    if not load_path.is_file():
                        print(f"\n❌ File not found: {filename}")
                        continue
                    try:
                        with open(load_path, 'r') as f:
                            load_state = json.load(f)
                        # Restore state
                        self.conversation_history = load_state.get('conversation_history', [])
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

                # --- Refactored PDF command handling ---
                # Check if the user input starts with '/pdf '
                elif user_input.lower().startswith("/pdf "):
                    # The user's command itself was already counted by self._messages_this_interval += 1
                    # Extract arguments (everything after the first space)
                    command_args = user_input.split()[1:] 
                    # Call the dedicated handler method with the arguments
                    self._handle_pdf_command(command_args)
                    # Skip sending the '/pdf ...' command string to the model
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

                        print(f"✅ Approximately cleared {messages_to_remove} message(s) (up to {tokens_actually_cleared} tokens). New total tokens: {self.current_token_count}")

                    except ValueError as e:
                        print(f"⚠️ Error processing /clear command: {e}")
                    except Exception as e:
                        print(f"🚨 An unexpected error occurred during /clear: {e}")
                    continue # Skip sending this command to the model

                # --- Prepare message content (Text + Files) ---
                message_content = [user_input] # Start with user text
                if self.active_files:
                    message_content.extend(self.active_files) # Add file objects
                    if self.verbose:
                        print(f"\n📎 Attaching {len(self.active_files)} files to the prompt:")
                        for f in self.active_files:
                            print(f"   - {f.display_name} ({f.name})")

                # --- Update manual history (for token counting ONLY - Use Text Only) --- 
                # Add user message BEFORE sending to model
                # Store only the text part for history counting simplicity
                new_user_content = types.Content(parts=[types.Part(text=user_input)], role="user")
                self.conversation_history.append(new_user_content)

                # --- Send Message --- 
                print("\n⏳ Sending message and processing...")
                # Prepare tool configuration **inside the loop** to use the latest budget
                tool_config = types.GenerateContentConfig(tools=self.tool_functions, thinking_config=self.thinking_config)

                # Send message using the chat object's send_message method
                # Pass the potentially combined list of text and files
                response = self.chat.send_message(
                    message=message_content, # Pass the list here
                    config=tool_config
                )

                # --- Update manual history and calculate new token count AFTER response --- 
                agent_response_content = None
                response_text = "" # Initialize empty response text
                if response.candidates and response.candidates[0].content:
                    agent_response_content = response.candidates[0].content
                    # Ensure we extract text even if other parts exist (e.g., tool calls)
                    if agent_response_content.parts:
                         # Simple concatenation of text parts for history
                         response_text = " ".join(p.text for p in agent_response_content.parts if hasattr(p, 'text'))
                    self.conversation_history.append(agent_response_content)
                else:
                    print("\n⚠️ Agent response did not contain content for history/counting.")

                # Print agent's response text to user
                # Use the extracted response_text or response.text as fallback
                print(f"\n🟢 \x1b[92mAgent:\x1b[0m {response_text or response.text}")

                # Calculate and store token count for the *next* prompt
                try:
                    # Get token count via the models endpoint
                    token_count_response = self.client.models.count_tokens(
                        model=self.model_name,
                        contents=self.chat.history
                    )
                    self.current_token_count = token_count_response.total_tokens
                except Exception as count_error:
                    # Don't block interaction if counting fails, just report it and keep old count
                    print(f"\n⚠️ \x1b[93mCould not update token count: {count_error}\x1b[0m")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n🔴 \x1b[91mAn error occurred during interaction: {e}\x1b[0m")
                traceback.print_exc() # Print traceback for debugging

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
        # Ensure db_path and blob_dir are Path objects if loaded from config
        db_path = self.db_path if isinstance(self.db_path, Path) else Path(self.db_path)
        blob_dir = self.blob_dir if isinstance(self.blob_dir, Path) else Path(self.blob_dir)

        if not db_path:
             print("\n⚠️ Database path not configured. Cannot save PDF metadata.")
             return None
        if not blob_dir:
             print("\n⚠️ Blob directory not configured. Cannot save extracted text.")
             return None
        # Gemini client check might depend on whether processing happens here or elsewhere
        # if not self.client:
        #     print("Gemini client not initialized. Cannot process PDF.")
        #     return

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
        conn = None
        try:
            conn = database.get_db_connection(db_path)
            if not conn:
                print("\n⚠️ Error: Failed to connect to the database.")
                return None

            paper_id = database.add_minimal_paper(conn, filename)
            if not paper_id:
                print("\n⚠️ Error: Failed to create initial database record.")
                return None 

            print(f"  📄 Created database record with ID: {paper_id}")

            if arxiv_id_arg:
                if isinstance(arxiv_id_arg, str) and len(arxiv_id_arg) > 5: 
                    database.update_paper_field(conn, paper_id, 'arxiv_id', arxiv_id_arg)
                    print(f"     Updated record with provided arXiv ID: {arxiv_id_arg}")
                else:
                    print(f"  ⚠️ Warning: Provided arXiv ID '{arxiv_id_arg}' seems invalid. Skipping update.")

            print(f"  ⚙️  Extracting text from '{filename}' (using placeholder)...")
            extracted_text = None
            try:
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(pdf_path)
                    extracted_text = ""
                    for page in reader.pages:
                        extracted_text += page.extract_text() + "\n"
                    if not extracted_text:
                        print("  ⚠️ Warning: PyPDF fallback extracted no text.")
                        extracted_text = f"Placeholder: No text extracted via PyPDF for {filename}" 
                except Exception as pypdf_err:
                    print(f"  ⚠️ PyPDF fallback failed: {pypdf_err}. Using basic placeholder.")
                    extracted_text = f"Placeholder: Error during PyPDF extraction for {filename}"

                if not extracted_text: 
                    raise ValueError("Text extraction resulted in empty content.")

            except Exception as extract_err:
                print(f"\n⚠️ Error during text extraction: {extract_err}")
                database.update_paper_field(conn, paper_id, 'status', 'error_process')
                conn.close()
                return None 

            print(f"  💬 Successfully extracted text ({len(extracted_text)} chars) [Placeholder/Fallback].")

            blob_filename = f"paper_{paper_id}_text.txt"
            blob_full_path = blob_dir / blob_filename
            blob_rel_path = blob_filename 

            print(f"  💾 Saving extracted text to {blob_full_path}...")
            save_success = tools.save_text_blob(blob_full_path, extracted_text)

            if not save_success:
                print("\n⚠️ Error: Failed to save text blob.")
                database.update_paper_field(conn, paper_id, 'status', 'error_blob')
                conn.close()
                return None
            else:
                print(f"     Successfully saved text blob.")
                update_success = database.update_paper_field(conn, paper_id, 'blob_path', blob_rel_path)
                if not update_success:
                    print(f"  ⚠️ Warning: Failed to update blob_path in database for ID {paper_id}.")
                    # Log and continue, as text is saved locally

            update_success = database.update_paper_field(conn, paper_id, 'status', 'complete')
            if update_success:
                print(f"\n✅ Processing complete for '{filename}' (ID: {paper_id}).")
            else:
                print(f"\n⚠️ Warning: Failed to update final status to 'complete' for ID {paper_id}.")

            conn.close() 
            return paper_id 

        except Exception as e:
            logger.error(f"An error occurred during PDF processing for '{filename}' (ID: {paper_id}): {e}", exc_info=True)
            print(f"\n❌ An unexpected error occurred during processing: {e}")
            if conn and paper_id:
                try:
                    database.update_paper_field(conn, paper_id, 'status', 'error_generic')
                except Exception as db_err:
                    logger.error(f"Failed to update error status for {paper_id}: {db_err}", exc_info=True)
            if conn:
                conn.close()
            return None 

        finally:
            if conn:
                conn.close()

    def _handle_list_command(self, args: list):
        """Handles the /list command to show papers in the database."""
        # ... (rest of the code remains the same)

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run the Code Agent")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose tool logging')
    args = parser.parse_args()
    config = load_config()
    print("🚀 Starting Code Agent...")
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n⚠️ No API key found in env, trying config.yaml.")
        api_key = config.get('api_key', DEFAULT_CONFIG['api_key'])
    if not api_key:
        print("\n❌ No API key found. Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)

    # Make project_root available to the tools module if needed indirectly
    # (Though direct definition in tools.py is preferred)
    # import src.tools
    # src.tools.project_root = project_root

    # Configure logging level based on verbose flag
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=level)
    # Suppress verbose logs from external libraries
    logging.getLogger('google_genai').setLevel(level)
    logging.getLogger('browser_use').setLevel(level)
    logging.getLogger('agent').setLevel(level)
    logging.getLogger('controller').setLevel(level)

    # Resolve PDF directory relative to project root
    project_root = Path(__file__).parent.parent # Assuming script is in src/
    pdf_dir_path = project_root / config.get('PDFS_TO_CHAT_WITH_DIRECTORY', DEFAULT_CONFIG['PDFS_TO_CHAT_WITH_DIRECTORY'])

    agent = CodeAgent(
        model_name=config.get('model_name', DEFAULT_CONFIG['model_name']),
        verbose=args.verbose or config.get('verbose', DEFAULT_CONFIG['verbose']),
        api_key=api_key,
        default_thinking_budget=config.get('default_thinking_budget', DEFAULT_CONFIG['default_thinking_budget']),
        pdf_dir=str(pdf_dir_path), # Pass absolute path to agent
        db_path=config.get('PAPER_DB_PATH', DEFAULT_CONFIG['PAPER_DB_PATH']),
        blob_dir=config.get('PAPER_BLOBS_DIR', DEFAULT_CONFIG['PAPER_BLOBS_DIR'])
    )
    agent.start_interaction()

if __name__ == "__main__":
    main()