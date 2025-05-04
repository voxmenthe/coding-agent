from google import genai
from google.genai import types
import os
import sys
from pathlib import Path
from src.tools import (
    read_file, list_files, edit_file, execute_bash_command,
    run_in_sandbox, find_arxiv_papers, get_current_date_and_time,
    upload_pdf_for_gemini, google_search, open_url
)
import traceback
import argparse
import functools
import logging
from prompt_toolkit.completion import WordCompleter, NestedCompleter, PathCompleter, Completer, Completion
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
import bisect # Added for efficient searching
import yaml

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
    'SAVED_CONVERSATIONS_DIRECTORY': 'SAVED_CONVERSATIONS/'
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
        except Exception as e:
            print(f"Error loading config: {e}")
    return config

# --- Agent Class ---
class CodeAgent:
    """A simple coding agent using Google Gemini (google-genai SDK)."""

    def __init__(self, model_name: str, verbose: bool, api_key: str, default_thinking_budget: int, pdf_dir: str):
        """Initializes the agent with API key and model name."""
        self.model_name = model_name
        self.verbose = verbose
        self.api_key = api_key
        self.model_name = f'models/{model_name}' # Add 'models/' prefix
        # Use imported tool functions
        self.tool_functions = [
            read_file,
            list_files,
            edit_file,
            execute_bash_command,
            run_in_sandbox,
            find_arxiv_papers,
            get_current_date_and_time,
            google_search,
            open_url
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

                if user_input.lower().startswith("/pdf "):
                    # Decrement message counter as this command 'message' doesn't persist initially
                    self._messages_this_interval -= 1
                    pdf_filenames_str = user_input[len("/pdf "):].strip()
                    filenames = pdf_filenames_str.split() # Split potential multiple filenames
                    processed_one = False

                    if not filenames:
                         print("\n⚠️ Usage: /pdf <filename1> [filename2] ...")
                         continue

                    # --- Process the FIRST valid PDF filename provided --- 
                    target_filename = filenames[0]
                    pdf_path = self.pdfs_dir_abs_path / target_filename

                    if not pdf_path.is_file() or not target_filename.lower().endswith('.pdf'):
                         print(f"\n❌ Error: PDF file '{target_filename}' not found or invalid in {self.pdfs_dir_abs_path}.")
                         # Ensure PDF list for completer is up-to-date if needed (complex)
                         # For now, just inform the user.
                         continue

                    pdf_path_str = str(pdf_path)
                    print(f"\n⬆️ Processing PDF: {target_filename}...")
                    # Account for messages added by upload/extraction process
                    # User command doesn't count, but the upload/extraction request/response do.
                    self._messages_this_interval += 2 # Estimate 2 messages (req+resp)

                    # Use the existing upload_pdf_for_gemini tool function
                    uploaded_file = upload_pdf_for_gemini(pdf_path_str)
                    if uploaded_file:
                        print("\n⚒️ Extracting text from PDF to seed context...")
                        extraction_response = self.chat.send_message(
                            message=[uploaded_file, "\n\nExtract the entire text of this PDF, organized by section. Include all tables, and figures (full descriptions where appropriate in place of images)."],
                            config=types.GenerateContentConfig(tools=self.tool_functions, thinking_config=self.thinking_config)
                        )
                        extraction_content = extraction_response.candidates[0].content
                        self.conversation_history.append(extraction_content)
                        # Stop attaching the file after ingestion
                        self.active_files = []
                        print("\n✅ PDF context seeded.")
                        processed_one = True
                    # No else needed, upload_pdf_for_gemini prints errors

                    # If multiple filenames were given, inform user only first was processed
                    if processed_one and len(filenames) > 1:
                        print(f"\nℹ️ Note: Processed '{filenames[0]}'. Multiple file processing per command not yet supported.")

                    continue # Skip sending this command to the model

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
                        contents=self.conversation_history
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
        pdf_dir=str(pdf_dir_path) # Pass absolute path to agent
    )
    agent.start_interaction()

if __name__ == "__main__":
    main()