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

# Choose your Gemini model - unless you want something crazy "gemini-2.5-flash-preview-04-17" is the default model
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
DEFAULT_THINKING_BUDGET = 256

# --- Code Agent Class ---
class CodeAgent:
    """A simple coding agent using Google Gemini (google-genai SDK)."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-preview-04-17", verbose: bool = False):
        """Initializes the agent with API key and model name."""
        self.api_key = api_key
        self.verbose = verbose
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
        print("   Use '/upload <path/to/file.pdf>' to seed PDF into context.")
        print("   Use '/reset' to clear the chat and start fresh.")

        # Prompt for thinking budget per session
        try:
            budget_input = input(f"Enter thinking budget (0 to 24000) for this session [{DEFAULT_THINKING_BUDGET}]: ").strip()
            self.thinking_budget = int(budget_input) if budget_input else DEFAULT_THINKING_BUDGET
        except ValueError:
            print(f"⚠️ Invalid thinking budget. Using default of {DEFAULT_THINKING_BUDGET}.")
            self.thinking_budget = DEFAULT_THINKING_BUDGET
        self.thinking_config = types.ThinkingConfig(thinking_budget=self.thinking_budget)

        # Prepare tool config with thinking_config
        tool_config = types.GenerateContentConfig(tools=self.tool_functions, thinking_config=self.thinking_config)

        while True:
            try:
                # Display token count from *previous* turn in the prompt
                # Also show number of active files
                active_files_info = f" [{len(self.active_files)} files]" if self.active_files else ""
                prompt_text = f"\n🔵 You ({self.current_token_count}{active_files_info}): "
                user_input = input(prompt_text).strip()

                if user_input.lower() in ["exit", "quit", "/exit", "/quit", "/q"]:
                    print("\n👋 Goodbye!")
                    break
                if not user_input:
                    continue

                # --- Handle User Commands --- 
                if user_input.lower().startswith("/upload "):
                    pdf_path_str = user_input[len("/upload "):].strip()
                    if pdf_path_str:
                        # Make sure genai is configured before calling upload
                        if not self.client:
                             self._configure_client()
                             if not self.client:
                                 print("\n\u274c Cannot upload: genai client not configured.")
                                 continue # Skip to next loop iteration
                        # Call the upload function (which prints status)
                        uploaded_file = upload_pdf_for_gemini(pdf_path_str)
                        if uploaded_file:
                            print("\n⚒️ Extracting text from PDF to seed context...")
                            extraction_response = self.chat.send_message(
                                message=[uploaded_file, "\n\nExtract the entire text of this PDF, organized by section. Include all tables, and figures (full descriptions where appropriate in place of images)."],
                                config=tool_config
                            )
                            extraction_content = extraction_response.candidates[0].content
                            self.conversation_history.append(extraction_content)
                            # Stop attaching the file after ingestion
                            self.active_files = []
                            print("\n✅ PDF context seeded.")
                        # No else needed, upload_pdf_for_gemini prints errors
                    else:
                        print("\n⚠️ Usage: /upload <relative/path/to/your/file.pdf>")
                    continue # Skip sending this command to the model

                elif user_input.lower() == "/reset":
                    print("\n🎯 Resetting context and starting a new chat session...")
                    self.chat = self.client.chats.create(model=self.model_name, history=[])
                    self.conversation_history = []
                    self.current_token_count = 0
                    print("\n✅ Chat session and history cleared.")
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
                # Prepare tool configuration
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
    print("🚀 Starting Code Agent...")
    api_key = os.getenv('GEMINI_API_KEY')

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

    agent = CodeAgent(api_key=api_key, model_name=MODEL_NAME, verbose=args.verbose)
    # Ensure agent's client is configured before starting interaction
    # This happens inside start_interaction now
    agent.start_interaction()

if __name__ == "__main__":
    main()