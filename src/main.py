from google import genai
from google.genai import types
import os
import sys
from pathlib import Path
import json
from creds import all_creds
from src.tools import read_file, list_files, edit_file, execute_bash_command, run_in_sandbox
import traceback

# Choose your Gemini model
MODEL_NAME = "gemini-2.0-flash" # Using latest flash model which is currently 2.0 NOT 1.5

# Define project root - needed here for agent initialization
project_root = Path(__file__).resolve().parents[1]

# --- Code Agent Class ---
class CodeAgent:
    """A simple coding agent using Google Gemini (google-genai SDK)."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """Initializes the agent with API key and model name."""
        self.api_key = api_key
        self.model_name = f'models/{model_name}' # Add 'models/' prefix
        # Use imported tool functions
        self.tool_functions = [
            read_file,
            list_files,
            edit_file,
            execute_bash_command,
            run_in_sandbox # Add the new sandbox tool
        ]
        self.client = None
        self.chat = None
        self._configure_client()

    def _configure_client(self):
        """Configures the Google Generative AI client."""
        print("\n\u2692\ufe0f Configuring genai client...")
        try:

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

        print("\n\u2692\ufe0f Agent ready. Ask me anything. Type 'exit' to quit.")

        # Prepare tool config once to pass to send_message
        tool_config = types.GenerateContentConfig(tools=self.tool_functions)

        while True:
            try:
                user_input = input("\n🔵 \x1b[94mYou:\x1b[0m ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    continue

                print("\n\u23f3 Sending message and processing...")
                # Use chat.send_message, passing the tool config
                response = self.chat.send_message(
                    message=user_input,
                    config=tool_config # Pass tools here for potential automatic function calling
                )

                # Access the final text response (assuming response structure is similar)
                # Check the actual response object structure if this causes errors
                print(f"\n🟢 \x1b[92mAgent:\x1b[0m {response.text}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n🔴 \x1b[91mAn error occurred during interaction: {e}\x1b[0m")
                traceback.print_exc() # Print traceback for debugging

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting Code Agent...")
    api_key = all_creds['GEMINI_HIMS_API_KEY_mlproj_V1']

    # Add project root to sys.path
    sys.path.insert(0, str(project_root))

    # Make project_root available to the tools module if needed indirectly
    # (Though direct definition in tools.py is preferred)
    # import src.tools
    # src.tools.project_root = project_root

    agent = CodeAgent(api_key=api_key, model_name=MODEL_NAME)
    agent.start_interaction()