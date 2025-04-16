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
        # self.chat = None # No longer maintaining a persistent chat object
        self._configure_client()

    def _configure_client(self):
        """Configures the Google Generative AI client."""
        print("\n\u2692\ufe0f Configuring genai client...")
        try:
            # Client likely picks up GOOGLE_API_KEY from env
            self.client = genai.Client(api_key=self.api_key)
            print("\u2705 Client configured successfully.")
        except Exception as e:
            print(f"\u274c Error configuring genai client: {e}")
            sys.exit(1)

    def start_interaction(self):
        """Starts the main interaction loop using generate_content."""
        if not self.client:
            print("\n\u274c Client not configured. Exiting.")
            return

        print("\n\u2692\ufe0f Agent ready. Ask me anything involving file operations (read, list, edit) or allowed bash commands (ls, cat, git...). Type 'exit' to quit.")

        # Define the tool configuration once
        tool_config = types.GenerateContentConfig(
            tools=self.tool_functions
        )

        while True:
            try:
                user_input = input("\n🔵 \x1b[94mYou:\x1b[0m ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("\n👋 Goodbye!")
                    break

                print("\n\u23f3 Sending message and processing...")
                # Use generate_content for each message, enabling automatic function calling via config
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_input,
                    config=tool_config,
                )

                # The SDK should handle the function call automatically and return the final text
                print(f"\n🟢 \x1b[92mAgent:\x1b[0m {response.text}")

            except Exception as e:
                print(f"\n🔴 \x1b[91mAn error occurred during interaction: {e}\x1b[0m")
                # Optionally add more specific error handling, e.g., for API errors
                # break

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting Code Agent...")
    api_key = all_creds['GEMINI_HIMS_API_KEY_mlproj_V1']

    # Set project_root globally for tools that might need it (though tools.py defines its own)
    # project_root = Path(__file__).resolve().parents[1] # Already defined above
    sys.path.insert(0, str(project_root))

    # Make project_root available to the tools module if needed indirectly
    # (Though direct definition in tools.py is preferred)
    # import src.tools
    # src.tools.project_root = project_root

    agent = CodeAgent(api_key=api_key, model_name=MODEL_NAME)
    agent.start_interaction()