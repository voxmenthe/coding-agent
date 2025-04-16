# main.py
from google import genai
from google.genai import types
import os
import sys
from pathlib import Path
import json
from creds import all_creds
import subprocess

# Choose your Gemini model
MODEL_NAME = "gemini-2.0-flash"  # Or "gemini-2.5-pro-preview-03-25"


# --- Tool Functions ---
def read_file(path: str) -> str:
    """Reads the content of a file at the given path."""
    print(f"\n\u2692\ufe0f Tool: Reading file: {path}")
    try:
        # Security check: Ensure path is within the project directory
        target_path = (project_root / path).resolve()
        if not target_path.is_relative_to(project_root):
            return "Error: Access denied. Path is outside the project directory."
        if not target_path.is_file():
            return f"Error: File not found at {path}"
        return target_path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"

def list_files(directory: str) -> str:
    """Lists files in the specified directory relative to the project root."""
    print(f"\n\u2692\ufe0f Tool: Listing files in directory: {directory}")
    try:
        target_dir = (project_root / directory).resolve()
        # Security check: Ensure path is within the project directory
        if not target_dir.is_relative_to(project_root):
            return "Error: Access denied. Path is outside the project directory."
        if not target_dir.is_dir():
            return f"Error: Directory not found at {directory}"

        files = [f.name for f in target_dir.iterdir()]
        return "\n".join(files) if files else "No files found."
    except Exception as e:
        return f"Error listing files: {e}"

def edit_file(path: str, content: str) -> str:
    """Writes or overwrites content to a file at the given path."""
    print(f"\n\u2692\ufe0f Tool: Editing file: {path}")
    try:
        # Security check: Ensure path is within the project directory
        target_path = (project_root / path).resolve()
        if not target_path.is_relative_to(project_root):
            return "Error: Access denied. Path is outside the project directory."

        # Create parent directories if they don't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content)
        return f"File '{path}' saved successfully."
    except Exception as e:
        return f"Error writing file: {e}"

def execute_bash_command(command: str) -> str:
    """Executes a whitelisted bash command in the project's root directory.

    Allowed commands (including arguments):
    - ls ...
    - cat ...
    - git add ...
    - git status ...
    - git commit ...
    - git push ...

    Args:
        command: The full bash command string to execute.

    Returns:
        The standard output and standard error of the command, or an error message.
    """
    print(f"\n\u2692\ufe0f Tool: Executing bash command: {command}")

    whitelist = ["ls", "cat", "git add", "git status", "git commit", "git push"]

    # Check if the command starts with any whitelisted prefix
    is_whitelisted = False
    for prefix in whitelist:
        if command.strip().startswith(prefix):
            is_whitelisted = True
            break

    if not is_whitelisted:
        return f"Error: Command '{command}' is not allowed. Only specific commands (ls, cat, git add/status/commit/push) are permitted."

    try:
        # Execute the command in the project root directory
        # Use shell=True cautiously, but it's simpler for handling complex commands/args here.
        # The whitelist check provides the primary security boundary.
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=project_root, # Ensure command runs in project root
            check=False # Don't raise exception on non-zero exit code, handle manually
        )
        output = f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n--- Command exited with code: {result.returncode} ---"
        return output.strip()

    except Exception as e:
        return f"Error executing command '{command}': {e}"


# --- Code Agent Class ---
class CodeAgent:
    """A simple coding agent using Google Gemini (google-genai SDK)."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """Initializes the agent with API key and model name."""
        self.api_key = api_key
        self.model_name = f'models/{model_name}' # Add 'models/' prefix
        self.tool_functions = [read_file, list_files, edit_file, execute_bash_command]
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

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    agent = CodeAgent(api_key=api_key, model_name=MODEL_NAME)
    agent.start_interaction()