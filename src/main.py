# main.py
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool, ToolConfig
import os
import sys
from pathlib import Path
import json
from creds import all_creds


# --- Configuration ---
# Load the API key
genai.configure(api_key=all_creds['GEMINI_HIMS_API_KEY_mlproj_V1'])

# Choose your Gemini model
MODEL_NAME = "gemini-2.0-flash"  # Or "gemini-2.5-pro-preview-03-25"


# --- Agent Class ---
class Agent:
    def __init__(self, model_name: str):
        print(f"✨ Initializing Agent with model: {model_name}")
        # We'll initialize the model later, when we know about tools
        self.model_name = model_name
        self.model = None # Placeholder
        self.chat = None # Placeholder for the chat session
        # Store tool functions and their definitions
        self.tools = {
            "read_file": read_file
            # Add more tools here later
        }
        self.tool_definitions = [
            READ_FILE_TOOL
            # Add more tool definitions here later
        ]

    def _initialize_chat(self):
        """Initializes or re-initializes the model and chat session."""
        print(f"🔄 Initializing model and chat (Tools: {'Yes' if self.tool_definitions else 'No'})...")

        self.model = genai.GenerativeModel(
            self.model_name,
            # system_instruction="You are a helpful coding assistant.", # Optional: Add system instructions
            tools=self.tool_definitions  # Use the agent's tool definitions
        )
        self.chat = self.model.start_chat(enable_automatic_function_calling=True)
        print("✅ Model and chat initialized.")

    def run(self):
        """Starts the main interaction loop."""
        self._initialize_chat() # Initialize without tools first

        print("\n💬 Chat with Gemini (type 'quit' or 'exit' to end)")
        print("-" * 30)

        while True:
            try:
                user_input = input("🔵 \x1b[94mYou:\x1b[0m ") # Blue text for user
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Send message to Gemini
                print("🧠 Thinking...")
                response = self.chat.send_message(user_input)

                # Print Gemini's response
                # Note: Automatic function calling handles tool calls implicitly here!
                # We'll need to adjust this later when we manually handle tools.
                text_response = response.text # Simplified for now
                print(f"🟠 \x1b[93mGemini:\x1b[0m {text_response}") # Orange text for Gemini

            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"🛑 An error occurred: {e}")
                # Consider whether to break or continue on error
                # break


# --- Tool Functions ---

def read_file(path: str) -> str:
    """
    Reads the content of a file at the given relative path.

    Args:
        path: The relative path to the file.

    Returns:
        The content of the file as a string, or an error message.
    """
    print(f"🛠️ Executing read_file tool with path: {path}")
    try:
        file_path = Path(path).resolve() # Resolve to absolute path for security/clarity
        # Basic security check: ensure file is within the project directory
        if not file_path.is_relative_to(Path.cwd()):
            raise SecurityError(f"Access denied: Path '{path}' is outside the project directory.")

        if not file_path.is_file():
            return f"Error: Path '{path}' is not a file or does not exist."
        content = file_path.read_text(encoding='utf-8')
        print(f"✅ read_file successful for: {path}")
        return content
    except SecurityError as e:
        print(f"🚨 Security Error in read_file: {e}")
        return f"Error: {e}"

class SecurityError(Exception):
    """Custom exception for security violations."""
    pass

# Tool Definitions (using FunctionDeclaration)
READ_FILE_TOOL = FunctionDeclaration(
    name="read_file",
    description="Read the contents of a specified file path relative to the current working directory. Use this to examine file contents.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The relative path of the file to read (e.g., 'src/main.py', 'README.md')."
            }
        },
        "required": ["path"]
    }
)

# --- Main Execution ---
if __name__ == "__main__":
    agent = Agent(MODEL_NAME)
    agent.run()