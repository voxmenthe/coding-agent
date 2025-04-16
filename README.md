# 🤖 Gemini Coding Agent

This project implements a Python-based interactive coding agent powered by Google's Gemini models (using the newer `google-genai` SDK).

*Adapted from Thorsten Ball's "How to Build an Agent"* https://ampcode.com/how-to-build-an-agent

The agent can:
- Understand and respond to natural language prompts.
- Interact with your local file system (read, list, edit files).
- Execute shell commands.
- Run commands within a secure Docker sandbox (no network access, resource limits).
- Maintain conversation history.
- Display the token count of the current conversation context in the input prompt.

## ✨ Features

- **Interactive Chat:** Engage in a conversational manner with the AI.
- **File Operations:** Ask the agent to read, list, or modify files within the project directory.
- **Command Execution:** Request the agent to run shell commands in the project's context.
- **Sandboxed Execution:** Safely run potentially risky commands in an isolated Docker container using the `run_in_sandbox` tool (requires Docker).
- **Tool Integration:** Leverages Gemini's function calling capabilities to use defined Python tools.
- **Context Token Count:** Displays the approximate token count for the next API call right in the prompt (e.g., `You (123):`).

## 🛠️ Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/voxmenthe/coding-agent.git
    cd coding-agent
    ```
2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv <your-env-name>
    source <your-env-name>/bin/activate
    sh project_setup.sh
    ```

3.  **Set up API Key:**
    - The agent expects your Google Gemini API key to be available.
    - Currently, it's hardcoded in `src/main.py` to read from `all_creds['GEMINI_API_KEY']`. You'll need to adapt this to your own credential management (e.g., environment variables, a `.env` file).
    - *Example using environment variable:* Modify `src/main.py` to use `os.environ.get("GEMINI_API_KEY")` and set the variable in main.py:
      ```python
      api_key = os.environ.get("GEMINI_API_KEY")
      ```

4.  **(Optional) Docker for Sandbox:**
    - If you want to use the `run_in_sandbox` tool, ensure you have Docker installed and running on your system.
    - You might need to pull the base image specified in `src/tools.py` (e.g., `python:3.12-slim`) if it's not already available locally:
      ```bash
      docker pull python:3.12-slim
      ```

## ▶️ Running the Agent

Navigate to the `src` directory and run the main script:

```bash
cd src
python main.py
```

The agent will initialize and present you with a prompt like:

```
⚒️ Agent ready. Ask me anything. Type 'exit' to quit.

🔵 You (0): 
```

## 💬 Usage

- Simply type your requests or questions at the `You (<token_count>):` prompt.
- To exit the agent, type `exit` or `quit`.
- Ask it to perform tasks like:
    - "Read the file src/tools.py"
    - "List files in the root directory"
    - "Edit README.md and add a section about future plans"
    - "run the command 'ls -l'"
    - "run 'pip list' in the sandbox"
- The number in parentheses indicates the approximate token count of the conversation history that will be sent with your *next* message.

## 📝 Notes

- The agent operates relative to the project root directory defined in `src/main.py`.
- Ensure the API key handling is secure and not committed to version control.