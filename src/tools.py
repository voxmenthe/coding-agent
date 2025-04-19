# --- Imports ---
import docker
from docker.errors import DockerException
from pathlib import Path
import subprocess
import os
import requests
import feedparser
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from src.find_arxiv_papers import build_query, fetch_entries # build_query now in find_arxiv_papers.py

# --- Project Root ---
project_root = Path(__file__).resolve().parents[1]

# --- Helper Functions ---
def _check_docker_running() -> tuple[bool, docker.DockerClient | None, str]:
    """Checks if the Docker daemon is running and returns client or error."""
    try:
        client = docker.from_env()
        client.ping() # Verify connection
        return True, client, "Docker daemon is running."
    except DockerException as e:
        error_msg = (
            f"Docker connection failed: {e}\n"
            "Please ensure Docker Desktop (or docker daemon) is running."
        )
        return False, None, error_msg
    except Exception as e:
        # Catch other potential issues like docker library not installed
        return False, None, f"Error checking Docker status: {e}"

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
        # Security check: Ensure path is within the current working directory
        cwd = Path(os.getcwd())
        target_path = (cwd / path).resolve()
        if not target_path.is_relative_to(cwd):
            return "Error: Access denied. Path is outside the current working directory."

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
        # --- Print stdout directly to console for visibility ---
        if result.stdout:
            print(f"\n\u25b6\ufe0f Command Output (stdout):\n{result.stdout.strip()}")
        # --- Format output for LLM --- 
        output = f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n--- Command exited with code: {result.returncode} ---"
        return output.strip()

    except Exception as e:
        return f"Error executing command '{command}': {e}"

def run_in_sandbox(command: str) -> str:
    """Executes a command inside a sandboxed Docker container.
    The sandbox's cwd is the directory where the agent is run from.
    Uses the 'python:3.12-slim' image.
    The project directory is mounted at /app.
    Network access is disabled for security.
    Resource limits (CPU, memory) are applied.

    Args:
        command: The command string to execute inside the container's /app directory.

    Returns:
        The combined stdout/stderr from the container, or an error message.
    """
    # Hardcode the image name
    image = "python:3.12-slim"
    print(f"\n\u2692\ufe0f Tool: Running in sandbox (Image: {image}): {command}")

    is_running, client, message = _check_docker_running()
    if not is_running:
        return f"Error: Cannot run sandbox. {message}"

    try:
        print(f"\n\u23f3 Starting Docker container (image: {image})...")
        container_output = client.containers.run(
            image=image, # Use the hardcoded image variable
            command=f"sh -c '{command}'", # Execute command within a shell in the container
            working_dir="/app",
            volumes={os.getcwd(): {'bind': '/app', 'mode': 'rw'}},
            remove=True,        # Remove container after execution
            network_mode='none',# Disable networking
            mem_limit='512m',   # Limit memory
            detach=False,       # Run synchronously
            stdout=True,        # Capture stdout
            stderr=True         # Capture stderr
        )
        output_str = container_output.decode('utf-8').strip()
        print(f"\n\u25b6\ufe0f Sandbox Output:\n{output_str}")
        return f"--- Container Output ---\n{output_str}"

    except DockerException as e:
        error_msg = f"Docker error during sandbox execution: {e}"
        print(f"\n\u274c {error_msg}")
        # Provide more specific feedback if image not found
        if "not found" in str(e).lower() or "no such image" in str(e).lower():
             error_msg += f"\nPlease ensure the image '{image}' exists locally or can be pulled."
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error during sandbox execution: {e}"
        print(f"\n\u274c {error_msg}")
        return f"Error: {error_msg}"

def get_current_date_and_time(timezone: str) -> str:
    """Returns the current date and time as ISO 8601 string in the specified timezone. Default is PST (America/Los_Angeles) if an invalid timezone is provided."""
    try:
        print(f"\n\u2692\ufe0f Tool: Getting current date and time for timezone: {timezone}")
        tz = ZoneInfo(timezone)
    except Exception as e:
        print(f"Error: Invalid timezone '{timezone}' provided. Using default: America/Los_Angeles")
        tz = ZoneInfo('America/Los_Angeles')
    now = datetime.now(tz)
    return now.isoformat()

def find_arxiv_papers(keywords: str, start_date: str, end_date: str, max_results: int) -> str:
    """Search arXiv for papers based on keywords and date range.

    Args:
        keywords: A string containing space-separated keywords or phrases to search for.
                  Use ' OR ' to separate distinct search terms/phrases (e.g., '"llm reasoning" OR grpo').
                  Provide meaningful keywords; logical operators like standalone 'and'/'or' will be ignored.
        start_date: Inclusive start date (YYYY-MM-DD).
        end_date: Inclusive end date (YYYY-MM-DD).
        max_results: Maximum number of results to return.

    Returns:
        JSON string of papers with title, link, summary, and published date."""
    # Parse date range
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
    except Exception:
        raise ValueError("start_date and end_date must be in YYYY-MM-DD format")
    # Define categories and keywords
    categories = ['cs.*', 'stat.*']
    # Split by ' OR ', convert to lowercase, and strip whitespace
    raw_keywords = [kw.strip().lower() for kw in keywords.split(' OR ') if kw.strip()]
    processed_keywords = [kw for kw in raw_keywords if kw not in ('and', 'or')]
    # Build query and fetch entries
    query = build_query(categories, processed_keywords)
    print(f"\n🔍 Arxiv search query: {query} | dates: {start_date} to {end_date} | max_results: {max_results}")
    entries = fetch_entries(query, max_results=max_results, verbose=False)
    print(f"\n🔍 Raw entries fetched: {len(entries)}")
    # Filter by publication date and format
    results = []
    for entry in entries:
        pub_str = entry.published[:10]
        pub_dt = datetime.fromisoformat(pub_str)
        if start_dt <= pub_dt <= end_dt:
            results.append({
                "title": entry.title.strip(),
                "link": entry.link,
                "summary": entry.summary.strip(),
                "published": pub_str
            })
            if len(results) >= max_results:
                break
    import json
    print(f"\n🔍 Filtered entries count: {len(results)}")
    return json.dumps(results, indent=2)