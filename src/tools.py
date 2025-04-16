# --- Imports ---
import docker
from docker.errors import DockerException
from pathlib import Path
import subprocess
import sys

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
            volumes={str(project_root): {'bind': '/app', 'mode': 'rw'}},
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