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
from google import genai
from google.api_core import exceptions as google_exceptions
from pypdf import PdfReader
import yaml

import time
import asyncio
from pydantic import SecretStr
from dotenv import load_dotenv
from src.agent_browser_utils import setup_browser, agent_loop
import logging

logger = logging.getLogger(__name__)

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# --- Constants ---
MAX_GEMINI_PDF_SIZE_BYTES = 200 * 1024 * 1024 # 200 MB, adjust as needed
GEMINI_SUPPORTED_MIME_TYPES = [
    "application/pdf",
]
GEMINI_PROCESSING_TIMEOUT_SECONDS = 300 # 5 minutes
GEMINI_POLLING_INTERVAL_SECONDS = 5

# --- Helper Functions ---

def check_pdf_size(pdf_path: Path) -> bool:
    """Checks if the PDF file size is within Gemini's limits."""
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    size = pdf_path.stat().st_size
    if size > MAX_GEMINI_PDF_SIZE_BYTES:
        logger.warning(
            f"PDF file {pdf_path.name} exceeds maximum size "
            f"({size / (1024*1024):.2f}MB > "
            f"{MAX_GEMINI_PDF_SIZE_BYTES / (1024*1024):.2f}MB)."
        )
        return False
    return True

def _check_docker_running() -> tuple[bool, docker.DockerClient | None, str]:
    """Checks if the Docker daemon is running and returns client or error."""
    try:
        client = docker.from_env()
        client.ping() 
        return True, client, "Docker daemon is running."
    except DockerException as e:
        error_msg = (
            f"Docker connection failed: {e}\n"
            "Please ensure Docker Desktop (or docker daemon) is running."
        )
        return False, None, error_msg
    except Exception as e:
        return False, None, f"Error checking Docker status: {e}"

# --- Gemini PDF Processing --- 

def extract_text_from_pdf_gemini(
    pdf_path: Path,
    genai_client: genai.client.Client,
    model_name: str,
) -> str | None:
    """
    Uploads a PDF to Gemini, extracts text using the specified model,
    and cleans up the uploaded file.

    Args:
        pdf_path: Path to the local PDF file.
        genai_client: An initialized google.genai.client.Client instance.
        model_name: The name of the Gemini model to use (e.g., 'gemini-1.5-flash-latest').

    Returns:
        The extracted text content as a string, or None if extraction fails.
    """
    # Re-adding size check logic (assuming it was correct before)
    if not check_pdf_size(pdf_path):
         return None

    logger.info(f"Uploading PDF {pdf_path.name} to Gemini...")
    uploaded_file: genai.types.File | None = None 
    try:
        # 1. Upload the file synchronously
        uploaded_file = genai_client.files.upload(
            file=pdf_path,
        )
        logger.info(
            f"Successfully uploaded {pdf_path.name} as Gemini file: "
            f"{uploaded_file.name} ({uploaded_file.uri})"
        )

        # 2. Generate content using the uploaded file
        logger.info(f"Requesting text extraction using model {model_name}...")
        prompt = "Extract all text content from this PDF document."
        response = genai_client.models.generate_content(
            model=model_name,
            contents=[prompt, uploaded_file]
        )

        # 3. Process the response
        # Using the same response processing logic as before
        if response.candidates and response.candidates[0].content.parts:
            extracted_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            if extracted_text:
                logger.info(f"Successfully extracted text from {pdf_path.name}.")
                return extracted_text.strip()
            else:
                logger.warning(f"Gemini returned no text for {pdf_path.name}.")
                return None
        else:
            logger.warning(f"Gemini response format unexpected or empty for {pdf_path.name}. Response: {response}")
            return None

    except google_exceptions.GoogleAPIError as e: 
        logger.error(f"Gemini API error during processing {pdf_path.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Gemini processing for {pdf_path.name}: {e}", exc_info=True)
        return None
    finally:
        # 4. Clean up the uploaded file
        if uploaded_file:
            try:
                logger.info(f"Deleting Gemini file {uploaded_file.name}...")
                genai_client.files.delete(name=uploaded_file.name)
                logger.info(f"Successfully deleted Gemini file {uploaded_file.name}.")
            except google_exceptions.GoogleAPIError as e: 
                logger.error(f"Failed to delete Gemini file {uploaded_file.name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error deleting Gemini file {uploaded_file.name}: {e}", exc_info=True)


# --- Tool Functions ---
def read_file(path: str) -> str:
    """Reads the content of a file at the given path relative to the current working directory."""
    print(f"\n\u2692\ufe0f Tool: Reading file: {path}")
    try:
        cwd = Path(os.getcwd())
        target_path = (cwd / path).resolve()
        if not target_path.is_relative_to(cwd):
            return "Error: Access denied. Path is outside the current working directory."
        if not target_path.is_file():
            return f"Error: File not found at {path}"
        return target_path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"

def list_files(directory: str) -> str:
    """Lists files in the specified directory relative to the current working directory."""
    print(f"\n\u2692\ufe0f Tool: Listing files in directory: {directory}")
    try:
        cwd = Path(os.getcwd())
        target_dir = (cwd / directory).resolve()
        if not target_dir.is_relative_to(cwd):
            return "Error: Access denied. Path is outside the current working directory."
        if not target_dir.is_dir():
            return f"Error: Directory not found at {directory}"

        files = [f.name for f in target_dir.iterdir()]
        return "\n".join(files) if files else "No files found."
    except Exception as e:
        return f"Error listing files: {e}"

def edit_file(path: str, content: str) -> str:
    """Writes or overwrites content to a file at the given path relative to the current working directory."""
    print(f"\n\u2692\ufe0f Tool: Editing file: {path}")
    try:
        cwd = Path(os.getcwd())
        target_path = (cwd / path).resolve()
        if not target_path.is_relative_to(cwd):
            return "Error: Access denied. Path is outside the current working directory."

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content)
        return f"File '{path}' saved successfully."
    except Exception as e:
        return f"Error writing file: {e}"

def execute_bash_command(command: str) -> str:
    """Executes a whitelisted bash command in the current working directory.

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

    is_whitelisted = False
    for prefix in whitelist:
        if command.strip().startswith(prefix):
            is_whitelisted = True
            break

    if not is_whitelisted:
        return f"Error: Command '{command}' is not allowed. Only specific commands (ls, cat, git add/status/commit/push) are permitted."

    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=os.getcwd(), 
            check=False 
        )
        if result.stdout:
            print(f"\n\u25b6\ufe0f Command Output (stdout):\n{result.stdout.strip()}")
        output = f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n--- Command exited with code: {result.returncode} ---"
        return output.strip()

    except Exception as e:
        return f"Error executing command: {e}"

# --- File System Operations ---

def save_text_blob(file_path: Path, content: str) -> bool:
    """Saves the given text content to the specified file path.

    Args:
        file_path: The absolute Path object for the file.
        content: The string content to save.

    Returns:
        True if saving was successful, False otherwise.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Successfully saved text blob to {file_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving text blob to {file_path}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving text blob to {file_path}: {e}", exc_info=True)
        return False

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
    image = "python:3.12-slim"
    print(f"\n\u2692\ufe0f Tool: Running in sandbox (Image: {image}): {command}")

    is_running, client, message = _check_docker_running()
    if not is_running:
        return f"Error: Cannot run sandbox. {message}"

    try:
        print(f"\n\u23f3 Starting Docker container (image: {image})...")
        container_output = client.containers.run(
            image=image, 
            command=f"sh -c '{command}'", 
            working_dir="/app",
            volumes={os.getcwd(): {'bind': '/app', 'mode': 'rw'}},
            remove=True,       # Remove container after execution
            network_mode='none', # Disable network access
            mem_limit='512m',   # Limit memory to 512MB
            detach=False,       # Run in foreground
            stdout=True,        # Capture stdout
            stderr=True         # Capture stderr
        )
        output_str = container_output.decode('utf-8').strip()
        print(f"\n\u25b6\ufe0f Sandbox Output:\n{output_str}")
        return f"--- Container Output ---\n{output_str}"

    except DockerException as e:
        error_msg = f"Docker error during sandbox execution: {e}"
        print(f"\n\u274c {error_msg}")
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
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
    except Exception:
        raise ValueError("start_date and end_date must be in YYYY-MM-DD format")
    categories = ['cs.*', 'stat.*']
    raw_keywords = [kw.strip().lower() for kw in keywords.split(' OR ') if kw.strip()]
    processed_keywords = [kw for kw in raw_keywords if kw not in ('and', 'or')]
    query = build_query(categories, processed_keywords)
    print(f"\n🔍 Arxiv search query: {query} | dates: {start_date} to {end_date} | max_results: {max_results}")
    entries = fetch_entries(query, max_results=max_results, verbose=False)
    print(f"\n🔍 Raw entries fetched: {len(entries)}")
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

def upload_pdf_for_gemini(pdf_path_str: str) -> genai.types.File | None:
    """
    Uploads a PDF file relative to the project root to Google Gemini
    using the File API and waits for it to be processed.

    Args:
        pdf_path_str: The path to the PDF file, relative to the project root.

    Returns:
        A google.genai.types.File object if successful, None otherwise.
        Prints errors to console.
    """
    global project_root
    if 'project_root' not in globals():
         project_root = Path(__file__).resolve().parents[1]
         print("⚠️ 'project_root' was not defined globally in tools.py, attempting definition.")
         if not project_root:
             print("Error: Could not define project_root. Please ensure it's set correctly.")

    try:
        target_path = (project_root / pdf_path_str).resolve()

        if not target_path.is_relative_to(project_root):
            print(f"\n\u274c Error: Access denied. Path '{pdf_path_str}' is outside the project directory.")
            return None
        if not target_path.is_file():
            print(f"\n\u274c Error: PDF file not found at resolved path '{target_path}'")
            return None
        if target_path.suffix.lower() != ".pdf":
            print(f"\n\u274c Error: File '{target_path.name}' is not a PDF.")
            return None

        print(f"\n\u2692\ufe0f Uploading '{target_path.name}'...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(f"\n\u001b[31mError: GEMINI_API_KEY environment variable not set.\nPlease export your API key before uploading PDFs.\u001b[0m")
            return None
        client = genai.Client(api_key=api_key)

        pdf_file = client.files.upload(file=target_path)
        print(f"\u2705 Uploaded '{pdf_file.display_name}' as: {pdf_file.name}")
        print("⏳ Waiting for processing...")

        start_time = time.time()
        timeout_seconds = 120 
        while pdf_file.state.name == "PROCESSING":
            if time.time() - start_time > timeout_seconds:
                 print(f"\n\u274c Error: File processing timed out after {timeout_seconds} seconds for {pdf_file.name}.")
                 try:
                     client.files.delete(name=pdf_file.name)
                     print(f"🧹 Cleaned up timed-out file: {pdf_file.name}")
                 except Exception as delete_e:
                     print(f"⚠️ Could not delete timed-out file {pdf_file.name}: {delete_e}")
                 return None

            time.sleep(5) 
            pdf_file = client.files.get(name=pdf_file.name) 

        if pdf_file.state.name == "ACTIVE":
            print(f"\u2705 File '{pdf_file.display_name}' is ready.")
            return pdf_file
        else:
            print(f"\n\u274c Error: File processing failed for '{pdf_file.display_name}'. Final State: {pdf_file.state.name}")
            try:
                 client.files.delete(name=pdf_file.name)
                 print(f"🧹 Cleaned up failed file: {pdf_file.name}")
            except Exception as delete_e:
                 print(f"⚠️ Could not delete failed file {pdf_file.name}: {delete_e}")
            return None

    except Exception as e:
        print(f"\n\u274c An error occurred during PDF upload/processing: {e}")
        if 'pdf_file' in locals() and hasattr(pdf_file, 'name'):
             try:
                 print(f"🧹 Attempting to delete potentially failed upload: {pdf_file.name}")
                 client.files.delete(name=pdf_file.name)
             except Exception as delete_e:
                 print(f"⚠️ Could not delete file during error cleanup: {delete_e}")
        return None

def google_search(query: str) -> str:
    """Search Google for the given query using browser-use and return JSON-formatted results."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17"),
            api_key=SecretStr(os.getenv("GEMINI_API_KEY"))
        )
        browser, context = asyncio.run(setup_browser(headless=True))
        result = asyncio.run(agent_loop(
            llm,
            context,
            f"Search Google for '{query}' and extract the first 10 results as JSON list of {{'title','url'}}.",
            initial_url=f"https://www.google.com/search?q={query}"
        ))
        return result or "No results."
    except Exception as e:
        return f"Error during google_search: {e}"

def open_url(url: str) -> str:
    """Open a URL using browser-use and return the page's visible text content."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17"),
            api_key=SecretStr(os.getenv("GEMINI_API_KEY"))
        )
        browser, context = asyncio.run(setup_browser(headless=True))
        result = asyncio.run(agent_loop(
            llm,
            context,
            f"Extract and return visible text content from the page at: {url}.",
            initial_url=url
        ))
        return result or "No content."
    except Exception as e:
        return f"Error during open_url: {e}"