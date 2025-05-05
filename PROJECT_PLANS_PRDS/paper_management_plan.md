# Plan: Persistent Storage and Querying for Processed PDFs

## 0. Current Priority: Basic Ingestion

The immediate goal is **baseline ingestion**: as soon as a PDF is loaded into chat (e.g., with `/pdf <filename>`), the system must persist:
* the raw extracted text
* `source_filename`
* `processed_timestamp`

Everything else is optional for the first milestone.

Any richer metadata (title, authors, abstract, arxiv_id, etc.) may be added later. Integration with the arXiv API is **TBD**; the database and code must remain functional even if no arXiv lookup is performed.

The design must also cope with PDF sources **outside arXiv** (e.g., conference websites, personal blogs). While we expect most papers to come from arXiv, a meaningful subset will not. Therefore, the schema and logic should be generic and *not* assume the presence of an arXiv identifier.

## 1. Context

Currently, the agent ([src/main.py](cci:7://file:///Volumes/bdrive/repos/coding-agent/src/main.py:0:0-0:0)) includes a `/pdf <filename>` command. This command:
1.  Takes a PDF filename located in the directory specified by `PDFS_TO_CHAT_WITH_DIRECTORY` in `src/config.yaml`.
2.  Uses the [upload_pdf_for_gemini](cci:1://file:///Volumes/bdrive/repos/coding-agent/src/tools.py:253:0-341:19) function ([src/tools.py](cci:7://file:///Volumes/bdrive/repos/coding-agent/src/tools.py:0:0-0:0)) to upload the file via the Google Generative AI File API.
3.  Sends a message to the Gemini model asking it to extract the text content of the PDF.
4.  The extracted text is returned as a model message and added to the current chat session's history (`self.conversation_history`).
5.  If the `/save` command is used, this extracted text (as part of the history) gets saved in the conversation JSON, but it's not stored separately or associated with structured metadata beyond the chat context.

The goal is to implement a system where, upon successful text extraction via the `/pdf` command, the extracted text and relevant metadata are automatically stored persistently and separately from the chat history, allowing for later querying and retrieval.

An example query might include:

* Find all the papers with the tags "vision" and "transformer" but not "RL"
* Load the abstract of each of these papers into the chat context OR load the full text etc.

So we will need some specific tools for this.

## 2. Requirements

*   **Persistence:** Store the extracted text content of each successfully processed PDF.
*   **Metadata Storage:** Store structured metadata associated with each PDF. Core fields should include:
    *   `source_filename`: Original PDF filename.
    *   `processed_timestamp`: When the PDF was processed.
    *   `title` (potentially extracted).
    *   `authors` (potentially extracted).
    *   `abstract` (potentially extracted).
    *   `references` (potentially extracted).
    *   `arxiv_url` (if applicable, will be available if arxiv search returns it).
    *   `github_repo` (if applicable, user-added).
    *   `last_updated`: When the metadata was last updated.
    *   `last_accessed`: When the paper was last read into chat context (starts with initial processing).
    *   `tags`: A list of user-defined tags (e.g., `["vision", "transformer", "2024"]`). The user can edit tags at any time
    *   `notes`: Free-form user notes. Can call up and edit at any time.
*   **Extensibility:** Allow adding new, optional metadata fields easily in the future without breaking the system.
*   **Querying:** Enable querying the stored metadata. A key use case is retrieving information based on tags (e.g., "list all abstracts for papers tagged 'vision'").
*   **Accessibility:** The stored text and metadata should be easily accessible, ideally for both the agent (via commands) and the user (potentially via direct file inspection/editing).
*   **Lightweight & VCS-Friendly:** Avoid heavy database dependencies if possible. The storage format should work reasonably well with Git (e.g., produce meaningful diffs).

## 3. Proposed Storage Option
### Lightweight DB (SQLite)

*   **Structure:**
    ```
    processed_pdfs/
    ├── metadata.db       # SQLite database file
    └── blobs/            # Optional: Store text blobs separately
        └── <unique_pdf_id>.txt
    ```
    *(Alternatively, the text could be stored directly in a TEXT column within the DB)*
*   **Schema:** A table (`processed_pdfs`) with columns for `id`, `source_filename`, `processed_timestamp`, `title`, `abstract`, `arxiv_url`, `tags` (could be stored as JSON string or normalized into separate tags table), `notes`, `blob_path` (if storing text separately).
*   **Pros:**
    *   Robust querying using SQL (filtering, joining, aggregation).
    *   Handles data integrity and updates reliably (transactions).
    *   Python's `sqlite3` module is built-in, so no external dependencies needed.
    *   Scales better for querying than flat files as the number of entries grows.
*   **Cons:**
    *   `metadata.db` is a binary file, making Git diffs less meaningful.
    *   Requires writing SQL queries (though simple `SELECT` statements are sufficient for basic needs).
    *   Slightly more complex initial setup compared to flat files.
*   **Query Implementation:** Connect to the SQLite DB, execute SQL `SELECT` queries with `WHERE` clauses (e.g., `WHERE tags LIKE '%"vision"%'` if storing tags as JSON, or join with a tags table).

## 4. Chosen Approach: Option D (SQLite + Blobs)

We will proceed with **Option D**. This involves using a central SQLite database (`paper_metadata.db`) to store metadata and storing the extracted text content in separate files (`blobs/<unique_id>.txt`) within a designated directory (`processed_pdfs/`). This approach leverages the built-in `sqlite3` module, provides robust querying capabilities, and handles data integrity well. While the database file itself isn't ideal for Git diffs, the separation of large text blobs helps mitigate this.


## 5. Implementation Plan

This plan integrates the PDF processing and storage into the existing agent structure (`src/main.py`, `src/tools.py`, etc.) in a modular way.

**Phase 1: Database Setup [✅ COMPLETE]**

*   **Status:** The core database module (`src/database.py`) has been created with functions for connection management, table creation (`init_db`), adding minimal paper records (`add_minimal_paper`), and updating specific fields (`update_paper_field`). Unit tests (`tests/test_database.py`) have been written and are passing.
*   **Implemented Schema (`papers` table in `src/database.py` - fields used by `/pdf` command):**
    *   `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
    *   `source_filename` (TEXT)
    *   `arxiv_id` (TEXT UNIQUE) - *Optionally added by `/pdf` command if provided as argument.* 
    *   `blob_path` (TEXT) - Relative path to the saved text blob (e.g., `paper_<id>_text.txt`).
    *   `status` (TEXT DEFAULT 'pending') - Updated by `/pdf` command (e.g., 'complete', 'error_process', 'error_blob').
    *   `added_date` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    *   `updated_date` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP) - Tracks local record updates.
    *   *Note: Fields like `title`, `authors`, `summary`, `source_pdf_url`, `publication_date`, `last_updated_date`, `categories`, `notes` exist in the schema but are **not** populated by the basic `/pdf` command.* 

**Phase 1.5: Core Saving Logic Integration [✅ COMPLETE]**

1.  **Database Functions (`src/database.py`):** Core functions (`get_db_connection`, `add_minimal_paper`, `update_paper_field`) are implemented and used directly.
2.  **Text Blob Saving Function (`src/tools.py`):** `save_text_blob(blob_full_path, text_content)` implemented and working.
3.  **Integration into `/pdf` Command (`src/main.py`):**
    *   The `/pdf` command handler (`_handle_pdf_command`) now orchestrates the process:
        *   Connects to the database.
        *   Calls `database.add_minimal_paper` to create a record, getting the `paper_id`.
        *   Optionally updates `arxiv_id` if provided.
        *   Calls `tools.extract_text_from_pdf_gemini` (which correctly returns the extracted text content).
        *   Calls `tools.save_text_blob` to save the text to `<blob_dir>/paper_<id>_text.txt`.
        *   Calls `database.update_paper_field` multiple times to update `blob_path` and `status`.
        *   Provides feedback to the user.
    *   **Gemini Integration:** The `extract_text_from_pdf_gemini` function in `tools.py` was successfully refactored to use the `google-genai` SDK (including `client.files.upload` and `client.models.generate_content`) and returns the extracted text. Manual integration tests confirmed this is working.

**Phase 2: Optional ArXiv API Integration [▶️ NEXT STEP]**

*Goal: Implement functionality to fetch metadata for a given ArXiv ID using the official API.* 

1.  **Install Dependency:** Add the `arxiv` library to the project's dependencies (e.g., in `requirements.txt` or `pyproject.toml`) and install it (`pip install arxiv`).
2.  **Create ArXiv Service Module (`src/arxiv_service.py`):**
    *   Implement `fetch_paper_metadata(arxiv_id)`:
        *   Use `arxiv.Client().results(id_list=[arxiv_id])`.
        *   Parse the `arxiv.Result` object (title, authors, summary, categories, published, updated, pdf_url).
        *   Handle errors (not found, API issues).
        *   Return a dictionary of metadata or `None`.
3.  **Create Basic Tests (`tests/test_arxiv_service.py`):**
    *   Use `pytest` and `unittest.mock` to mock `arxiv.Client`.
    *   Test successful data extraction and error handling for `fetch_paper_metadata`.

*(The `search_papers` function and its tests can be deferred to a later part of this phase if desired).* 

**Phase 3: Paper Downloading and Database Integration [Planned]**

*Goal: Add a command to download the PDF from ArXiv using the fetched metadata and update the database.* 

1.  **Implement `/download_arxiv <arxiv_id>` Command (`src/main.py`):**
    *   Call `arxiv_service.fetch_paper_metadata`.
    *   If metadata is found:
        *   Extract the `pdf_url`.
        *   Define the local `download_path` using a configured directory and a consistent naming scheme (e.g., `<PAPER_DOWNLOAD_DIR>/<arxiv_id>.pdf`).
        *   Use the `requests` library (`pip install requests`) to download the PDF from `pdf_url` and save it to `download_path`. Handle potential download errors.
        *   If download succeeds:
            *   Prepare a paper data dictionary using the metadata fetched from arXiv (title, authors, summary, categories, publication_date, last_updated_date, source_pdf_url) and add arxiv_id, download_path, and set status to 'downloaded'.
            *   Establish a database connection (`conn = database.get_db_connection()`).
            *   Call `database.add_paper(conn, paper_data)`. Handle potential integrity errors if the paper already exists.
            *   Close the connection (`database.close_db_connection(conn)`).
            *   Report success to the user (e.g., "Successfully downloaded and recorded arXiv: `<arxiv_id>` - `<title>`").
        *   If download fails, report the error.

2.  **Add Configuration (`src/config.yaml`):**
    *   Add `PAPER_DOWNLOAD_DIR: "downloaded_papers/"` (or similar). Ensure this directory exists or is created.
    *   Update `load_config` in `main.py` to load this new path.

**Phase 4: Basic Querying and Retrieval**

(This corresponds to the original Phase 2)
Note: All sqlite db-related slash commands should start with [db_](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:17:0-26:16).

1.  **Implement Query Function (`src/database.py`):**
    *   Enhance `db_find_papers_by_tag(tag)`: Returns a list of paper metadata dictionaries matching the tag. Handles JSON parsing for the `tags` column. *(Requires tags to be added first, see Phase 5)*
    *   Add `db_find_papers(criteria)`: More general query function (e.g., by author, title keyword).
2.  **Create `/db_find_papers` Command (`src/main.py`):**
    *   Add a new command handler (e.g., `_handle_find_papers`).
    *   Parse arguments (e.g., `--tag <tag>`, `--author <author>`, `--title_contains <keyword>`, `--status <status>`).
    *   Call the appropriate query function from [database.py](cci:7://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:0:0-0:0).
    *   Format and display the results (e.g., list of IDs, titles, status, tags).
3.  **Create `/db_load_paper` Command (`src/main.py`):**
    *   Add a new command handler (e.g., `_handle_load_paper`).
    *   Takes an `arxiv_id` or `paper_id`.
    *   Calls `database.get_paper_by_arxiv_id` or `database.get_paper_by_id`.
    *   If found, display key metadata (Title, Authors, Abstract, Status, Tags, Notes).
    *   Update the `last_accessed` timestamp field (which needs to be added to the schema first).

## History Management & Token Counting

**Current Situation:**

1.  **Gemini's Internal History:** The `self.chat` object (`google.genai.chats.Chat`) manages conversation history internally for context awareness during `send_message` calls. PDF context is injected by modifying the *next* message sent to this object.
2.  **Manual History (`self.conversation_history`):** A separate list (`MutableSequence[types.Content]`) maintained within `CodeAgent` specifically for token counting (`self.model.count_tokens(self.conversation_history)`).
3.  **The Problem:** The `_update_token_count` function currently attempts to access `self.chat.history` to get the history for token counting. This fails with `AttributeError: 'Chat' object has no attribute 'history'`. While the official documentation for `google.generativeai.Chat` *does* specify a `history` attribute, we are encountering this error in practice after the response stream is processed.

**Recommended Approach:**

Given the unreliability (in our current execution context) of accessing `self.chat.history`, we should rely **solely** on our manually managed `self.conversation_history` for token counting. This gives us direct control and avoids relying on potentially inconsistent internal state or access patterns of the `Chat` object.

**Implementation Plan (History Fix):**

1.  **Modify `_process_response_stream`:** After successfully accumulating the full model response (`full_response_text`), create a `types.Content` object for the model's response (`role="model"`) and append it to `self.conversation_history`.
2.  **Modify `_update_token_count`:** Change the function to use `self.conversation_history` instead of `self.chat.history` when calling `self.model.count_tokens()`.
3.  **Verification:** Test the `/pdf` command followed by a query, ensuring the token count updates correctly after both the user message (with PDF context) and the model's response.

## Prompt Management Feature (`/prompt`)

**Goal:** Allow users to quickly load and use predefined prompts stored in files.

**Requirements:**

1.  **Prompt Storage:** Create a `prompts/` directory in the project root.
2.  **Prompt Files:** Store prompts as plain text (`.txt`) or markdown (`.md`) files within `prompts/`. The filename (without extension) will serve as the prompt name (e.g., `summarize.txt` -> prompt name `summarize`).
3.  **Slash Command:** Implement `/prompt <prompt_name>`.
4.  **Autocompletion:** Provide nested autocompletion for `/prompt`, showing available prompt names from the `prompts/` directory.
5.  **Functionality:** When `/prompt <name>` is used:
    *   Load the content of `prompts/<name>.txt` or `prompts/<name>.md`.
    *   Display the loaded prompt content to the user (optional, maybe truncated).
    *   Prepend the loaded prompt content to the *next* user message sent to the model (similar mechanism to PDF context injection using a `self.pending_prompt` variable).

**Implementation Plan (`/prompt` command):**

1.  **Create Directory:** Add an empty `prompts/` directory to the repository.
2.  **Add Sample Prompts:** Create 1-2 sample prompt files (e.g., `prompts/summarize.txt`, `prompts/explain_code.md`).
3.  **Modify `CodeAgent.__init__`:** Add `self.pending_prompt: Optional[str] = None`.
4.  **Implement Prompt Loading Logic:** Create a helper function `_load_prompt(prompt_name: str) -> Optional[str]` that:
    *   Constructs paths to `prompts/<prompt_name>.txt` and `prompts/<prompt_name>.md`.
    *   Checks if either file exists.
    *   Reads and returns the content of the first one found, or `None` if neither exists.
5.  **Implement Prompt Listing Logic:** Create a helper function `_list_available_prompts() -> List[str]` that:
    *   Lists files in `prompts/`.
    *   Filters for `.txt` and `.md` extensions.
    *   Returns a list of filenames without extensions.
6.  **Update `_build_completer`:** Modify the `NestedCompleter` definition:
    *   Add `/prompt` as a top-level command.
    *   For `/prompt`, use a dictionary where keys are dynamically generated by `_list_available_prompts()` and values are `None` (or another Completer if further nesting is needed later).
7.  **Modify `_handle_command`:**
    *   Add a case for `user_input.startswith('/prompt ')`.
    *   Parse the `<prompt_name>`.
    *   Call `_load_prompt()`.
    *   If successful, store the loaded content in `self.pending_prompt` and print a confirmation message.
    *   If not found, print an error message listing available prompts using `_list_available_prompts()`.
8.  **Modify `start_interaction` Loop:**
    *   *Before* checking for `self.pending_pdf_context`, check for `self.pending_prompt`.
    *   If `self.pending_prompt` exists, prepend it to `message_to_send` (e.g., `f"PROMPT:\n---\n{self.pending_prompt}\n---\n\nUSER QUERY:\n{user_input}"`), print a confirmation, and clear `self.pending_prompt`.
    *   Ensure the PDF context logic runs *after* the prompt logic if both are pending (prompt takes precedence or they combine appropriately).

## 6. Configuration

*   Update `src/config.yaml`:
    ```yaml
    # PAPER_DB_PATH: "processed_pdfs/paper_metadata.db" # Keep if different from default
    # PAPER_BLOBS_DIR: "processed_pdfs/blobs/" # Keep for future text blob storage
    PAPER_DOWNLOAD_DIR: "downloaded_papers/" # Add this
    ```
*   Ensure `load_config` in `main.py` reads these values.
*   Ensure the `PAPER_DOWNLOAD_DIR` directory is created if it doesn't exist upon application start or before the first download attempt.

## 7. Future Enhancements

*   More sophisticated querying (semantic search on text blobs if implemented).
*   Automatic tagging suggestions.
*   UI for browsing/managing papers.
*   Duplicate detection based on title/content hashing (especially for non-arXiv PDFs).
*   Refactoring download/processing logic to be reusable by both `/download_arxiv` and potentially a modified `/pdf` command later.
*   Implement `last_accessed` timestamp updates.