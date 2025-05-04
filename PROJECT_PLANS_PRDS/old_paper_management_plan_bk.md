# Plan: Persistent Storage and Querying for Processed PDFs


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

This plan integrates the PDF processing and storage into the existing agent structure (`src/main.py`, `src/tools.py`) in a modular way.

**Phase 1: Database Setup and Core Saving Logic**

1.  **Create Database Helper Module (`src/db_utils.py`):**
    *   Define the SQLite database path (e.g., `PROJECT_ROOT/processed_pdfs/paper_metadata.db`).
    *   Define the blobs directory path (e.g., `PROJECT_ROOT/processed_pdfs/blobs/`).
    *   Implement `init_db()`: Creates the database file, the `blobs` directory, and the `papers` table if they don't exist.
        *   **Schema (`papers` table):**
            *   `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
            *   `unique_paper_id` (TEXT UNIQUE NOT NULL) - e.g., UUID or timestamp-based slug.
            *   `source_filename` (TEXT) - Original filename.
            *   `blob_path` (TEXT NOT NULL) - Relative path to the text blob file (e.g., `blobs/<unique_paper_id>.txt`).
            *   `processed_timestamp` (TEXT ISO8601 NOT NULL).
            *   `title` (TEXT).
            *   `authors` (TEXT) - Store as JSON string list `["Author 1", "Author 2"]`.
            *   `abstract` (TEXT).
            *   `arxiv_id` (TEXT) - Store just the ID (e.g., `1706.03762`).
            *   `tags` (TEXT) - Store as JSON string list `["tag1", "tag2"]`.
            *   `notes` (TEXT).
            *   `last_accessed` (TEXT ISO8601).
    *   Implement `add_paper(metadata)`: Inserts a new paper record. Takes a dictionary of metadata. Generates `unique_paper_id` and `blob_path`. Returns the `id` or `unique_paper_id`.
    *   Implement `get_paper_by_fuzzy_string_match(string)`: Retrieves a paper record.
    *   Implement `get_paper_by_id(unique_paper_id)`: Retrieves a paper record.
    *   Implement `update_paper(unique_paper_id, updates)`: Updates specific fields for a paper.
    *   Implement basic query functions (e.g., `find_papers_by_tag(tag)`, `find_papers_by_abstract_keyword(keyword)`, etc.).

2.  **Create Text Blob Saving Function (`src/tools.py`):**
    *   Implement `save_text_blob(unique_id, text_content)`: Saves the extracted text to `processed_pdfs/blobs/<unique_id>.txt`. Uses the path defined in `db_utils.py`.

3.  **Integrate into `/pdf` Command (`src/main.py`):**
    *   Modify the `/pdf` command handler (`_handle_pdf_command` or similar).
    *   After successful text extraction using `upload_pdf_for_gemini`:
        *   Import functions from `db_utils` and `tools`.
        *   Call `init_db()` (it's safe to call repeatedly).
        *   Generate a `unique_paper_id` (should be a unique hash of the source filename).
        *   Call `save_text_blob(unique_paper_id, extracted_text)`.
        *   Prepare a basic `metadata` dictionary:
            *   `unique_paper_id`: The generated ID.
            *   `source_filename`: The input filename.
            *   `blob_path`: Construct the relative path (e.g., `f"blobs/{unique_paper_id}.txt"`).
            *   `processed_timestamp`: Use `get_current_date_and_time`.
            *   `last_accessed`: Same as `processed_timestamp`.
            *   `tags`: `[]` (empty list initially).
            *   Other fields initially `None` or extracted if possible (basic title/author extraction can be added later).
        *   Call `add_paper(metadata)`.
        *   Provide feedback to the user: "Processed and saved `<filename>` with ID `<unique_paper_id>`."
    *   **Important:** Ensure the `upload_pdf_for_gemini` in `tools.py` is modified or wrapped to return the *extracted text content* itself, not just the `File` object or status message. Currently, it seems focused only on the upload/processing status and doesn't explicitly fetch/return the text. This might require adding a separate call to Gemini using the uploaded file reference to get the text. Check the google.genai docs for more info here: https://googleapis.github.io/python-genai/index.html and here: https://ai.google.dev/gemini-api/docs/document-processing?lang=python . Make sure to read the relevant documentation on the web before attempting to implement.


**Phase 2: Basic Querying and Retrieval**

Note: All sqlite db-related slash commands should start with `db_`.

1.  **Implement Query Function (`src/db_utils.py`):**
    *   Enhance `db_find_papers_by_tag(tag)`: Returns a list of paper metadata dictionaries matching the tag. Handles JSON parsing for the `tags` column.
    *   Add `db_find_papers(criteria)`: More general query function (e.g., by author, title keyword).
2.  **Create `/db_find_papers` Command (`src/main.py`):**
    *   Add a new command handler (e.g., `_handle_find_papers`).
    *   Parse arguments (e.g., `--tag <tag>`, `--author <author>`).
    *   Call the appropriate query function from `db_utils.py`.
    *   Format and display the results (e.g., list of IDs, titles, tags).
3.  **Create `/db_load_paper` Command (`src/main.py`):**
    *   Add a new command handler (e.g., `_handle_load_paper`).
    *   Takes a `unique_paper_id` as argument.
    *   Calls `get_paper_by_id` from `db_utils.py`.
    *   Reads the text content from the `blob_path` using `read_file` (or a new helper).
    *   Decide how to load into context:
        *   Option 1: Add full text as a user message ("Loaded paper <ID>: <title>\n\n<full_text>").
        *   Option 2: Add metadata summary and maybe abstract ("Loaded paper <ID>: <title>\nAuthors: ...\nAbstract: ..."). Offer a follow-up to load full text.
    *   Update the `last_accessed` timestamp for the paper using `update_paper`.

**Phase 3: Metadata Extraction and Editing**

1.  **Metadata Extraction (`src/tools.py` or `src/metadata_extractor.py`):**
    *   Implement `extract_metadata_from_text(text_content)`: Uses heuristics or potentially a simpler LLM call to extract Title, Authors, Abstract from the first page(s) of the text.
    *   Modify the `/pdf` command handler (`src/main.py`) to call this after getting the text and before `add_paper`, populating more fields in the initial `metadata` dictionary.
2.  **Implement Update Functions (`src/db_utils.py`):**
    *   Refine `update_paper` to handle specific fields like adding/removing tags (requires reading current tags, modifying the list, and writing back the JSON string).
3.  **Create `/tag_paper` Command (`src/main.py`):**
    *   Add handler `_handle_tag_paper`.
    *   Takes `unique_paper_id`, `--add <tag>`, `--remove <tag>`.
    *   Calls `update_paper` via a helper in `db_utils` to modify the `tags` list.
4.  **Create `/add_note` Command (`src/main.py`):**
    *   Add handler `_handle_add_note`.
    *   Takes `unique_paper_id` and the note text.
    *   Calls `update_paper` to set/append the `notes` field.

**Phase 4: Integration with ArXiv Search**

1.  **Modify `/find_arxiv_papers` (`src/tools.py`):**
    *   When displaying results, check if a paper with the same ArXiv ID already exists in the local DB using a new `get_paper_by_arxiv_id` function in `db_utils.py`. Indicate if it's already downloaded/processed.
2.  **Add `/download_arxiv <arxiv_id>` Command (`src/main.py`):**
    *   Fetches paper details (including PDF link) from ArXiv API using the ID.
    *   Downloads the PDF to a temporary location or the designated `PDFS_TO_CHAT_WITH_DIRECTORY`.
    *   Calls the existing `/pdf` command logic (or refactors it) to process the downloaded file.
    *   Ensures the `arxiv_id` is stored correctly during the `add_paper` step.

## 6. Configuration

*   Add new entries to `src/config.yaml` for the database and blob directory paths:
    ```yaml
    PAPER_DB_PATH: "processed_pdfs/paper_metadata.db"
    PAPER_BLOBS_DIR: "processed_pdfs/blobs/"
    ```
*   Update `load_config` in `main.py` and pass these paths to `db_utils` functions or classes.

## 7. Future Enhancements

*   More sophisticated querying (semantic search on text blobs).
*   Automatic tagging suggestions.
*   UI for browsing/managing papers.
*   Duplicate detection based on title/content hashing.
