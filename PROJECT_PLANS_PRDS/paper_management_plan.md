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
*   **Metadata Storage:** Store structured metadata associated with each PDF.

    *Baseline (MVP) fields*  
    *   `source_filename`
    *   `processed_timestamp`
    *   `blob_path`  <!-- path to the raw text file -->

    *Optional / future fields* (TBD)  
    *   `title`, `authors`, `abstract`, `references`
    *   `arxiv_id` / `arxiv_url`
    *   `github_repo`
    *   `last_updated`, `last_accessed`
    *   `tags`
    *   `notes`
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

*   **Status:** The core database module ([src/database.py](src/database.py)) has been created with functions for connection management, table creation, and CRUD operations. Unit tests ([tests/test_database.py](tests/test_database.py)) have been written and are passing.
*   **Implemented Schema ([papers](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:136:0-140:27) table in `src/database.py`):**
    *   [id](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:117:0-129:79) (INTEGER PRIMARY KEY AUTOINCREMENT)
    *   [arxiv_id](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:117:0-129:79) (TEXT UNIQUE NOT NULL) - e.g., '2310.06825v1'
    *   `title` (TEXT NOT NULL)
    *   `authors` (TEXT) - Stored as JSON string list
    *   `summary` (TEXT)
    *   `source_pdf_url` (TEXT UNIQUE) - Direct link to arXiv PDF
    *   `download_path` (TEXT) - Local path where PDF is saved
    *   `publication_date` (TIMESTAMP)
    *   `last_updated_date` (TIMESTAMP) - From arXiv metadata
    *   `categories` (TEXT) - Stored as JSON string list
    *   [status](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:152:0-166:42) (TEXT DEFAULT 'pending') - e.g., pending, downloaded, processing, summarized, complete, error
    *   `notes` (TEXT)
    *   `added_date` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    *   `updated_date` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP) - Tracks local record updates


**Phase 1.5: Database Setup and Core Saving Logic part 2**

1.  **Create Database Helper Module (`src/db_utils.py`):**
    *   Define the SQLite database path (e.g., `PROJECT_ROOT/processed_pdfs/paper_metadata.db`).
    *   Define the blobs directory path (e.g., `PROJECT_ROOT/processed_pdfs/blobs/`).
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

**Phase 2 (TBD): Optional ArXiv API Integration**

1.  **Create ArXiv Service Module (`src/arxiv_service.py`):**
    *   Install the [arxiv](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:117:0-129:79) library (`pip install arxiv`).
    * Basic readme with how to use instructions is here: https://github.com/lukasschwab/arxiv.py/blob/master/README.md - need to revisit this to make sure we're implementing correctly
    *   Implement `fetch_paper_metadata(arxiv_id)`:
        *   Uses `arxiv.Client().results(id_list=[arxiv_id])` to query the API.
        *   Parses the `arxiv.Result` object to extract key metadata: title, authors (list of strings), summary, categories (list of strings), published (datetime), updated (datetime), pdf_url.
        *   Handles cases where the paper is not found or the API call fails.
        *   Returns a dictionary containing the extracted metadata or `None` on failure.
    *   Implement `search_papers(query, max_results=10)`:
        *   Uses `arxiv.Client().results(Search(query=query, max_results=max_results, sort_by=SortCriterion.Relevance))`.
        *   Returns a list of dictionaries, each containing metadata for a found paper (similar structure to `fetch_paper_metadata` output).

2.  **Create Tests (`tests/test_arxiv_service.py`):**
    *   Use `pytest` and `unittest.mock`.
    *   Mock `arxiv.Client` and its `results` method.
    *   Create mock `arxiv.Result` objects to simulate API responses (success, paper not found, multiple results for search).
    *   Test `fetch_paper_metadata` for successful data extraction and error handling.
    *   Test `search_papers` for correct parsing of multiple results.

**Phase 3: Paper Downloading and Database Integration**

1.  **Implement `/download_arxiv <arxiv_id>` Command (`src/main.py`):**
    *   Create a new command handler function (e.g., `_handle_download_arxiv`).
    *   Import `arxiv_service` and `database`. Load necessary config (e.g., download directory).
    *   Call `arxiv_service.fetch_paper_metadata(arxiv_id)`. If `None`, inform the user the paper wasn't found.
    *   If metadata is found:
        *   Extract the `pdf_url`.
        *   Define the local `download_path` using a configured directory and a consistent naming scheme (e.g., `<PAPER_DOWNLOAD_DIR>/<arxiv_id>.pdf`).
        *   Use the `requests` library (`pip install requests`) to download the PDF from `pdf_url` and save it to `download_path`. Handle potential download errors.
        *   If download succeeds:
            *   Prepare a [paper_data](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:43:0-56:5) dictionary using the metadata fetched from arXiv (title, authors, summary, categories, publication_date, last_updated_date, source_pdf_url) and add [arxiv_id](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:117:0-129:79), `download_path`, and set [status](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:152:0-166:42) to `'downloaded'`.
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
    *   Takes an [arxiv_id](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:117:0-129:79) as argument.
    *   Calls [get_paper_by_arxiv_id](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:117:0-129:79) from [database.py](cci:7://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:0:0-0:0).
    *   If found, display key metadata (Title, Authors, Abstract, Status, Tags, Notes).
    *   Update the `last_accessed` timestamp for the paper using [update_paper_field](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:168:0-196:48). *(Need to implement `last_accessed` update logic)*.
    *   *(Loading actual text content will depend on future text extraction implementation)*.

**Phase 5: Metadata Extraction and Editing**

(This corresponds to the original Phase 3)

1.  **Metadata Extraction (`src/tools.py` or `src/metadata_extractor.py`):**
    *   *(Deferred until text extraction from PDF is implemented)* Implement `extract_metadata_from_text(text_content)`: Uses heuristics or potentially a simpler LLM call to extract Title, Authors, Abstract from the first page(s) of the text.
    *   *(Deferred)* Modify the processing workflow (whether it's `/pdf` or `/download_arxiv` post-processing) to call this and potentially update the DB record.
2.  **Implement Update Functions (`src/database.py`):**
    *   Refine [update_paper_field](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:168:0-196:48) or add specific helpers (e.g., `update_paper_tags`) to handle modifying list-based fields like `tags`. Requires reading current tags, modifying the list, and writing back the JSON string.
3.  **Create `/db_tag_paper` Command (`src/main.py`):**
    *   Add handler `_handle_tag_paper`.
    *   Takes [arxiv_id](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:117:0-129:79), `--add <tag>`, `--remove <tag>`, or `--set <tag1>,<tag2>`.
    *   Calls [update_paper_field](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:168:0-196:48) (or a dedicated tag helper) to modify the `tags` list in the database.
4.  **Create `/db_add_note` Command (`src/main.py`):**
    *   Add handler `_handle_add_note`.
    *   Takes [arxiv_id](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:117:0-129:79) and the note text (potentially multi-line).
    *   Calls [update_paper_field](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:168:0-196:48) to set or replace the `notes` field.

**Phase 6: Integration with ArXiv Search**

(This corresponds to the original Phase 4)

1.  **Modify `/find_arxiv_papers` (`src/tools.py`):**
    *   When displaying results from the [arxiv](cci:1://file:///Volumes/bdrive/repos/coding-agent/tests/test_database.py:117:0-129:79) library search, for each result, call `database.get_paper_by_arxiv_id` to check if it's already in the local DB.
    *   Append status information to the display (e.g., `[Status: Downloaded]`, `[Status: Pending]`, `[Not in DB]`).
2.  **Enhance `/download_arxiv <arxiv_id>` Command (`src/main.py`):**
    *   Before attempting download, check if the paper already exists using `database.get_paper_by_arxiv_id`. If it exists, inform the user and perhaps show the current status/metadata instead of re-downloading. Allow an option to force re-download/overwrite.

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