# Detailed Implementation Plan (v0.1) - Multi-Agent Research Synthesizer

This plan elaborates on the 3-week roadmap outlined in `PLANS/integrated_plan.md`, providing more granular steps.

**Status Legend:** `[X]` = Done, `[-]` = Partially Done / In Progress, `[ ]` = Not Started

---

## Phase 1: Foundations - Memory & Examples **[COMPLETED]**

**Goal:** Establish the core memory infrastructure and demonstrate its usage.

**Tasks:**

1.  **[-] Project Setup & CI:**
    *   `[X]` Confirm/Set up Python environment (using `poetry` and `.venv`).
    *   `[X]` Update `pyproject.toml` with initial dependencies (`anyio`, `sqlite3`, `sentence-transformers`, `numpy`, `filelock`, `pydantic`, `pdfplumber`, `pytest`, `ruff`, etc.).
    *   `[ ]` Initialize `pre-commit` with hooks for `ruff` (linting + formatting).
    *   `[ ]` Set up basic GitHub Actions workflow for linting (`ruff check .`) and running tests (`pytest`).
    *   `[X]` Create the core directory structure (`src/memory`, `tests/memory`, `examples/memory`) and `__init__.py` files.

2.  **[X] Memory System - Core Definition:**
    *   `[X]` Define the `MemoryDoc` dataclass in `src/memory/adapter.py`.
        *   *Annotation:* Structure refined during implementation.
    *   *(Conceptual)* Define the `MemoryAdapter` abstract base class idea (though not formally implemented as ABC yet).

3.  **[X] Memory System - Embedding Manager:**
    *   `[X]` In `src/memory/embedding_manager.py`, create `EmbeddingManager` class.
    *   `[X]` Implement `__init__` (loads sentence transformer model, sets up embedding directory path).
    *   `[X]` Implement `generate_embedding(text) -> np.ndarray`.
    *   `[X]` Implement `save_embedding(embedding, doc_id)`.
    *   `[X]` Implement `load_embedding(doc_id)`.
    *   `[X]` Implement `find_similar_embeddings(query_embedding, doc_ids, k)`.
    *   `[X]` Add basic unit tests in `tests/memory/test_embedding_manager.py` (covered implicitly via adapter tests).

4.  **[X] Memory System - Hybrid SQLite Adapter Implementation:**
    *   `[X]` In `src/memory/hybrid_sqlite.py`, create `HybridSQLiteAdapter`.
    *   `[X]` Implement the `__init__` method (takes DB path, embedding dir path; instantiates `EmbeddingManager`; connects to DB; calls `_setup_schema`).
        *   *Annotation:* Uses a single `sqlite3` connection. Likely thread-safe for reads, but writes are serialized by the single connection. High-concurrency writes may require explicit locking or a queue later.
    *   `[X]` Implement `_setup_schema` (creates `memories` table and `memories_fts` virtual table with synchronization triggers).
        *   *Annotation:* FTS triggers (`AFTER INSERT/DELETE/UPDATE`) established for reliable sync.
    *   `[X]` Implement `add` method:
        *   Generates UUID if needed.
        *   Generates embedding using `EmbeddingManager`.
        *   Saves embedding to file using `EmbeddingManager`.
        *   Inserts metadata (including embedding path, tags, source agent) into `memories` table.
    *   `[X]` Implement `query` method (FTS keyword search):
        *   Uses `MATCH` on `memories_fts` table.
        *   Applies filtering based on `filter_tags` and `filter_source_agents` via JOINs or subqueries.
        *   Returns list of `MemoryDoc` objects (without embeddings loaded).
    *   `[X]` Implement `semantic_query` method (Vector similarity search):
        *   Generates query embedding using `EmbeddingManager`.
        *   Applies pre-filtering based on `filter_tags` and `filter_source_agents` to get candidate `doc_ids`.
        *   Uses `EmbeddingManager.find_similar_embeddings` on candidate embeddings.
        *   Retrieves full `MemoryDoc` details for top K results.
        *   Returns list of `MemoryDoc` objects.
    *   `[X]` Add comprehensive unit tests in `tests/memory/test_hybrid_sqlite.py` covering: schema, add, FTS query (with/without filters), semantic query (with/without filters), edge cases.

5.  **[DEFERRED] Memory System - MemoryService (Singleton):**
    *   `[ ]` *Decision: Deferred singleton wrapper. Agents will use `HybridSQLiteAdapter` instances directly for now.* Consider re-introducing if complex state management or write queueing becomes necessary later.

6.  **[ ] Agent Skeletons & CLI:**
    *   `[ ]` Create `src/core/scheduler.py` with `TaskScheduler` stub.
    *   `[ ]` Create `src/core/agents/base.py` with `BaseAgent` abstract class.
    *   `[ ]` Create `src/core/agents/ingestor.py`, `summarizer.py`, etc. with skeleton classes.
    *   `[ ]` Create `src/cli.py` with basic `click` setup.
    *   `[ ]` Add basic unit tests for agent skeletons.

7.  **[X] Example Usage Scripts:**
    *   `[X]` Create `examples/memory/example_basic_usage.py` demonstrating init, add, basic FTS query, basic semantic query.
    *   `[X]` Create `examples/memory/example_advanced_querying.py` demonstrating adding diverse docs and using filters with both query types.
    *   `[X]` Create `examples/README.md` explaining the examples and how to run them.

---

## Phase 2: Agents & Concurrency **[NEXT UP]**

**Goal:** Implement core agent logic, the concurrency wrapper, and integrate them.

**Tasks:**

1.  **[X] Memory System - RRF Fusion:**
    *   `[X]` Update `HybridSQLiteAdapter` to add a new method, e.g., `hybrid_query()`.
    *   `[X]` This method performs both FTS (`query`) and semantic (`semantic_query`) searches internally.
    *   `[X]` Implement Reciprocal Rank Fusion (RRF) logic to combine the ranked results.
    *   `[X]` Add configuration for RRF weighting (`rrf_k` parameter).
    *   `[X]` Add unit tests for RRF implementation.
    *   `[ ]` Benchmark query performance (latency). (*Deferred - Low Priority*)

2.  **[X] Concurrency - Scheduler Implementation:**
    *   `[X]` Define `TaskScheduler` class in `src/core/scheduler.py`.
    *   `[X]` Implement `__init__(max_concurrent_tasks: int)`.
    *   `[X]` Store tasks internally (e.g., a list of agent instances).
    *   `[X]` Implement `add_task(agent_instance: BaseAgent)` method.
    *   `[X]` Implement `async run()` method which internally calls `_run_internal` (merged logic).
    *   `[X]` Implement `async _run_internal()` (merged into `run`): 
        *   `[X]` Initialize `anyio.Semaphore(self.max_concurrent_tasks)`.
        *   `[X]` Use `anyio.create_task_group()`.
        *   `[X]` Iterate tasks, calling `task_group.start_soon(self._run_agent_task, agent_instance, semaphore)`.
    *   `[X]` Implement private `async _run_agent_task(agent_instance, semaphore)`:
        *   `[X]` Acquire the semaphore: `async with semaphore:`.
        *   `[X]` Log agent start.
        *   `[X]` Execute agent's main logic (`await agent_instance.run()`).
            *   *Decision:* Assumed agent `run` methods are `async`. Checked `BaseAgent` definition.
        *   `[X]` Log agent completion or error.
        *   `[X]` Use `try...except` for agent error logging; `async with` handles semaphore release.
    *   `[X]` Add tests for `TaskScheduler` in `tests/core/test_scheduler.py`:
        *   `[X]` Test task addition & initialization.
        *   `[X]` Test basic execution flow with mock agents.
        *   `[X]` Test concurrency limiting using mock agents with delays and callbacks.
        *   `[X]` Test error handling (scheduler continues if one agent fails).
    *   `[X]` Add `trio` dependency for `pytest-anyio` cross-backend testing.
    *   *Notes:* Implemented using `anyio` library for async operations. Handles concurrency with a semaphore and manages task execution within a task group. Basic error handling logs agent exceptions but allows the scheduler to continue.

3.  **[X] Concurrency - Adapter Usage Review:**
    *   `[X]` **Analyze `HybridSQLiteAdapter` Thread Safety:** Analysis complete. The current adapter uses `check_same_thread=False`, making the single `sqlite3` connection unsafe for concurrent writes if the adapter instance is shared across tasks/threads.
    *   `[X]` **Consider Connection Strategy & Decision:** **Decision Made: Each agent task will receive its own `HybridSQLiteAdapter` instance.** This avoids sharing the unsafe connection object. SQLite's built-in file-level locking will handle concurrency between these separate connections to the same database file. This requires passing configuration (like DB path) to agents or using an adapter factory during agent initialization.
    *   `[-]` ~~Evaluate Async Adapter (`aiosqlite`):~~ Not pursued as the primary safety issue is resolved by instance-per-task strategy.
    *   `[X]` **Decision Documentation:** Decision documented here and in the high-level plan.

4.  **[X] Core Agents (Ingestor, Summarizer, Synthesizer):**
    *   `[X]` Implement `IngestorAgent`:
        *   `[X]` Basic structure and `run` method defined.
        *   `[X]` Initial implementation using `pdfplumber` (later refactored).
        *   `[X]` Refactored to use `pymupdf` strategy internally.
        *   `[X]` Implemented `_process_pdf_pymupdf` helper method.
        *   `[X]` Integrated `_chunk_text` (mocked in tests, needs real implementation).
        *   `[X]` Integrated memory adapter `add` call (mocked in tests, needs real adapter integration).
        *   `[X]` Unit tests added covering pymupdf, mocked mistral_ocr, mixed paths, and error handling.
        *   `[ ]` **Define Configuration:** Add `INGESTION_STRATEGY` setting in `src/config.py` (options: `pymupdf`, `mistral_ocr`, `gemini_native`). Also add required API keys (`MISTRAL_API_KEY`, `GEMINI_API_KEY`) handling (e.g., via environment variables or config). Read this config within the agent.
        *   `[ ]` **Input Handling:** Agent accepts PDF sources (e.g., file paths provided via constructor/config). Needs robust handling of these paths (checking existence, permissions).
        *   `[ ]` **Strategy Selection:** Implement logic within the agent's `run` method to select the processing method based on the loaded `INGESTION_STRATEGY` configuration.
            *   **Method 0: PyMuPDF (`pymupdf`)**
                *   `[X]` Core text extraction logic implemented in `_process_pdf_pymupdf`.
                *   `[ ]` Needs integration with configuration and proper error handling within the main `run` flow.
            *   **Method 1: Mistral OCR API (`mistral_ocr`)**
                *   `[ ]` Reference `src/arxiv_ocr.py` or similar examples.
                *   `[ ]` Initialize `MistralClient` (requires `MISTRAL_API_KEY` from config/env).
                *   `[ ]` Prepare PDF content (read bytes from file path).
                *   `[ ]` Call `mistral_client.files.create_async()` to upload.
                *   `[ ]` Call `mistral_client.ocr.process_async()` using the returned file ID.
                *   `[ ]` Process the returned `OCRResponse` object to extract structured text content (likely iterating through `pages`).
                *   `[ ]` Handle potential errors (API key missing, API limits, network issues, file processing errors).
                *   `[ ]` Delete the uploaded file using `mistral_client.files.delete_async()`.
            *   **Method 2: Gemini Native File Upload & Extraction (`gemini_native`)**
                *   `[ ]` Reference `src/tools.py::upload_pdf_for_gemini` and related usage.
                *   `[ ]` Initialize `genai.Client` (requires `GOOGLE_API_KEY` from config/env).
                *   `[ ]` Use helper function (potentially adapted from `upload_pdf_for_gemini`) to upload the PDF via `client.files.upload()` and wait for `ACTIVE` state. This returns a `types.File` object.
                *   `[ ]` **Crucially:** Send a *separate* request to the Gemini chat model (`genai.GenerativeModel` or similar, *not* the file client) including the `types.File` object and a specific prompt instructing it to extract the full text content (e.g., "Extract the entire text content of the provided PDF, maintaining structure where possible.").
                *   `[ ]` Process the model's response (`response.text`) to get the extracted text.
                *   `[ ]` Handle file upload errors, processing timeouts, extraction errors, and potential API key issues.
                *   `[ ]` Consider file lifecycle management (deleting uploaded files after extraction using `client.files.delete()`).
        *   `[ ]` **Chunking Logic:** Implement a text chunking strategy (e.g., fixed size with overlap using `textwrap` or a more sophisticated sentence-boundary based approach) on the extracted text obtained from *any* of the selected methods. This should happen *after* text extraction.
        *   `[ ]` **Storage:** Use the *actual* `HybridSQLiteAdapter` instance (passed during agent initialization) to call `adapter.add(MemoryDoc(...))` for *each* text chunk, storing relevant metadata (source PDF path, chunk number, ingestion method used).
        *   `[ ]` **Error Handling:** Implement robust error handling throughout the `run` method for file operations, API interactions, chunking, and adapter usage. Log errors clearly.
    *   `[ ]` Implement `SummarizerAgent`: Use `HybridSQLiteAdapter.query/hybrid_query` to get chunks for one paper, generate a summary (initially maybe just concatenate chunks, later use a real LLM), use `HybridSQLiteAdapter.add` to store summary.
    *   `[ ]` Implement `SynthesizerAgent`: Use `HybridSQLiteAdapter.query/hybrid_query` to get summaries/chunks across papers, generate synthesis (initially mock/placeholder, later use a real LLM), use `HybridSQLiteAdapter.add`.
    *   `[X]` Ensure agents correctly instantiate/receive `HybridSQLiteAdapter` instances (Decision made: instance per agent).
    *   `[ ]` Add unit tests for these agents (mocking adapter/LLM).
        *   `[X]` Added tests for `IngestorAgent` covering `pymupdf`, `mistral_ocr` (mocked), error handling, mixed paths.
        *   `[ ]` Add tests for `SummarizerAgent`.
        *   `[ ]` Add tests for `SynthesizerAgent`.
5.  **[ ] CLI Integration:**
    *   `[ ]` Update `src/cli.py`'s `run-pipeline` command.
    *   `[ ]` Instantiate `TaskScheduler`.
    *   `[ ]` Instantiate required Agent classes (passing adapter instance).
    *   `[ ]` Submit agents to the scheduler and run it.

6.  **[ ] Integration Test:**
    *   `[ ]` Create `tests/integration/test_pipeline.py`.
    *   `[ ]` Define a test running the full pipeline via `cli.py` (Ingestor -> Summarizer -> Synthesizer).
    *   `[ ]` Use sample PDF(s).
    *   `[ ]` Verify expected outputs (e.g., memories created) by querying the adapter.
    *   `[ ]` Run with `max_concurrent=1` and `max_concurrent=3`.

---

## Phase 3: Refinement, API & Polish **[PLANNED]**

**Goal:** Implement the Critic agent, add an API layer, refine performance, and complete documentation.

**Tasks:**

1.  **[ ] Agent - Critic Role:**
    *   `[ ]` Create `CriticAgent` class in `src/core/agents/critic.py`.
    *   `[ ]` Implement `run_step` logic using adapter queries and `add`.
    *   `[ ]` Integrate `CriticAgent` into the pipeline.
    *   `[ ]` Add unit tests.

2.  **[ ] API Layer (FastAPI):**
    *   `[ ]` Add dependencies.
    *   `[ ]` Create `src/api.py`.
    *   `[ ]` Implement FastAPI app with `/run` and `/status` endpoints triggering the `TaskScheduler` pipeline.
    *   `[ ]` Add API tests.

3.  **[ ] Documentation & Demo:**
    *   `[ ]` Update `README.md` (main project README) significantly.
    *   `[ ]` Create `demo.py` or `run_demo.sh`.
    *   `[ ]` Consider blog-style documentation.

4.  **[ ] Performance & Polish:**
    *   `[ ]` Profiling and tuning.
    *   `[ ]` Refine scheduler timeouts.
    *   `[ ]` Ensure final CI pipeline passes reliably and quickly.
    *   `[ ]` Final code cleanup.
