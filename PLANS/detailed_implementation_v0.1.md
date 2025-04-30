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
    *   `[ ]` ~~Initialize `pre-commit` with hooks for `ruff` (linting + formatting).~~
    *   `[ ]` ~~Set up basic GitHub Actions workflow for linting (`ruff check .`) and running tests (`pytest`).~~
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

6.  **[ ] Agent Skeletons & CLI: (Focus on Minimal Viable Pipeline First)**
    *   `[X]` `src/core/scheduler.py` exists (implemented in Phase 2).
    *   `[X]` `src/core/agents/base.py` exists.
    *   `[X]` `src/core/agents/ingestor.py` exists (implementation in progress - Phase 2).
    *   `[ ]` Create `src/core/agents/summarizer.py` with skeleton class (Minimal version - Phase 2).
    *   `[ ]` Create `src/core/agents/synthesizer.py` with skeleton class (Minimal version - Phase 2).
    *   `[X]` `src/cli.py` exists (integration needed - Phase 2).
    *   `[ ]` Add basic unit tests for minimal agent skeletons.

7.  **[X] Example Usage Scripts:**
    *   `[X]` Create `examples/memory/example_basic_usage.py` demonstrating init, add, basic FTS query, basic semantic query.
    *   `[X]` Create `examples/memory/example_advanced_querying.py` demonstrating adding diverse docs and using filters with both query types.
    *   `[X]` Create `examples/README.md` explaining the examples and how to run them.

---

## Phase 2: Agents & Concurrency **[IN PROGRESS - FOCUS: Minimal Pipeline]**

**Goal:** Implement *minimal functional* core agent logic (including basic LLM calls), integrate with the scheduler, and enable a basic end-to-end run via CLI.

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

4.  **[ ] Core Agents (Minimal Implementation First):**
    *   `[-]` Implement `IngestorAgent` (**Priority: Minimal Viable**):
        *   `[X]` Basic structure and `run` method defined.
        *   `[X]` Refactored to use `pymupdf` strategy internally (`_process_pdf_pymupdf`).
        *   `[ ]` **Implement basic text chunking logic** (e.g., fixed size, simple overlap) on the extracted text from `pymupdf`.
        *   `[ ]` **Integrate actual `HybridSQLiteAdapter`**: Instantiate/receive adapter instance and use `adapter.add(MemoryDoc(...))` for each chunk. Replace mocks.
        *   `[ ]` **Basic Input Handling:** Ensure agent accepts PDF path(s) and handles basic file existence checks.
        *   `[ ]` **Basic Error Handling:** Add `try...except` blocks around file processing, chunking, and adapter calls. Log errors.
        *   `[ ]` *(Deferred)* Define Configuration (`INGESTION_STRATEGY`, API keys).
        *   `[ ]` *(Deferred)* Implement `mistral_ocr` strategy.
        *   `[ ]` *(Deferred)* Implement `gemini_native` strategy.
        *   `[ ]` *(Deferred)* Implement strategy selection logic based on configuration.
        *   `[X]` Unit tests added covering pymupdf (continue to adapt tests for real adapter/chunking).
    *   `[ ]` Implement `SummarizerAgent` (**Minimal Version**):
        *   `[ ]` Define basic class structure inheriting `BaseAgent`, handling API key/client init.
        *   `[ ]` Implement `run` method: Use passed `HybridSQLiteAdapter` instance to `query/hybrid_query` for relevant chunks (e.g., based on source PDF metadata).
        *   `[ ]` **Format a simple prompt** with the retrieved chunks/text.
        *   `[ ]` **Make a basic Gemini API call** (e.g., using `genai` client, similar to `src/main.py`) to generate a summary.
        *   `[ ]` **Parse the LLM response**.
        *   `[ ]` Use `adapter.add()` to store the generated summary `MemoryDoc`.
        *   `[ ]` Add basic unit tests (mocking adapter and LLM call).
    *   `[ ]` Implement `SynthesizerAgent` (**Minimal Version**):
        *   `[ ]` Define basic class structure inheriting `BaseAgent`, handling API key/client init.
        *   `[ ]` Implement `run` method: Use adapter to query for summaries/chunks across papers.
        *   `[ ]` **Format a simple prompt** with the retrieved summaries/text.
        *   `[ ]` **Make a basic Gemini API call** to generate synthesis.
        *   `[ ]` **Parse the LLM response**.
        *   `[ ]` Use `adapter.add()` to store the generated synthesis `MemoryDoc`.
        *   `[ ]` Add basic unit tests (mocking adapter and LLM call).
    *   `[X]` Ensure agents correctly instantiate/receive `HybridSQLiteAdapter` instances (Decision made: instance per agent).

5.  **[ ] Minimal CLI Integration:**
    *   `[ ]` Update `src/cli.py`'s `run-pipeline` command (or create a new minimal command).
    *   `[ ]` Instantiate `TaskScheduler`.
    *   `[ ]` Instantiate *minimal* `IngestorAgent`, `SummarizerAgent`, `SynthesizerAgent` (passing adapter instance).
    *   `[ ]` Submit agents to the scheduler and `await scheduler.run()`.

6.  **[ ] Minimal Integration Test:**
    *   `[ ]` Create `tests/integration/test_minimal_pipeline.py`.
    *   `[ ]` Define a test running the basic pipeline via `cli.py` command (Minimal Ingestor -> Minimal Summarizer -> Minimal Synthesizer).
    *   `[ ]` Use a sample PDF.
    *   `[ ]` Verify expected minimal outputs (e.g., placeholder memories created) by querying the adapter after the run.
    *   `[ ]` Run with `max_concurrent=1`.

---

## Phase 3: Enhancements, API & Polish **[PLANNED - AFTER MINIMAL PIPELINE]**

**Goal:** Implement advanced agent logic (refined LLM usage, Critic), add an API layer, refine performance, and complete documentation.

**Tasks:**

1.  **[ ] Agent Enhancements:**
    *   `[ ]` Enhance `IngestorAgent`: Add configuration (`INGESTION_STRATEGY`, API keys), implement Mistral/Gemini strategies, add strategy selection logic.
    *   `[ ]` Enhance `SummarizerAgent` & `SynthesizerAgent`: **Refine prompts, improve error handling, potentially explore different models/parameters.**
    *   `[ ]` Add/improve unit tests for enhanced agent logic (including LLM interactions).

2.  **[ ] Agent - Critic Role:**
    *   `[ ]` Create `CriticAgent` class in `src/core/agents/critic.py`.
    *   `[ ]` Implement `run` logic using adapter queries and `add` (potentially LLM-assisted).
    *   `[ ]` Integrate `CriticAgent` into the pipeline (likely via scheduler).
    *   `[ ]` Add unit tests.

3.  **[ ] API Layer (FastAPI):**
    *   `[ ]` Add dependencies.
    *   `[ ]` Create `src/api.py`.
    *   `[ ]` Implement FastAPI app with `/run` and `/status` endpoints triggering the `TaskScheduler` pipeline.
    *   `[ ]` Add API tests.

4.  **[ ] Documentation & Demo:**
    *   `[ ]` Update `README.md` (main project README) significantly.
    *   `[ ]` Create `demo.py` or `run_demo.sh`.
    *   `[ ]` Consider blog-style documentation.

5.  **[ ] Performance & Polish:**
    *   `[ ]` Profiling and tuning.
    *   `[ ]` Refine scheduler timeouts, agent error handling.
    *   `[ ]` Benchmark concurrent vs sequential execution.
    *   `[ ]` ~~Ensure final CI pipeline passes reliably and quickly.~~
    *   `[ ]` Final code cleanup.
