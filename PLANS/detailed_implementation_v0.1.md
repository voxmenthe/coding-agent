# Detailed Implementation Plan (v0.1) - Multi-Agent Research Synthesizer

This plan elaborates on the 3-week roadmap outlined in `PLANS/integrated_plan.md`, providing more granular steps. It assumes the target architecture described therein, which differs significantly from the current single-agent setup in `src/main.py`.

**Status Legend:** `[X]` = Done, `[-]` = Partially Done / In Progress, `[ ]` = Not Started

---

## Week 1: Foundations - Memory & Ingestion

**Goal:** Establish the core memory infrastructure (skeleton) and the first agent responsible for data intake.

**Tasks:**

1.  **[X] Project Setup & CI:**
    *   `[X]` Confirm/Set up Python environment (using `poetry` and `.venv`).
    *   `[X]` Update `pyproject.toml` with initial dependencies (`anyio`, `sqlite3`, `sentence-transformers`, `numpy`, `filelock`, `pydantic`, `pdfplumber`, `pytest`, `ruff`, etc.). (Note: `requirements.txt` deleted in favor of `pyproject.toml`).
    *   `[ ]` Initialize `pre-commit` with hooks for `ruff` (linting + formatting).
    *   `[ ]` Set up basic GitHub Actions workflow for linting (`ruff check .`) and running tests (`pytest`).
    *   `[X]` Create the proposed directory structure and `__init__.py` files.

2.  **[X] Memory System - Core Interfaces:**
    *   `[X]` Define the `MemoryDoc` dataclass in `src/memory/adapter.py`.
        *   *Annotation:* Updated `timestamp` field's default factory to use `datetime.now(timezone.utc)` to resolve `DeprecationWarning` for `utcnow()`.
    *   `[X]` Define the `MemoryAdapter` abstract base class with `add` and `query` methods in `src/memory/adapter.py`.

3.  **[X] Memory System - Hybrid SQLite Adapter (Skeleton):**
    *   `[X]` In `src/memory/hybrid_sqlite.py`, create `HybridSQLiteAdapter` inheriting from `MemoryAdapter`.
    *   `[X]` Implement the `__init__` method (connects, calls `_setup_schema`).
    *   `[X]` Implement `_setup_schema` (creates `memories` and `memories_fts` tables).
    *   `[X]` Implement `add` method (basic version - inserts metadata and text, no embeddings yet).
    *   `[X]` Implement `query` method (basic FTS version - uses `MATCH`, returns MemoryDoc list).
    *   `[X]` Add basic unit tests in `tests/memory/test_hybrid_sqlite.py` for schema creation and basic `add`/`query`.
        *   *Annotation:* Extensive debugging was required to get FTS working reliably. The initial schema was simplified (removing timestamp, metadata, source agent from DB table for now). Various FTS synchronization methods (`content=` vs manual triggers) and tokenizer configurations were tested. The final working solution uses manual triggers (`AFTER INSERT/DELETE/UPDATE`) to sync `memories.text_content` into `memories_fts.text_content`, with the `memories_fts.text_content` column explicitly indexed (i.e., *not* `UNINDEXED`) and using the `porter` tokenizer. Tests `test_add_memory_basic` and `test_query_memory_fts` now pass consistently.

4.  **[X] Memory System - MemoryService (Singleton Skeleton):**
    *   `[X]` In `src/memory/service.py`, implement the `MemoryService` singleton pattern.
    *   `[X]` `__init__` (or `_init` called from `__new__`) should instantiate `HybridSQLiteAdapter`.
    *   `[X]` Add methods `add_memory(doc: MemoryDoc)` and `query_memory(query: str, k: int)` that delegate to the adapter instance.

5.  **[-] Agent Skeletons & CLI:**
    *   `[X]` Create `src/core/scheduler.py` with `TaskScheduler` stub.
    *   `[X]` Create `src/core/agents/base.py` with `BaseAgent` abstract class.
    *   `[X]` In `src/core/agents/ingestor.py`, create `IngestorAgent` skeleton with `pdfplumber` logic and basic chunking/memory saving.
    *   `[X]` Create `src/cli.py` with basic `click` setup and `run-pipeline` stub.
    *   `[ ]` Add basic unit tests in `tests/core/agents/test_ingestor.py` (mock `MemoryService`).

---

## Week 2: Concurrency, Embeddings & Core Agent Logic

**Goal:** Implement core concurrency, integrate semantic search via embeddings + RRF, and build out core agent logic.

**Tasks:**

1.  **[ ] Memory System - Embeddings & Semantic Search:**
    *   `[ ]` Integrate `sentence-transformers` model loading in `MemoryService` or `HybridSQLiteAdapter`.
    *   `[ ]` Update `HybridSQLiteAdapter.add` to generate embeddings for `MemoryDoc.text`.
    *   `[ ]` Save embeddings as `.npy` files in `.memory_db/embeddings/` (use `MemoryDoc.id` as filename).
    *   `[ ]` Update `HybridSQLiteAdapter.query` to perform semantic search (load relevant embeddings, compute cosine similarity against query embedding).
    *   `[ ]` Add unit tests for embedding generation and semantic query functionality.

2.  **[ ] Memory System - RRF Fusion:**
    *   `[ ]` Update `HybridSQLiteAdapter.query` to perform both FTS and semantic searches.
    *   `[ ]` Implement Reciprocal Rank Fusion (RRF) logic to combine the ranked results from FTS and semantic search.
    *   `[ ]` Add configuration for RRF weighting (e.g., `k` constant in RRF formula).
    *   `[ ]` Benchmark query performance (latency) with and without RRF.
    *   `[ ]` Add unit tests for RRF implementation.

3.  **[ ] Concurrency - Memory Writes:**
    *   `[ ]` Refactor `MemoryService.add_memory` to be an `async` method.
    *   `[ ]` Introduce an `asyncio.Queue` within `MemoryService` to serialize write operations (`add_memory` puts items onto the queue).
    *   `[ ]` Implement a single background writer coroutine that consumes from the queue and performs the actual DB/file writes.
    *   `[ ]` Add `filelock` around `.npy` file writing to prevent race conditions if multiple processes/threads were ever used (defense-in-depth).
    *   `[ ]` Add tests to verify concurrent calls to `add_memory` don't corrupt data or cause deadlocks.

4.  **[ ] Concurrency - Scheduler Integration:**
    *   `[ ]` Fully implement `TaskScheduler._run_agent` using `anyio.to_thread.run_sync` to run synchronous agent `run_step` methods in a thread pool.
    *   `[ ]` Implement `TaskScheduler.run` using `anyio.create_task_group` to manage concurrent agent runs.
    *   `[ ]` Integrate `TaskScheduler` into `cli.py`'s `run-pipeline` command to orchestrate the agents.
    *   `[ ]` Add tests for `TaskScheduler` functionality.

5.  **[ ] Agent Tools (`save_memory`, `recall_memory`):**
    *   `[ ]` Implement the `save_memory` and `recall_memory` functions in `src/tools.py`.
    *   `[ ]` `save_memory` should create a `MemoryDoc` and call `await MemoryService().add_memory(doc)`. Since agents run sync, they'll need a way to call this async function (e.g., `anyio.from_thread.run(MemoryService().add_memory, doc)`).
    *   `[ ]` `recall_memory` should call the (now RRF-enabled) `MemoryService().query_memory(...)`.
    *   `[ ]` Update `IngestorAgent` (and later others) to use these tools instead of directly calling `MemoryService`.

6.  **[ ] Core Agents (Summarizer, Synthesizer):**
    *   `[ ]` Create skeleton classes `SummarizerAgent` and `SynthesizerAgent` in `src/core/agents/`.
    *   `[ ]` Implement basic `run_step` logic for each:
        *   `Summarizer`: Use `recall_memory` to get chunks for one paper, generate a summary (mock LLM initially), use `save_memory` to store summary.
        *   `Synthesizer`: Use `recall_memory` to get summaries/chunks across papers, generate synthesis (mock LLM), use `save_memory`.
    *   `[ ]` Add basic unit tests for these agents (mocking tools/LLM).

7.  **[ ] Integration Test:**
    *   `[ ]` Create `tests/integration/test_pipeline.py`.
    *   `[ ]` Define a test that runs the full pipeline via `cli.py` (Ingestor -> Summarizer -> Synthesizer).
    *   `[ ]` Use sample PDF(s) as input.
    *   `[ ]` Verify expected outputs (e.g., certain types of memories created) using `MemoryService.query_memory`.
    *   `[ ]` Run the test with `max_concurrent=1` and `max_concurrent=3` to check basic concurrency.

---

## Week 3: Refinement, API & Polish

**Goal:** Implement the Critic agent, add an API layer, refine performance, and complete documentation.

**Tasks:**

1.  **[ ] Agent - Critic Role:**
    *   `[ ]` Create `CriticAgent` class in `src/core/agents/critic.py`.
    *   `[ ]` Implement `run_step` logic:
        *   Use `recall_memory` to fetch summaries and syntheses.
        *   Identify potential overlaps or suggest improvements (mock LLM).
        *   Use `save_memory` to store critiques/feedback, possibly with specific tags.
    *   `[ ]` Integrate `CriticAgent` into the `TaskScheduler` pipeline in `cli.py`.
    *   `[ ]` Add unit tests for `CriticAgent`.

2.  **[ ] API Layer (FastAPI):**
    *   `[ ]` Add `fastapi` and `uvicorn` dependencies to `pyproject.toml`.
    *   `[ ]` Create `src/api.py` (or similar).
    *   `[ ]` Implement a basic FastAPI application.
    *   `[ ]` Add a `/run` endpoint that accepts input parameters (e.g., PDF paths) and triggers the agent pipeline via `TaskScheduler`. This should likely run the pipeline in the background and return a task ID.
    *   `[ ]` Add a `/status/{task_id}` endpoint to check the status of a pipeline run (Pending, Running, Complete, Failed) and retrieve results. (Requires modifying `TaskScheduler` or adding a task tracking layer).
    *   `[ ]` Add basic API tests.

3.  **[ ] Documentation & Demo:**
    *   `[ ]` Update `README.md` significantly: Architecture overview, setup instructions, CLI usage examples, API endpoint documentation.
    *   `[ ]` Create a `demo.py` or `run_demo.sh` script showcasing a typical run using the CLI.
    *   `[ ]` Consider adding brief blog-style documentation explaining the hybrid memory approach and concurrency model.

4.  **[ ] Performance & Polish:**
    *   `[ ]` Profile memory and CPU usage during a typical pipeline run. Identify bottlenecks.
    *   `[ ]` Tune agent logic/prompts (if using real LLMs by now) for quality and efficiency.
    *   `[ ]` Refine scheduler timeouts (`TaskScheduler(timeout_s=...)`).
    *   `[ ]` Ensure final CI pipeline (lint, unit tests, integration tests) passes reliably and runs quickly (target < 60s).
    *   `[ ]` Perform final code cleanup and refactoring based on learnings.
