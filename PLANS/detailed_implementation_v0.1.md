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

1.  **[ ] Memory System - RRF Fusion:**
    *   `[ ]` Update `HybridSQLiteAdapter` to add a new method, e.g., `hybrid_query()`.
    *   `[ ]` This method performs both FTS (`query`) and semantic (`semantic_query`) searches internally.
    *   `[ ]` Implement Reciprocal Rank Fusion (RRF) logic to combine the ranked results.
    *   `[ ]` Add configuration for RRF weighting.
    *   `[ ]` Add unit tests for RRF implementation.
    *   `[ ]` Benchmark query performance (latency).

2.  **[ ] Concurrency - Scheduler Implementation:**
    *   `[ ]` Fully implement `TaskScheduler` in `src/core/scheduler.py` based on the `anyio` design.
    *   `[ ]` Implement `_run_agent` using `anyio.to_thread.run_sync`.
    *   `[ ]` Implement `run` method using `anyio.create_task_group`.
    *   `[ ]` Add tests for `TaskScheduler` functionality.

3.  **[ ] Concurrency - Adapter Usage Review:**
    *   `[ ]` Review concurrency needs for `HybridSQLiteAdapter` when used by multiple agents via `TaskScheduler`.
    *   `[ ]` Decide if separate adapter instances per agent thread are needed, or if internal locking/queueing is required for shared instances (especially for writes).
    *   `[ ]` Add `filelock` around `.npy` file writing in `EmbeddingManager` for robustness if not already present.

4.  **[ ] Core Agents (Ingestor, Summarizer, Synthesizer):**
    *   `[ ]` Implement `IngestorAgent`: Use `pdfplumber` for parsing, chunking logic, use `HybridSQLiteAdapter.add` to store chunks.
    *   `[ ]` Implement `SummarizerAgent`: Use `HybridSQLiteAdapter.query/semantic_query` to get chunks for one paper, generate a summary (mock/real LLM), use `HybridSQLiteAdapter.add` to store summary.
    *   `[ ]` Implement `SynthesizerAgent`: Use `HybridSQLiteAdapter.query/semantic_query` to get summaries/chunks across papers, generate synthesis (mock/real LLM), use `HybridSQLiteAdapter.add`.
    *   `[ ]` Ensure agents correctly instantiate/receive `HybridSQLiteAdapter` instances.
    *   `[ ]` Add unit tests for these agents (mocking adapter/LLM).

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
