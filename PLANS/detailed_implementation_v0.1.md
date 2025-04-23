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
    *   `[X]` Define the `MemoryAdapter` abstract base class with `add` and `query` methods in `src/memory/adapter.py`.

3.  **[-] Memory System - Hybrid SQLite Adapter (Skeleton):**
    *   `[X]` In `src/memory/hybrid_sqlite.py`, create `HybridSQLiteAdapter` inheriting from `MemoryAdapter`.
    *   `[X]` Implement the `__init__` method (connects, calls `_setup_schema`).
    *   `[X]` Implement `_setup_schema` (creates `memories` and `memories_fts` tables).
    *   `[X]` Implement `add` method (basic version - inserts metadata and text, no embeddings yet).
    *   `[X]` Implement `query` method (basic FTS version - uses `MATCH`, returns MemoryDoc list).
    *   `[ ]` Add basic unit tests in `tests/memory/test_hybrid_sqlite.py` for schema creation and basic `add`/`query`. **<-- NEXT STEP**

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
    *   Integrate `sentence-transformers` into `HybridSQLiteAdapter`.
{{ ... }}

5.  **[ ] Agent Tools (`save_memory`, `recall_memory`):**
    *   Implement the `save_memory` and `recall_memory` functions in `src/tools.py` as wrappers around `MemoryService().add_memory` and `MemoryService().query_memory`.
    *   *Decision refinement:* The `MemoryService.add_memory` method will become async later (Week 2, Task 3). The tools calling it will need to adapt. For now, agents call sync tools, which call sync service methods. This will be refactored in Week 2.

6.  **[ ] Core Agents (Summarizer, Synthesizer):**
    *   Create skeleton classes `SummarizerAgent` and `SynthesizerAgent` in `src/core/agents/`.
{{ ... }}
