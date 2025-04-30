### Updated Comprehensive Plan (v0.1, 2‑3‑week sprint)  
Multi‑Agent Research Synthesizer  
(Concurrency‑friendly core + Hybrid/Pluggable Memory - **SQLite+Embeddings Implemented**, **RRF Implemented**, **no Chroma**)

---

#### 1. Vision & Success Criteria
*   Three collaborating agents (Summarizer, Synthesizer, Critic) run concurrently through an async wrapper and generate ≥3 distinct research ideas from two PDFs in ≤ 1.5 × single‑call latency.
*   **[DONE]** Memory layer is **Hybrid/Pluggable**: local embeddings + SQLite FTS5, with **RRF implemented** (`hybrid_query`).
*   Clean repo, docs; CLI MVP plus stub REST endpoint.

---

### 2. High‑Level Architecture
```text
                    CLI / FastAPI
                       │
           ┌───────────▼───────────┐
           │ TaskScheduler (async) │  ← **[IMPLEMENTED]** AnyIO Task‑Group, Semaphore
           └───┬──────────┬────────┘
               │          │
       ┌───────▼───┐ ┌────▼────┐
       │ Agent A   │ │ Agent B │   ... each: async run()
       └───────────┘ └─────────┘
               │ uses HybridSQLiteAdapter
┌──────────────▼─────────────────────────┐
│ Hybrid Memory Backend (SQLite + FTS5 + │
│  embeddings on FS)                     │
│  (RRF fusion **implemented**)          │
└─────────────────────────────────────────┘
```

---

### 3. Memory Sub‑System

#### 3.1 `MemoryAdapter` Interface (Conceptual)
*_(Interface not formally defined yet, but `HybridSQLiteAdapter` serves this role).* 
```python
# Conceptual - Not yet a formal ABC
class MemoryAdapter:
    def add(self, doc: "MemoryDoc") -> str: ...
    def query(self, query_text: str, k: int = 10, ...) -> list["MemoryDoc"]: ... # FTS
    def semantic_query(self, query_text: str, k: int = 10, ...) -> list["MemoryDoc"]: ... # Semantic
    # --> Add hybrid_query signature <--
    def hybrid_query(self, query_text: str, k: int = 10, ...) -> list[tuple["MemoryDoc", float]]: ... # RRF
```

#### 3.2 `HybridSQLiteAdapter` **[IMPLEMENTED]**
*   **SQLite schema** (`memories` table + `memories_fts` virtual table using FTS5). **[DONE]**
*   Embeddings managed via `EmbeddingManager`, stored as `.npy` files. **[DONE]**
*   FTS5 index on `text_content`, explicit synchronization triggers added. **[DONE]**
*   Filtering by `tags` and `source_agent` implemented for both FTS and semantic queries. **[DONE]**
*   **[DONE]** RRF implemented via `hybrid_query()` method.

#### 3.3 Concurrency Guarantees **[Decision Made]**
*   Adapter uses a single `sqlite3` connection internally with `check_same_thread=False`.
*   **Decision:** To ensure safety with the concurrent `TaskScheduler`, **each agent task will be given its own separate `HybridSQLiteAdapter` instance.** This avoids unsafe sharing of the internal connection object. SQLite's file-level locking will manage concurrency between these separate connections.
*   Embedding file writes remain a lower-priority concern regarding locking. *(Consider `filelock` later if issues arise)*.

#### 3.4 `MemoryDoc` Dataclass **[DEFINED & USED]**
```python
# Located in src/memory/adapter.py
@dataclass
class MemoryDoc:
    id: str | None = None # Auto-generated if None
    text: str | None = None # text_content in DB
    embedding: np.ndarray | None = None # Not stored directly in DB
    embedding_path: Path | str | None = None # Path to .npy file
    timestamp: datetime | None = None # Auto-generated if None
    tags: list[str] | None = None
    source_agent: str | None = None
    metadata: dict[str, Any] | None = None
    score: float | None = None # For query results
```

---

### 4. Concurrency Design (Wrapper - **Implemented**)

#### 4.1 Scheduler Skeleton
*   **[DONE]** `TaskScheduler` implemented in `src/core/scheduler.py` using `anyio`. See detailed plan Task 2.2 for implementation notes.

---

### 5. Agent Roles (v0.1 - Prioritizing Minimal Viability)

| Role | Responsibilities | Tools needed | Status | Notes |
|------|------------------|--------------|--------|-------|
| Ingestor | PDF → chunks, add to memory | Configurable: `pymupdf` (**Priority**)/Mistral/Gemini, `HybridSQLiteAdapter.add()` | `[-] In Progress` | `PyMuPDF parsing tested. **Needs basic chunking & real adapter integration ASAP**. Config integration & other strategies deferred.` |
| Summarizer | Minimal summary per paper | `HybridSQLiteAdapter.query/hybrid_query()`, **Basic LLM call (Gemini)** | TODO | `Focus on basic structure, data flow, and simple Gemini API call. Needs API key handling.` |
| Synthesizer | Minimal synthesis | Same, **Basic LLM call (Gemini)** | TODO | `Focus on basic structure, data flow, and simple Gemini API call. Needs API key handling.` |
| Critic | Detect overlap, propose fixes | Same | TODO | `Deferred until core pipeline works.` |

*   **[DONE]** `BaseAgent` defined in `src/core/agents/base.py`.
*   Agents will need access to an instance of `HybridSQLiteAdapter` (**Strategy Decided:** Each agent gets its own instance).

---

### 6. Module & File Layout **[Partially Updated]**

```
src/
 ├─ core/                 # **[Partially Implemented]**
 │   ├─ scheduler.py      # **[DONE]**
 │   └─ agents/           # **[Partially Implemented]**
 │        ├─ base.py       # **[DONE]**
 │        ├─ ingestor.py   # **[-] In Progress (Minimal)**
 │        ├─ summarizer.py # TODO (Minimal)
 │        ├─ synthesizer.py# TODO (Minimal)
 │        └─ critic.py     # TODO (Deferred)
 ├─ memory/
 │   ├─ adapter.py        # MemoryDoc definition
 │   ├─ hybrid_sqlite.py  # HybridSQLiteAdapter **[DONE]**
 │   └─ embedding_manager.py # EmbeddingManager **[DONE]**
 └─ cli.py                # TODO (or enhance existing)

examples/
 ├─ memory/
 │   ├─ example_basic_usage.py       **[DONE]**
 │   └─ example_advanced_querying.py **[DONE]**
 └─ README.md                 **[DONE]**

tests/
 └─ memory/
     └─ test_hybrid_sqlite.py    **[UPDATED]**
```

---

### 7. Key Code Snippets **[Outdated - See Examples]**
*_(Removed outdated `MemoryService` and `save/recall` snippets. Refer to `examples/memory/` scripts for current usage patterns of `HybridSQLiteAdapter`)*.

---

### 8. Implementation Road‑Map **[Updated Status & Focus]**

| Week | Deliverables | Status | Notes |
|------|--------------|--------|-------|
| **1** | • ~~Repo cleanup, pre‑commit, GH Actions lint/test~~ | *(Removed)* | |
|       | • `HybridSQLiteAdapter` Full Impl | **DONE** | |
|       | • Example scripts created & tested | **DONE** | |
| **2** | • RRF fusion completed | **DONE** | Implemented in `hybrid_query` |
|       | • Scheduler Implementation | **DONE** | Using `anyio` |
|       | • **Concurrency Review** | **DONE** | Decision made - use one adapter instance per agent task |
|       | • **Minimal Agent Implementation** (Ingestor [`pymupdf`, basic chunking, real adapter], Summarizer [**Basic LLM**], Synthesizer [**Basic LLM**]) | `[-] In Progress` | **HIGHEST PRIORITY** |
|       | • **Minimal CLI Integration** (Run basic pipeline) | TODO | **HIGH PRIORITY** |
|       | • **Minimal Integration Test** (Basic pipeline flow) | TODO | **HIGH PRIORITY** |
| **3** | • *Enhance Agents:* Ingestor (config, other strategies), Summarizer/Synthesizer (**Refined LLM Prompts/Logic**) | TODO | *After minimal pipeline* |
|       | • Critic role implementation | TODO | *After minimal pipeline* |
|       | • Thin FastAPI (`/run`, `/status`) | TODO | *Lower priority* |
|       | • README demo script, blog‑style docs | TODO | Examples README done |
|       | • Performance / rate‑limit tuning, timeouts | TODO | *Lower priority* |
|       | • ~~Final CI: unit, integration, benchmark < 60 s~~ | *(Removed)* | |

--- 

### 9. Testing & Benchmarks **[Partially Updated]**
*   Unit Tests: `pytest` tests for `HybridSQLiteAdapter` (**including RRF**) exist. **[DONE]**. `TaskScheduler` tests exist. **[DONE]**. Need tests for minimal agents.
*   Integration: Need **basic end-to-end test** for minimal pipeline first.
*   Examples: Standalone example scripts serve as basic integration/usage tests. **[DONE]**
*   Benchmark: Script needed to compare concurrent vs. sequential. RRF benchmark *not* yet done.

---

### 10. Next Steps (Immediate Focus - Minimal Working System)
1.  **Minimal Agent Implementation:** Complete the `Ingestor` agent (using `pymupdf`, adding basic chunking, integrating the real `HybridSQLiteAdapter`). Create `Summarizer` and `Synthesizer` agents **using basic Gemini LLM calls** (reference `src/main.py`) for their core logic. Ensure each agent initializes/receives its **own** `HybridSQLiteAdapter` instance and necessary API keys/client config.
2.  **Minimal CLI & Integration:** Update `cli.py` to instantiate the scheduler and minimal agents (including LLM config), running a basic pipeline. Add a simple integration test (`tests/integration/test_minimal_pipeline.py`) to verify this flow.
3.  *(Deferred)* Enhance Agents: Implement advanced ingestion strategies, refine LLM prompts/error handling, add Critic agent.
4.  *(Deferred)* API Layer & Polish: Implement FastAPI, improve docs, performance tuning.