### Updated Comprehensive Plan (v0.1, 2‑3‑week sprint)  
Multi‑Agent Research Synthesizer  
(Concurrency‑friendly core + Hybrid/Pluggable Memory - **SQLite+Embeddings Implemented**, **no Chroma**)

---

#### 1. Vision & Success Criteria
*   Three collaborating agents (Summarizer, Synthesizer, Critic) run concurrently through an async wrapper and generate ≥3 distinct research ideas from two PDFs in ≤ 1.5 × single‑call latency.
*   **[DONE]** Memory layer is **Hybrid/Pluggable**: local embeddings + SQLite FTS5. *(RRF planned)*.
*   Clean repo, CI, docs; CLI MVP plus stub REST endpoint.

---

### 2. High‑Level Architecture
```text
                    CLI / FastAPI
                       │
           ┌───────────▼───────────┐
           │ TaskScheduler (async) │  ← AnyIO Task‑Group, thread pool
           └───┬──────────┬────────┘
               │          │
       ┌───────▼───┐ ┌────▼────┐
       │ Agent A   │ │ Agent B │   ... each: sync run_step()
       └───────────┘ └─────────┘
               │ uses HybridSQLiteAdapter
┌──────────────▼─────────────────────────┐
│ Hybrid Memory Backend (SQLite + FTS5 + │
│  embeddings on FS)                     │
│  (RRF fusion planned)                  │
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
    def query(self, query_text: str, k: int = 10, filter_tags: list[str] | None = None, filter_source_agents: list[str] | None = None) -> list["MemoryDoc"]: ... # FTS
    def semantic_query(self, query_text: str, k: int = 10, filter_tags: list[str] | None = None, filter_source_agents: list[str] | None = None) -> list["MemoryDoc"]: ... # Semantic
```

#### 3.2 `HybridSQLiteAdapter` **[IMPLEMENTED]**
*   **SQLite schema** (`memories` table + `memories_fts` virtual table using FTS5). **[DONE]**
*   Embeddings managed via `EmbeddingManager`, stored as `.npy` files (e.g., `.memory_db/embeddings/{uuid}.npy`). **[DONE]**
*   FTS5 index on `text_content`, explicit synchronization triggers added. **[DONE]**
*   Filtering by `tags` and `source_agent` implemented for both FTS and semantic queries. **[DONE]**
*   RRF planned for future combination of FTS and semantic results.

#### 3.3 Concurrency Guarantees
*   Adapter uses a single `sqlite3` connection internally.
*   For multi-threaded/process use, consider separate adapter instances per worker or implement connection pooling/queueing if needed.
*   Embedding file writes are currently not explicitly locked (potential race condition if multiple agents add the *exact* same doc simultaneously, though UUIDs make this unlikely). *(Consider adding `filelock`)*.

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

### 4. Concurrency Design (Wrapper - Not Implemented Yet)

#### 4.1 Scheduler Skeleton
```python
# Proposed design - Needs implementation
class TaskScheduler:
    # ... (design seems reasonable, uses AnyIO)
```

---

### 5. Agent Roles (v0.1 - Not Implemented Yet)

| Role | Responsibilities | Tools needed | Status |
|------|------------------|--------------|--------|
| Ingestor | PDF → chunks, add to memory | `pdfplumber`, `HybridSQLiteAdapter.add()` | TODO |
| Summarizer | Summaries per paper | `HybridSQLiteAdapter.query/semantic_query()`, LLM | TODO |
| Synthesizer | Cross‑paper ideas | Same | TODO |
| Critic | Detect overlap, propose fixes | Same | TODO |

*Agents will need access to an instance of `HybridSQLiteAdapter`.*

---

### 6. Module & File Layout **[Partially Updated]**

```
src/
 ├─ core/                 # TODO
 │   ├─ scheduler.py      # TODO
 │   └─ agents/           # TODO
 │        ├─ base.py
 │        ├─ ingestor.py
 │        ├─ summarizer.py
 │        ├─ synthesizer.py
 │        └─ critic.py
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

### 8. Implementation Road‑Map **[Updated Status]**

| Week | Deliverables | Status |
|------|--------------|--------|
| **1** | • Repo cleanup, pre‑commit, GH Actions lint/test | *(Assumed Mostly Done)*<br>• ~~`MemoryAdapter` + `HybridSQLiteAdapter` skeleton~~ | **DONE** (Full adapter)<br>• ~~`MemoryService` with add/query (semantic only)~~ | **DONE** (Via Adapter)<br>• Ingestor agent + unit tests | TODO |
| **2** | • RRF fusion completed & benchmarked | TODO<br>• ~~`save_memory` / `recall_memory` tools wired into agents~~ | (Use adapter directly) TODO<br>• Scheduler integrated, Summarizer & Synthesizer roles | TODO<br>• ~~Single‑writer coroutine + locking proof~~ | (Review concurrency needs) TODO<br>• Integration test (max_concurrent=1 & 3) | TODO<br>• **[NEW]** Example scripts created & tested | **DONE** |
| **3** | • Critic role, free‑form feedback persistence | TODO<br>• Thin FastAPI (`/run`, `/status`) | TODO<br>• README demo script, blog‑style docs | TODO (Examples README done)<br>• Performance / rate‑limit tuning, timeouts | TODO<br>• Final CI: unit, integration, benchmark < 60 s | TODO |

--- 

### 9. Testing & Benchmarks **[Partially Updated]**
*   Unit Tests: `pytest` tests for `HybridSQLiteAdapter` exist and cover core functionality including filtering. **[DONE]** Need tests for agents.
*   Integration: Need end-to-end tests with scheduler and agents.
*   Examples: Standalone example scripts serve as basic integration/usage tests. **[DONE]**
*   Benchmark: Script needed to compare concurrent vs. sequential. RRF benchmark needed.

---

### 10. Next Steps (Immediate Focus)
1.  **Agent Implementation:** Start implementing the agent roles (`Ingestor`, `Summarizer`, `Synthesizer`, `Critic`), using the `HybridSQLiteAdapter` for memory operations.
2.  **Scheduler Implementation:** Implement the `TaskScheduler` based on the proposed design using `anyio`.
3.  **Integration:** Integrate the agents with the scheduler.
4.  **Concurrency Review:** Re-evaluate concurrency needs for the adapter (locking, connection handling) once agents are running concurrently.
5.  **(Optional) RRF:** Implement Reciprocal Rank Fusion in the adapter for combined query results.