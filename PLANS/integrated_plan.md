### Updated Comprehensive Plan (v0.1, 2‑3‑week sprint)  
Multi‑Agent Research Synthesizer  
(Concurrency‑friendly core + Hybrid/Pluggable Memory, **no Chroma**)

---

#### 1. Vision & Success Criteria
* Three collaborating agents (Summarizer, Synthesizer, Critic) run concurrently through an async wrapper and generate ≥3 distinct research ideas from two PDFs in ≤ 1.5 × single‑call latency.  
* Memory layer is **Hybrid/Pluggable**: local embeddings + SQLite FTS5 + Reciprocal Rank Fusion (RRF).  
* Clean repo, CI, docs; CLI MVP plus stub REST endpoint.

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
               │ shared MemoryService (singleton)
┌──────────────▼─────────────────────────┐
│ Hybrid Memory Backend (SQLite + FTS5   │
│  embeddings on FS, RRF fusion)         │
└─────────────────────────────────────────┘
```

---

### 3. Memory Sub‑System

#### 3.1 MemoryAdapter Interface
```python
class MemoryAdapter(ABC):
    @abstractmethod
    def add(self, doc: "MemoryDoc") -> str: ...
    @abstractmethod
    def query(self, query: str, *, k: int = 10,
              tags: list[str] | None = None,
              source_agents: list[str] | None = None) -> list["MemoryDoc"]: ...
```

#### 3.2 HybridSQLiteAdapter  
* **SQLite schema** (`memories` + `memories_fts`).  
* Embeddings: `.memory_db/embeddings/{uuid}.npy`.  
* FTS5 index on `text_content`.  
* RRF implementation combines:  
  • cosine‑similarity ranks (semantic)  
  • FTS BM25 ranks (lexical).  

#### 3.3 Concurrency Guarantees
* **One writer coroutine** inside `MemoryService`; writes are funneled through an `asyncio.Queue` to avoid database contention.  
* Reads: each thread/worker opens its own SQLite connection (`check_same_thread=False`).  
* File‑level `filelock` around `.npy` write.

#### 3.4 MemoryDoc Dataclass
```python
@dataclass
class MemoryDoc:
    id: str
    text: str
    tags: list[str]
    source_agent: str
    timestamp: datetime
    meta: dict[str, Any] = field(default_factory=dict)
```

---

### 4. Concurrency Design (unchanged wrapper)

#### 4.1 Scheduler Skeleton
```python
class TaskScheduler:
    def __init__(self, max_concurrent=3, timeout_s=120):
        self.sem  = anyio.Semaphore(max_concurrent)
        self.to   = timeout_s
        self.wq   = asyncio.Queue()

    async def _run_agent(self, agent):
        async with self.sem:
            async with anyio.fail_after(self.to):
                return await anyio.to_thread.run_sync(agent.run_step)

    async def run(self, agents):
        async with anyio.create_task_group() as tg:
            for a in agents:
                tg.start_soon(self._run_agent, a)
```

---

### 5. Agent Roles (v0.1)

| Role | Responsibilities | Tools used |
|------|------------------|-----------|
| Ingestor | PDF → chunks, add to memory | `pdfplumber`, `MemoryService.add()` |
| Summarizer | Summaries per paper | `MemoryService.query()`, LLM |
| Synthesizer | Cross‑paper ideas | Same |
| Critic | Detect overlap, propose fixes | Same |

All agents call `save_memory()` at the end of each `run_step()`.

---

### 6. Module & File Layout

```
src/
 ├─ core/
 │   ├─ scheduler.py
 │   └─ agents/
 │        ├─ base.py
 │        ├─ ingestor.py
 │        ├─ summarizer.py
 │        ├─ synthesizer.py
 │        └─ critic.py
 ├─ memory/
 │   ├─ adapter.py          # interface
 │   ├─ hybrid_sqlite.py    # HybridSQLiteAdapter
 │   └─ service.py          # MemoryService singleton
 ├─ tools.py                # save_memory / recall_memory wrappers
 └─ cli.py                  # click group
```

---

### 7. Key Code Snippets

#### 7.1 MemoryService (excerpt)
```python
class MemoryService:
    _instance: "MemoryService" | None = None

    def __new__(cls, db_path=".memory_db/memory.db"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(db_path)
        return cls._instance

    def _init(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn_pool = sqlite3.connect(self.db_path, check_same_thread=False)
        self.emb_dir   = self.db_path.parent / "embeddings"
        self.emb_dir.mkdir(exist_ok=True)
        self.model = SentenceTransformer(os.getenv("EMBED_MODEL","all-MiniLM-L6-v2"))
        self._setup_schema()

    def add_memory(self, doc: MemoryDoc):
        emb = self.model.encode(doc.text)
        np.save(self.emb_dir / f"{doc.id}.npy", emb)
        with self.conn_pool:
            self.conn_pool.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?)",
                (doc.id, doc.text, str(self.emb_dir / f"{doc.id}.npy"),
                 doc.timestamp.isoformat(), doc.source_agent,
                 json.dumps({"tags": doc.tags, **doc.meta}))
            )
            self.conn_pool.execute(
                "INSERT INTO memories_fts(rowid, text_content) VALUES (?,?)",
                (doc.id, doc.text)
            )

    def rrf_query(self, query:str, k:int=10):
        ...
```

#### 7.2 save_memory / recall_memory Tools
```python
def save_memory(content:str, tags:list[str]|None=None, source_agent_id:str=""):
    doc = MemoryDoc(id=str(uuid4()), text=content,
                    tags=tags or [], source_agent=source_agent_id,
                    timestamp=datetime.utcnow())
    MemoryService().add_memory(doc)
    return "✅ memory saved"

def recall_memory(query:str, top_k:int=5, filter_tags=None, filter_agent_ids=None):
    docs = MemoryService().rrf_query(query, k=top_k)
    # simple JSON response
    return json.dumps([asdict(d) for d in docs], indent=2)
```

---

### 8. Implementation Road‑Map

| Week | Deliverables |
|------|--------------|
| **1** | • Repo cleanup, pre‑commit, GH Actions lint/test<br>• `MemoryAdapter` + `HybridSQLiteAdapter` skeleton<br>• `MemoryService` with add/query (semantic only)<br>• Ingestor agent + unit tests |
| **2** | • RRF fusion completed & benchmarked<br>• `save_memory` / `recall_memory` tools wired into agents<br>• Scheduler integrated, Summarizer & Synthesizer roles<br>• Single‑writer coroutine + locking proof<br>• Integration test (max_concurrent=1 & 3) |
| **3** | • Critic role, free‑form feedback persistence<br>• Thin FastAPI (`/run`, `/status`)<br>• README demo script, blog‑style docs<br>• Performance / rate‑limit tuning, timeouts<br>• Final CI: unit, integration, benchmark < 60 s |

---

### 9. Testing & Benchmarks

* **Unit:**  
  • MemoryService: add + query returns correct doc count.  
  • RRF: fused ranking > individual ranks (toy data).

* **Integration:**  
  `pytest -m pipeline` loads 2 PDFs, runs 3 agents concurrently, asserts ≥3 unique ideas.

* **Benchmark:**  
  Sequential vs. `max_concurrent=3`; pass if `t_async ≤ 1.5 × t_seq`.

---

### 10. Dependencies

```
sentence-transformers
numpy
anyio
click
pdfplumber
fastapi[all]  # optional extra
filelock
```
(`chromadb` removed.)

---

### 11. Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| SQLite write contention | Single writer coroutine; WAL mode |
| Model load per thread | Global singleton; warm‑up at service init |
| Large memory (>100 k) | Phase‑3: switch to FAISS index |
| API rate limits | AnyIO token bucket; exponential backoff |

---

### 12. Deliverable Recap

* `src/memory/` with hybrid backend, RRF.  
* Concurrency wrapper (AnyIO), max concurrent configurable.  
* Three agents integrated, saving & recalling memories.  
* CLI command `coding-agent multi_run --config demo.yaml`.  
* README + example session GIF (optional) for “wow” factor.

---

This consolidated plan drops Chroma entirely, adopts the uploaded hybrid memory design, and retains the proven concurrency wrapper so we can still ship a polished, demo‑ready v0.1 within 3 weeks.