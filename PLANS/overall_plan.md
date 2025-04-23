### Product‑Requirements & Implementation Plan  
AI Multi‑Agent Research Synthesizer (v0.1 – 2‑3 week sprint)

---

#### 1. Vision & Scope
- Build an open‑source framework that lets multiple LLM‑powered agents collaboratively generate new research hypotheses and deep summaries from a user‑supplied set of papers.  
- Demo‑ready in 2–3 weeks: three agent roles running concurrently via a wrapper scheduler, CLI interface (existing), thin REST API stub, vector‑store + rolling‑summary memory.

Success metric v0.1:  
Given 2 input PDFs, system returns (a) structured per‑paper summary, (b) list of ≥3 distinct, non‑trivial research ideas, all within ~2× single‑LLM‑call latency.

---

#### 2. Key “Wow” Factors
1. Multi‑agent reasoning core with concurrency.  
2. Clean, documented codebase + tests + GitHub Actions CI.  
3. Modular memory layer (FAISS/Chroma + summary roll‑ups).  
4. Easy config: swap APIs (OpenAI, Gemini) or local models later.  

---

#### 3. High‑Level Architecture
```text
CLI / FastAPI           Human feedback (free‑text)
      │
┌──────▼───────────┐
│  Scheduler (async task‑queue wrapper)
│  • max_concurrent = cfg
└──────┬─────┬─────┘
       │     │
  ┌────▼─┐ ┌─▼────┐ ...
  │Agent │ │Agent │  each has:
  │Role A│ │Role B│  • run_step()
  └──────┘ └──────┘
       │ shared StorageAdapter
┌──────▼────────────────────────────────────┐
│ Memory Layer                              │
│ • Vector store (Chroma or FAISS or other) │
│ • Rolling summaries (jsonl)               │
└───────────────────────────────────────────┘
```

---

#### 4. Memory & Data Design
- Vector store: Chroma in‑proc SQLite (swap‑able). Keys: `doc_chunk_id`, `paper_id`, embedding.  
- Rolling summary: per‑agent JSONL, each line `{step, agent, summary}`; periodically compressed by a “Compressor” utility.  
- Metadata store: TinyDB or YAML for `paper_id → title, url, abstract`.

---

#### 5. Agent Roles for v0.1
| Role | Purpose | Key I/O fields |
|------|---------|----------------|
| Ingestor | Parse PDF → chunks, embed, store metadata | `file_path` |
| Summarizer | Produce structured summary for each paper | `paper_id` |
| Synthesizer | Cross‑paper synthesis + idea generation | `[summary_ids]` |
| Critic | Review Synth output, flag overlaps, propose improvements | `synth_id` |

Each `run_step()` returns `AgentResult(message:str, citations:list[str], artifacts:dict)`.

---

#### 6. Async Scheduler Wrapper
```python
import asyncio, anyio

class TaskScheduler:
    def __init__(self, max_concurrent=3):
        self.sem = anyio.Semaphore(max_concurrent)
        self.queue = asyncio.Queue()

    async def worker(self):
        while True:
            agent = await self.queue.get()
            async with self.sem:
                await asyncio.to_thread(agent.run_step)
            self.queue.task_done()

    def submit(self, agent):
        self.queue.put_nowait(agent)

    async def run(self):
        workers = [asyncio.create_task(self.worker()) for _ in range(self.sem.value)]
        await self.queue.join()
        for w in workers: w.cancel()
```
Agents stay synchronous; concurrency handled at scheduler level.

---

#### 7. Implementation Roadmap
##### Week 1
1. Repo hygiene: `dev-20250419` → `main`; pre‑commit, black, mypy.  
2. Add `memory/` module (vector + summary).  
3. Implement Ingestor + Summarizer roles; unit tests.  
4. Scheduler wrapper integrated; benchmark just these two roles.

##### Week 2
1. Synthesizer + Critic roles.  
2. Free‑form human feedback ingestion (store in `feedback/` YAML).  
3. Command `multi_run` in CLI: reads config YAML mapping roles & count.  
4. GitHub Actions: lint, tests, basic benchmark (<60 s).

##### Week 3
1. Thin FastAPI (`/run` & `/status`) exposing same scheduler.  
2. README “Launch in 5 min” tutorial; example demo script.  
3. Performance tuning: semaphore size vs. rate‑limit; timeout & retry.  
4. Draft blog‑style docs in `docs/`, OpenAPI schema stub.

---

#### 8. Key Code Snippets

h4. Agent base
```python
class BaseAgent:
    role = "agent"
    def __init__(self, storage, llm):
        self.storage = storage
        self.llm = llm

    def run_step(self):
        context = self._gather_context()
        prompt = self._build_prompt(context)
        response = self.llm(prompt)
        self._persist(response)
        return response
```

h4. Storage adapter (simplified)
```python
class StorageAdapter:
    def __init__(self):
        self.vs = chromadb.PersistentClient(path=".chromadb").get_or_create_collection("papers")
        self.summary_path = Path("memory/summary.jsonl")

    def add_embedding(self, doc_id, embedding, meta):
        self.vs.add([doc_id], [embedding], metadatas=[meta])

    def query(self, query_emb, k=5):
        return self.vs.query([query_emb], n_results=k)

    def append_summary(self, record: dict):
        with self.summary_path.open("a") as f:
            json.dump(record, f); f.write("\n")
```

h4. CLI enhancement
```python
@cli.command()
@click.option("--config", type=click.File("r"), default="config.yaml")
def multi_run(config):
    cfg = yaml.safe_load(config)
    scheduler = TaskScheduler(cfg["max_concurrent"])
    for role_cfg in cfg["agents"]:
        agent_cls = get_agent_class(role_cfg["role"])
        scheduler.submit(agent_cls(storage, llm, **role_cfg.get("params", {})))
    asyncio.run(scheduler.run())
```

---

#### 9. Testing & Benchmarking
- Unit: pytest‑parametrized `run_step` stubs with fake LLM.  
- Integration: mini‑pipeline on 2 toy PDFs (public domain).  
- Benchmark script captures wall‑clock vs. sequential baseline.  
- Criteria: < 1.5× sequential latency when `max_concurrent=3`.

---

#### 10. Deployment & Distribution
- `pip install coding-agent[fastapi]` extra.  
- Dockerfile (CPU): poetry install + uvicorn entrypoint.  
- GitHub README badges: CI status, coverage, PyPI version.

---

#### 11. Risk & Mitigation
| Risk | Mitigation |
|------|------------|
| Async deadlocks / task leaks | Wrapper design, timeout + cancel; `anyio.fail_after()` |
| API rate‑limits | Exponential backoff, config tokens/sec |
| Mode collapse | Diversity check in Critic; memory retrieval of prior ideas |
| Time overrun | Hard cutoff on web UI; CLI is the MVP |

---

#### 12. Future Extensions (post‑v0.1)
- Dynamic agent birth/death via Planner role.  
- Feedback‑aware fine‑tuning on cloud GPUs.  
- UI dashboard with live agent timelines (React + WebSocket).  
- Multi‑modal: plug CV encoder for figures & tables.

---

Focus next 48 hours: finish StorageAdapter + Ingestor + Scheduler skeleton; push branch `feat/async-scheduler` for review.