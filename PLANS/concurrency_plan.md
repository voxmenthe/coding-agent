### Clarifying “Agents stay synchronous; concurrency handled at scheduler level”

1. Each agent’s `run_step()` is an ordinary, blocking Python function.  
   • It can freely call the (sync) `requests` client for OpenAI/Gemini, read/write files, etc.  
   • No `async def`/`await` inside the agent code → easier unit‑testing and debugging.

2. The **scheduler** owns the event loop.  
   • It wraps every `run_step()` in `asyncio.to_thread()` (or `run_sync_in_worker_thread` in AnyIO).  
   • The loop can therefore launch many blocking functions in parallel worker threads while still being “async” from the outside.

3. Result: you get overlapping network I/O without sprinkling `await` all over your business logic.  
   • Bugs stay local to agent logic; scheduler bugs stay local to the wrapper.

---

### Fool‑Proof Concurrency Plan

#### 1. Concurrency primitives

| Component | Library | Why |
|-----------|---------|-----|
| Event loop | `anyio` (`asyncio` under the hood) | Structured‑concurrency helpers, cancellation, timeouts |
| Thread off‑load | `anyio.to_thread.run_sync()` | Runs blocking `run_step()` in a worker thread |
| Concurrency limit | `anyio.Semaphore(max_concurrent)` | Back‑pressure so you never exceed API rate limits |
| Task group | `anyio.create_task_group()` | Auto‑cancels children on error, collects exceptions |

We isolate all async complexity behind a **Scheduler** class.

```python
# src/core/scheduler.py
import anyio, logging

log = logging.getLogger(__name__)

class TaskScheduler:
    def __init__(self, max_concurrent: int = 3, timeout_s: int = 120):
        self.sem         = anyio.Semaphore(max_concurrent)
        self.timeout_s   = timeout_s
        self._task_group = None            # created in `run()`

    async def _run_agent(self, agent):
        async with self.sem:
            try:
                log.info("Agent %s started", agent.role)
                # hard timeout so a hung LLM call doesn't freeze everything
                async with anyio.fail_after(self.timeout_s):
                    result = await anyio.to_thread.run_sync(agent.run_step)
                log.info("Agent %s finished", agent.role)
                return result
            except Exception as e:
                log.exception("Agent %s failed: %s", agent.role, e)
                raise

    async def run(self, agents: list):
        async with anyio.create_task_group() as tg:
            self._task_group = tg
            results = []
            for agent in agents:
                tg.start_soon(
                    lambda a=agent: results.append(anyio.from_thread.run_sync(lambda: a)),  # placeholder
                )
        return results
```

#### 2. Thread‑safety guarantees

| Shared resource | Guard |
|-----------------|-------|
| Vector store    | Chroma already serialises SQLite writes; thread‑safe for reads. |
| Rolling summary file | `filelock.FileLock` around append, or switch to TinyDB (thread‑safe). |
| STDOUT logging  | Python logging is thread‑safe. |

#### 3. Error propagation & cancellation

AnyIO’s task‑group gives us “fail‑fast semantics”:  
• If one agent raises, all other tasks get cancelled.  
• Exception surfaces to CLI or REST call → user sees a clear traceback.

We’ll wrap `multi_run` like:

```python
@click.command()
def multi_run():
    agents = build_agents_from_config()
    try:
        results = anyio.run(TaskScheduler(max_concurrent=cfg.n).run, agents)
    except Exception:
        click.echo("❌ Pipeline failed; see log for details.")
```

#### 4. Debugging strategy

1. Run unit tests on single agents (`run_step()` is synchronous → trivial).  
2. Integration test with `max_concurrent=1` (sequential) to get deterministic baseline.  
3. Repeat with `max_concurrent=3`; if behaviour diverges, inspect:  
   • race conditions in shared summary file (add lock)  
   • API rate‑limit errors (increase semaphore delay/back‑off)  
4. Enable `anyio` debug mode: `ANYIO_DEBUG=True python -m pytest`.

#### 5. Rate‑limit & back‑pressure

```python
class LLMLimiter:
    def __init__(self, max_rps=3):
        self.token_bucket = anyio.CapacityLimiter(max_rps)

    async def call(self, fn, *args, **kw):
        async with self.token_bucket:
            return await anyio.to_thread.run_sync(fn, *args, **kw)
```

Agents call `llm_client.call()` instead of the raw SDK; the limiter prevents burst errors.

#### 6. Flow diagram (v0.1)

```
            anyio.run
                │
        TaskScheduler.run()
                │
───────────── Task‑Group ─────────────
│      │        │            │
_a1_  _a2_     _a3_        _a4_
(to_thread)  (to_thread) ...
│      │        │
run_step()...
```

All arrows below the dashed line run in worker threads; they share the vector store and rolling summary under file locks.

---

### Why this is robust

• Agent code unchanged whether you call it directly or through scheduler → easy local repro.  
• Only one module (`scheduler.py`) touches `async` keywords → low cognitive load.  
• Structured concurrency with AnyIO gives automatic cleanup; no “dangling tasks”.  
• Hard per‑agent timeout avoids deadlocks on hung HTTP calls.  

If you’re comfortable with this design we can merge the above skeleton first, then implement the individual agents atop it.