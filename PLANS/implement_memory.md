# Revised Plan: Implementing Multi-Agent Memory System

This plan outlines the implementation of a persistent, shared memory system designed for a multi-agent architecture. It replaces the previous single-agent, ChromaDB-based approach with a custom solution leveraging local embeddings via `sentence-transformers` and a hybrid retrieval mechanism based on Reciprocal Rank Fusion (RRF).

## 1. Goals

*   **Shared Context:** Establish a persistent, shared long-term memory accessible by multiple collaborating agents.
*   **Enhanced Collaboration:** Enable agents to build upon each other's findings, decisions, and generated artifacts stored in memory.
*   **Relevant Recall:** Implement sophisticated hybrid retrieval (semantic + lexical) to surface the most relevant memories for a given query.
*   **Local & Customizable:** Utilize local embedding generation and a custom retrieval pipeline for greater control and potential offline capability.

## 2. Architectural Overview

*   **Multi-Agent Model:** This plan assumes a system where multiple agent instances (potentially running as separate processes or threads) collaborate. The exact orchestration (e.g., coordinator/specialists, peer-to-peer) is managed outside the memory system, but the memory system *must* support shared access.
*   **Shared Memory Service:** A central `MemoryService` will manage the storage, indexing, embedding, and retrieval logic. Agent instances will interact with this service (or dedicated tools wrapping it).
*   **Memory Types:**
    *   **Agent Working Memory (Implicit):** Each agent instance might maintain its own short-term context (e.g., current conversation history within its own `ChatSession` if using a similar model as `main.py`). This is *not* managed by the shared `MemoryService`.
    *   **Shared Long-Term Memory (Explicit):** Managed by the `MemoryService`, storing significant information, code, decisions, and summaries explicitly saved by agents.

## 3. Memory Storage Backend

*   **Technology Stack:**
    *   **Metadata & Text:** SQLite database for storing memory metadata and the textual content. SQLite provides transactional integrity and standard SQL querying capabilities.
    *   **Embeddings:** Embeddings will be stored as separate binary files (e.g., NumPy `.npy` format) on the filesystem, referenced by the SQLite database. This avoids bloating the database and allows for efficient loading.
    *   **Lexical Index:** SQLite's FTS5 (Full-Text Search) extension will be used to create an efficient index over the textual content for keyword-based retrieval.
*   **Database Schema (`memory.db`):**
    *   `memories` table:
        *   `id` (TEXT, PRIMARY KEY): Unique identifier (e.g., UUID).
        *   `text_content` (TEXT): The textual content of the memory.
        *   `embedding_path` (TEXT): Relative path to the `.npy` file containing the vector embedding.
        *   `timestamp` (TEXT): ISO 8601 timestamp of creation.
        *   `source_agent_id` (TEXT): Identifier for the agent that created the memory.
        *   `tags` (TEXT): JSON-encoded list of string tags for filtering.
        *   `metadata_json` (TEXT): JSON-encoded dictionary for other arbitrary metadata.
    *   `memories_fts` table:
        *   A virtual FTS5 table indexing the `text_content` column of the `memories` table.
*   **Filesystem Structure:**
    *   `.memory_db/`
        *   `memory.db` (SQLite database file)
        *   `embeddings/`
            *   `{id_1}.npy`
            *   `{id_2}.npy`
            *   ...
*   **Concurrency:** The `MemoryService` accessing SQLite must handle concurrent write access if multiple agent processes interact with it simultaneously. Using appropriate transaction management and potentially connection pooling within the service is necessary. Reads are generally less problematic but should still use separate connections per thread/process if needed.

## 4. Embedding Strategy

*   **Library:** `sentence-transformers` (Python library).
*   **Model:** Utilize a pre-trained model suitable for semantic search. A good starting point is `all-MiniLM-L6-v2` (fast, good performance) or `multi-qa-mpnet-base-dot-v1` (often better for Q&A/search tasks). The specific model should be configurable.
*   **Process:** When a memory is saved, the `MemoryService` will:
    1.  Take the `text_content`.
    2.  Use the loaded `sentence-transformers` model to generate its vector embedding.
    3.  Save the embedding vector to a `.npy` file in the `embeddings/` directory.
    4.  Store the path to this file in the `embedding_path` column of the `memories` table.

## 5. Hybrid Retrieval Strategy (RRF)

Retrieval will combine results from semantic and lexical searches using Reciprocal Rank Fusion (RRF) for robust ranking.

*   **Semantic Search Component:**
    1.  Embed the input `query` using the same `sentence-transformers` model.
    2.  Load relevant embedding vectors from the `.npy` files (potentially filtering by tags/agent IDs first by querying SQLite).
    3.  Calculate cosine similarity between the query embedding and the loaded memory embeddings.
    4.  Rank memories based on similarity scores (highest first).
    5.  *Optimization Note:* For very large memory stores, pre-loading embeddings or using an approximate nearest neighbor (ANN) index library like `faiss-cpu` could be considered in future phases.
*   **Lexical Search Component:**
    1.  Use the input `query` against the `memories_fts` table using SQLite's `MATCH` operator.
    2.  SQLite's FTS5 provides its own relevance ranking (based on BM25 implicitly).
    3.  Retrieve the ranked list of memory `id`s from the FTS query.
*   **Reciprocal Rank Fusion (RRF):**
    1.  Obtain the two ranked lists of memory `id`s (one from semantic, one from lexical search).
    2.  For each memory `id` appearing in either list, calculate its RRF score:
        `RRF_Score(id) = (1 / (k + rank_semantic(id))) + (1 / (k + rank_lexical(id)))`
        *   `rank_semantic(id)` is the rank in the semantic results (or infinity if not present).
        *   `rank_lexical(id)` is the rank in the lexical results (or infinity if not present).
        *   `k` is a constant damping factor, typically set to 60, to mitigate the impact of high ranks.
    3.  Re-rank all retrieved memory `id`s based on their combined `RRF_Score` (highest first).
    4.  Retrieve the full memory details (text, metadata) for the top `top_k` results from the `memories` table based on the final RRF ranking.

## 6. Agent Integration & Tools

*   **`MemoryService` Class (`src/memory_system.py`):**
    *   Encapsulates all logic: SQLite connection, FTS setup, embedding model loading, saving memories (text + embedding + metadata), performing hybrid RRF retrieval.
    *   Provides methods like `initialize()`, `add_memory()`, `retrieve_memories()`.
*   **Agent Tools (`src/tools.py`):**
    *   `save_memory(content: str, tags: list[str] = None, source_agent_id: str) -> str`:
        *   A thin wrapper calling `MemoryService.add_memory()`.
        *   Requires `source_agent_id` to track provenance.
        *   Returns confirmation/status.
    *   `recall_memory(query: str, top_k: int = 5, filter_tags: list[str] = None, filter_agent_ids: list[str] = None) -> str`:
        *   A thin wrapper calling `MemoryService.retrieve_memories()`.
        *   Implements the hybrid RRF logic via the service.
        *   Allows filtering by tags and/or source agents.
        *   Formats results (e.g., list of strings, JSON) for the calling agent.
*   **Integration (`src/main.py` or Agent Core):**
    *   A single instance of `MemoryService` should likely be initialized and made accessible to all agent instances (e.g., passed during agent initialization, accessed via a singleton pattern, or provided by an orchestrator).
    *   The `save_memory` and `recall_memory` tools need to be registered with each agent instance that requires memory access.

## 7. File Structure Changes

*   **New File:** `src/memory_system.py` (Contains `MemoryService` class and related logic).
*   **Modify:** `src/tools.py` (Add `save_memory`, `recall_memory` tool definitions, import `MemoryService`).
*   **Modify:** `src/main.py` (or agent initialization logic): Import and initialize `MemoryService`, provide access to agents, register new tools).
*   **New Directory:** `.memory_db/` (Created at runtime, added to `.gitignore`).

## 8. Phased Rollout

1.  **Phase 1: Core Memory Service & Tools:** Implement `MemoryService` with SQLite/FTS5/filesystem storage, `sentence-transformers` embedding, basic semantic *and* lexical search (separately), and integrate `save_memory` / `recall_memory` tools (initially returning combined *unfused* results or just semantic results). Ensure multi-agent access to the service works.
2.  **Phase 2: Hybrid Retrieval (RRF):** Implement the Reciprocal Rank Fusion logic within `MemoryService.retrieve_memories()` to combine semantic and lexical results effectively. Refine tool output.
3.  **Phase 3: Advanced Features & Optimization:** Consider automatic memory summarization/synthesis (adapting the "Inception" idea), query expansion, reranking models, performance optimizations (e.g., FAISS for large scale), and more sophisticated concurrency controls if needed.

## 9. Dependencies

*   **Add:**
    *   `sentence-transformers`
    *   `numpy`
    *   `sqlite3` (standard library, but confirm usage)
*   **Remove:** `chromadb` (if previously added).
*   **Update:** `requirements.txt` accordingly.

## 10. Future Considerations

*   **Scalability:** SQLite performance might degrade with millions of entries. Consider migrating to PostgreSQL or a dedicated vector database if scale demands it. Faiss index for semantic search becomes more critical at scale.
*   **Memory Management:** Implement strategies for memory pruning, updating, or archiving old/irrelevant entries.
*   **User Interface:** Potentially add ways for users to directly inspect or manage the agent memory.