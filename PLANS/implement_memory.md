# Plan: Implementing Memory for the Coding Agent

This plan outlines the steps to implement a memory system for the coding agent, inspired by the concepts in `REFERENCE_INFO/memory_for_agents.md` but adapted for a single-agent architecture based on `src/main.py`.

## 1. Goal

Implement a persistent memory system that allows the agent to:
*   Retain key information, decisions, code snippets, and context across multiple interactions within a session and potentially across sessions.
*   Recall relevant past information to improve context awareness, reduce redundancy, and enhance task continuity.
*   Adapt the "Inception Memory" concept for synthesizing and structuring memories for better recall.

## 2. Memory Types

*   **Short-Term Memory (Implicit):** Continue using the `google-genai` SDK's `ChatSession` history for immediate turn-by-turn conversational context. This requires no change.
*   **Long-Term Memory (Explicit & Persistent):** Implement a dedicated system for storing and retrieving structured or summarized information identified as important during interactions.

## 3. Core Components & Implementation Strategy

We will implement an explicit long-term memory system using a persistent vector database, allowing for semantic retrieval.

*   **Memory Storage:**
    *   **Technology:** Use `chromadb` as a local, persistent vector database. It's mentioned in the reference document and is relatively easy to set up.
    *   **Location:** Store the database files within the project, perhaps in a `.memory_db/` directory (add to `.gitignore`).
    *   **Schema:** Each memory entry in the Chroma collection should include:
        *   `id`: Unique identifier (e.g., UUID).
        *   `document`: The textual content of the memory.
        *   `embedding`: The vector embedding of the document (handled by Chroma).
        *   `metadata`: A dictionary containing:
            *   `timestamp`: ISO 8601 timestamp of when the memory was created.
            *   `source`: Indication of how the memory was created (e.g., 'user_command', 'agent_summary', 'tool_output').
            *   `tags`: Optional list of keywords/tags for filtering.
            *   `interaction_id`: (Optional) Link back to the specific interaction/turn if applicable.

*   **Memory Creation/Ingestion:**
    *   **Approach:** Start with agent-driven explicit saving via a dedicated tool. Automatic summarization can be a future enhancement.
    *   **Tool:** Create a new tool `save_memory(content: str, tags: list[str] = None) -> str`:
        *   Takes the text content and optional tags.
        *   Generates an embedding for the content (using an appropriate model, possibly the same one used by Gemini or a dedicated embedding model).
        *   Adds the content, embedding, and metadata (timestamp, source='agent_tool', tags) to the ChromaDB collection.
        *   Returns a confirmation message.

*   **Memory Retrieval:**
    *   **Approach:** Implement agent-driven retrieval via a dedicated tool, using semantic similarity.
    *   **Tool:** Create a new tool `recall_memory(query: str, top_k: int = 3, filter_tags: list[str] = None) -> str`:
        *   Takes a search query, the number of results (`top_k`), and optional tags for filtering.
        *   Generates an embedding for the query.
        *   Performs a similarity search against the ChromaDB collection, potentially applying metadata filters based on `filter_tags`.
        *   Formats the retrieved `documents` (and maybe relevant metadata like timestamps) into a string.
        *   Returns the formatted string of relevant memories or a "No relevant memories found" message.

*   **Integration with `CodeAgent` (`src/main.py`):**
    *   **Initialization:** In `CodeAgent.__init__`, initialize the ChromaDB client and get/create the memory collection.
    *   **Tool Registration:** Add the new `save_memory` and `recall_memory` functions (defined likely in `src/memory_system.py` and imported into `src/tools.py`) to the `self.tool_functions` list.
    *   **Workflow:** The agent will decide when to use `save_memory` (e.g., after generating important code, receiving critical instructions, or summarizing a sub-task) and `recall_memory` (e.g., when starting a new related task, or when the user's query suggests needing past context). The retrieved memories will be available in the context for the *next* generation step after the tool call.
    *   **Automatic Retrieval (Phase 2):** In a later phase, consider automatically calling `recall_memory` based on the user's input *before* the main `send_message` call and injecting the results directly into the prompt sent to the LLM, perhaps under a specific header like `## Relevant Previous Context:`.

*   **"Inception" Adaptation:**
    *   The "Inception Memory Agent" concept (processing raw short-term memory into structured long-term memory) is complex. For this single-agent setup, we simplify:
        *   **Phase 1:** Rely on the agent explicitly calling `save_memory` with meaningful content.
        *   **Phase 3 (Future):** Introduce a `summarize_and_save_memory` tool that takes recent conversation history, uses the LLM to summarize key points/decisions/code, and then calls `save_memory` with the summary. This mimics the synthesis step.

## 4. File Structure Changes

*   **New File:** `src/memory_system.py`
    *   Will contain functions for:
        *   `initialize_memory()`: Sets up ChromaDB client and collection.
        *   `add_memory_entry()`: Adds an entry to the DB.
        *   `search_memory()`: Queries the DB.
*   **Modify:** `src/tools.py`
    *   Import functions from `src/memory_system.py`.
    *   Define the `save_memory` and `recall_memory` tool functions, wrapping the underlying `memory_system` functions and adding necessary docstrings/type hints for the agent.
*   **Modify:** `src/main.py`
    *   Import `initialize_memory` from `src/memory_system.py`.
    *   Call `initialize_memory` in `CodeAgent.__init__`.
    *   Ensure the new tools from `tools.py` are included in the agent's tool list.

## 5. Phased Rollout

1.  **Phase 1: Manual Memory Tools:** Implement `chromadb`, `save_memory`, `recall_memory`, and integrate them as tools accessible by the agent. Focus on the agent *choosing* to use them.
2.  **Phase 2: Automatic Retrieval:** Implement automatic `recall_memory` before prompt generation and develop a prompt injection strategy.
3.  **Phase 3: Automatic Summarization/Ingestion:** Implement a `summarize_and_save_memory` tool or mechanism for automated memory creation based on conversation history.

## 6. Dependencies

*   Add `chromadb` and potentially an embedding model library (if not using Gemini's embedding capabilities directly via an API, though `google-generativeai` likely has embedding functions) to project dependencies (e.g., `requirements.txt`).